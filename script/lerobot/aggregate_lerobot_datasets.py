"""
Merge multiple LeRobot datasets (each produced by convert_robotwin_to_lerobot.py,
one per RoboTwin task folder) into a single combined LeRobot dataset.

This is a plain-sequential-Python reimplementation of the AggregateDatasets /
validate_all_metadata pattern from convert_rel_to_abs_libero/libero_h5.py and
libero_utils/lerobot_utils.py, without the datatrove/ray dependency -- current
RoboTwin scale (a handful of task folders x tens of episodes) doesn't need
distributed execution.

Note: merged `total_tasks` is NOT simply the sum across inputs -- identical
instruction strings appearing in multiple source datasets are deduplicated to
a single task_index in the aggregated dataset.

Usage:
    python script/lerobot/aggregate_lerobot_datasets.py \\
        --input-dirs /tmp/lerobot_out/grab_roller /tmp/lerobot_out/move_stapler_pub \\
        --output-dir /tmp/lerobot_out/aggregated
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot_compat import import_lerobot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_all_metadata(all_metadata):
    """Ensure every input dataset shares the same fps / robot_type / features schema."""
    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm(all_metadata, desc="Validate metadata"):
        if fps != meta.fps:
            raise ValueError(f"Same fps expected, got fps={meta.fps} instead of {fps}")
        if robot_type != meta.robot_type:
            raise ValueError(f"Same robot_type expected, got robot_type={meta.robot_type} instead of {robot_type}")
        if features != meta.features:
            raise ValueError(f"Same features expected, got features={meta.features} instead of {features}")

    return fps, robot_type, features


def aggregate(input_dirs, output_dir: Path, repo_id: str = None, delete_inputs: bool = False):
    lb = import_lerobot()
    LeRobotDatasetMetadata = lb.LeRobotDatasetMetadata

    all_metadata = [LeRobotDatasetMetadata("", root=d) for d in input_dirs]
    fps, robot_type, features = validate_all_metadata(all_metadata)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    repo_id = repo_id or f"local/{output_dir.name}"
    aggr_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
    )

    dataset_task_index_to_aggr = {}
    aggr_task_index = 0
    for dataset_index, meta in enumerate(tqdm(all_metadata, desc="Merge task index")):
        task_index_to_aggr_task_index = {}
        for task_index, task in meta.tasks.items():
            if task not in aggr_meta.task_to_task_index:
                aggr_meta.tasks[aggr_task_index] = task
                aggr_meta.task_to_task_index[task] = aggr_task_index
                aggr_task_index += 1
            task_index_to_aggr_task_index[task_index] = aggr_meta.task_to_task_index[task]
        dataset_task_index_to_aggr[dataset_index] = task_index_to_aggr_task_index

    dataset_episode_index_shift = {}
    dataset_index_shift = {}
    for dataset_index, meta in enumerate(tqdm(all_metadata, desc="Merge episodes")):
        dataset_episode_index_shift[dataset_index] = aggr_meta.total_episodes
        dataset_index_shift[dataset_index] = aggr_meta.total_frames

        for episode_index, episode_dict in meta.episodes.items():
            aggr_episode_index = episode_index + aggr_meta.total_episodes
            episode_dict = dict(episode_dict)
            episode_dict["episode_index"] = aggr_episode_index
            aggr_meta.episodes[aggr_episode_index] = episode_dict

        for episode_index, episode_stats in meta.episodes_stats.items():
            aggr_episode_index = episode_index + aggr_meta.total_episodes
            episode_stats = dict(episode_stats)
            episode_stats["index"] = {
                **episode_stats["index"],
                "min": episode_stats["index"]["min"] + aggr_meta.total_frames,
                "max": episode_stats["index"]["max"] + aggr_meta.total_frames,
                "mean": episode_stats["index"]["mean"] + aggr_meta.total_frames,
            }
            episode_stats["episode_index"] = {
                **episode_stats["episode_index"],
                "min": np.array([aggr_episode_index]),
                "max": np.array([aggr_episode_index]),
                "mean": np.array([aggr_episode_index]),
            }
            df = pd.read_parquet(meta.root / meta.get_data_file_path(episode_index))
            df["task_index"] = df["task_index"].map(dataset_task_index_to_aggr[dataset_index])
            episode_stats["task_index"] = {
                **episode_stats["task_index"],
                "min": np.array([df["task_index"].min()]),
                "max": np.array([df["task_index"].max()]),
                "mean": np.array([df["task_index"].mean()]),
                "std": np.array([df["task_index"].std()]),
            }
            aggr_meta.episodes_stats[aggr_episode_index] = episode_stats

        aggr_meta.info["total_episodes"] += meta.total_episodes
        aggr_meta.info["total_frames"] += meta.total_frames
        aggr_meta.info["total_videos"] += len(aggr_meta.video_keys) * meta.total_episodes

    aggr_meta.info["total_tasks"] = len(aggr_meta.tasks)
    aggr_meta.info["total_chunks"] = aggr_meta.get_episode_chunk(aggr_meta.total_episodes - 1) + 1
    aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.info['total_episodes']}"}

    logger.info("Writing merged metadata")
    for episode_dict in tqdm(aggr_meta.episodes.values(), desc="Write episodes"):
        lb.write_episode(episode_dict, aggr_meta.root)
    for episode_index, episode_stats in tqdm(aggr_meta.episodes_stats.items(), desc="Write episode stats"):
        lb.write_episode_stats(episode_index, episode_stats, aggr_meta.root)
    for task_index, task in tqdm(aggr_meta.tasks.items(), desc="Write tasks"):
        lb.write_task(task_index, task, aggr_meta.root)
    lb.write_info(aggr_meta.info, aggr_meta.root)

    logger.info("Copying data + video files")
    for dataset_index, meta in enumerate(tqdm(all_metadata, desc="Copy episode data")):
        episode_index_shift = dataset_episode_index_shift[dataset_index]
        index_shift = dataset_index_shift[dataset_index]
        task_index_to_aggr = dataset_task_index_to_aggr[dataset_index]

        for episode_index in range(meta.total_episodes):
            aggr_episode_index = episode_index + episode_index_shift

            data_path = meta.root / meta.get_data_file_path(episode_index)
            aggr_data_path = aggr_meta.root / aggr_meta.get_data_file_path(aggr_episode_index)
            aggr_data_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_parquet(data_path)
            df["index"] += index_shift
            df["episode_index"] += episode_index_shift
            df["task_index"] = df["task_index"].map(task_index_to_aggr)
            df.to_parquet(aggr_data_path)

            for vid_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(episode_index, vid_key)
                aggr_video_path = aggr_meta.root / aggr_meta.get_video_file_path(aggr_episode_index, vid_key)
                aggr_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, aggr_video_path)

    logger.info(f"Aggregated {len(input_dirs)} dataset(s) -> {output_dir} "
                f"({aggr_meta.info['total_episodes']} episodes, {aggr_meta.info['total_tasks']} unique tasks)")

    if delete_inputs:
        for d in input_dirs:
            shutil.rmtree(d)

    return aggr_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--delete-inputs", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    aggr_meta = aggregate(args.input_dirs, args.output_dir, repo_id=args.repo_id, delete_inputs=args.delete_inputs)

    if args.push_to_hub:
        lb = import_lerobot()
        dataset = lb.LeRobotDataset(repo_id=aggr_meta.repo_id, root=args.output_dir)
        dataset.push_to_hub(tags=["LeRobot", "robotwin", "aggregated"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    main()
