"""
Convert a RoboTwin task-folder (as produced by `script/collect_data.py`, e.g.
`data/grab_roller_aloha-agilex_clean_50/`) directly into a LeRobot v2.1 dataset.

RoboTwin's native per-episode HDF5 schema (see envs/_base_task.py:get_obs and
envs/utils/pkl2hdf5.py) is read directly -- no intermediate ALOHA-style HDF5
conversion step is needed (contrast with policy/GO1/scripts/process_data.py +
convert_aloha_data_to_lerobot_robotwin.py, which requires that intermediate
format).

Action/state representation: end-effector pose (`endpose`), NOT joint angles.
Both `observation.state` and `action` are a 16-dim vector:
    [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
     right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
Quaternions are wxyz (RoboTwin's native order) and are kept as-is -- do not
assume xyzw when consuming this dataset downstream (e.g. with
scipy.spatial.transform.Rotation, which defaults to xyzw).

`action[t]` is defined as the *next* frame's absolute pose (`state[t+1]`),
mirroring how policy/DP/process_data.py and policy/GO1/scripts/process_data.py
build their action labels from RoboTwin's inherently-absolute data. The last
frame of each episode is dropped since it has no successor.

Cameras: only head_camera, left_camera, right_camera are kept (front_camera is
dropped, matching what every existing RoboTwin policy converter uses
downstream). Images are stored in LeRobot "video" mode (mp4-encoded).

--head-camera-source lets the "head_camera" output feature be populated from
RoboTwin's third-person "observer" camera instead of the real head camera --
recorded as a top-level `third_view_rgb` HDF5 dataset (see
envs/_base_task.py:get_obs, gated by task_config's data_type.third_view: true;
data/third_view/<task>/demo_clean/ is collected this way) rather than nested
under observation/head_camera/rgb like every other camera. The output feature
is still named "observation.images.head_camera" either way -- only the pixel
source changes -- so a checkpoint/data-config built around the original
head_camera schema needs no changes to train or eval on a third-view dataset;
pair with GR00TRoboTwinEndposeAdapter(third_view=True) (policy/GR00T/
gr00t_adapter_endpose.py) / script/eval_gr00t_endpose_thirdview.py so eval-time
observations are built from the same third_view_rgb source.

By default, each episode's language instruction is randomly sampled from its
own instructions/episode<N>.json (per --instruction-set). Pass --instruction
"..." to use one fixed instruction string for every episode/frame instead.

Usage:
    python script/lerobot/convert_robotwin_to_lerobot.py \\
        --src-path data/grab_roller_aloha-agilex_clean_50 \\
        --output-path /tmp/lerobot_out/grab_roller \\
        --max-episodes 2
"""

import argparse
import logging
import random
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

from lerobot_compat import import_lerobot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")
DEFAULT_FPS = 15

STATE_ACTION_NAMES = {
    "motors": [
        "left_x", "left_y", "left_z", "left_qw", "left_qx", "left_qy", "left_qz", "left_gripper",
        "right_x", "right_y", "right_z", "right_qw", "right_qx", "right_qy", "right_qz", "right_gripper",
    ]
}

FOLDER_NAME_RE = re.compile(r"^(.+)_([A-Za-z0-9\-]+)_([A-Za-z0-9]+)_(\d+)$")
EPISODE_INDEX_RE = re.compile(r"episode(\d+)\.hdf5$")


def camera_h5_path(camera: str, head_camera_source: str) -> str:
    """HDF5 path to a camera's (T,) JPEG-byte-string dataset within an episode file.
    Every camera except a head_camera_source="third_view" head_camera lives at
    observation/<camera>/rgb; that one case reads the top-level third_view_rgb
    dataset instead (see module docstring)."""
    if camera == "head_camera" and head_camera_source == "third_view":
        return "third_view_rgb"
    return f"observation/{camera}/rgb"


def decode_jpeg_stream(raw: np.ndarray, ep_path: Path, camera: str) -> np.ndarray:
    """Decode a RoboTwin rgb dataset: (T,) NUL-padded JPEG byte strings -> (T,H,W,3) uint8 RGB."""
    frames = []
    for i, buf in enumerate(raw):
        jpeg_bytes = bytes(buf).rstrip(b"\x00")
        img_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"cv2.imdecode failed for {ep_path} camera={camera} frame={i}")
        frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def load_raw_episode(ep_path: Path, cameras: tuple = DEFAULT_CAMERAS, layout: str = "hwc", head_camera_source: str = "head"):
    """
    Read one RoboTwin episode{N}.hdf5 and return (frames, image_shape) where:
      - frames is a list[dict] of length T-1, one dict per LeRobot frame, keyed
        exactly like the `features` dict this converter builds ("observation.state",
        "action", "observation.images.<camera>"); or None if the episode has < 2 frames.
      - image_shape is the per-camera (H, W, 3) shape read from this episode (or None
        if frames is None).
    """
    with h5py.File(ep_path, "r") as ep:
        num_raw_frames = ep["endpose/left_endpose"].shape[0]
        if num_raw_frames < 2:
            logger.warning(f"{ep_path}: only {num_raw_frames} frame(s), need >=2 (state, next-state pair); skipping")
            return None, None

        left_pose = ep["endpose/left_endpose"][()]
        left_grip = ep["endpose/left_gripper"][()][:, None]
        right_pose = ep["endpose/right_endpose"][()]
        right_grip = ep["endpose/right_gripper"][()][:, None]
        state = np.concatenate([left_pose, left_grip, right_pose, right_grip], axis=1).astype(np.float32)

        if state.shape[0] != num_raw_frames:
            raise ValueError(f"{ep_path}: state length {state.shape[0]} != endpose length {num_raw_frames}")

        imgs = {}
        image_shape = None
        for cam in cameras:
            raw = ep[camera_h5_path(cam, head_camera_source)][()]
            if raw.shape[0] != num_raw_frames:
                raise ValueError(f"{ep_path}: camera '{cam}' has {raw.shape[0]} frames, expected {num_raw_frames}")
            decoded = decode_jpeg_stream(raw, ep_path, cam)
            if image_shape is None:
                image_shape = decoded.shape[1:]
            elif decoded.shape[1:] != image_shape:
                raise RuntimeError(f"{ep_path}: camera '{cam}' shape {decoded.shape[1:]} != other cameras' shape {image_shape}")
            imgs[cam] = decoded

    # action[t] = state[t+1] (next absolute pose); drop the last frame (no successor).
    obs_state = state[:-1]
    act_state = state[1:]
    for cam in cameras:
        imgs[cam] = imgs[cam][:-1]
        if layout == "chw":
            imgs[cam] = np.transpose(imgs[cam], (0, 3, 1, 2))

    num_frames = obs_state.shape[0]
    frames = []
    for i in range(num_frames):
        frame = {
            "observation.state": obs_state[i],
            "action": act_state[i],
        }
        for cam in cameras:
            frame[f"observation.images.{cam}"] = imgs[cam][i]
        frames.append(frame)

    return frames, image_shape


def build_features(image_shape: tuple, cameras: tuple, layout: str) -> dict:
    h, w, c = image_shape
    if layout == "hwc":
        img_feature_shape = (h, w, c)
        img_names = ["height", "width", "channel"]
    elif layout == "chw":
        img_feature_shape = (c, h, w)
        img_names = ["channels", "height", "width"]
    else:
        raise ValueError(f"Unknown image layout '{layout}', expected 'hwc' or 'chw'")

    features = {}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": img_feature_shape,
            "names": img_names,
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (16,),
        "names": STATE_ACTION_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (16,),
        "names": STATE_ACTION_NAMES,
    }
    return features


def parse_task_folder_name(name: str):
    """'grab_roller_aloha-agilex_clean_50' -> ('grab_roller', 'aloha-agilex', 'clean', 50)."""
    m = FOLDER_NAME_RE.match(name)
    if not m:
        return None, None, None, None
    task_name, embodiment, config_token, episode_count = m.groups()
    return task_name, embodiment, config_token, int(episode_count)


def discover_episodes(src_path: Path):
    """Return sorted list of (index, path) for every episode*.hdf5 in src_path/data, gap-tolerant."""
    data_dir = src_path / "data"
    episodes = []
    for p in data_dir.glob("episode*.hdf5"):
        m = EPISODE_INDEX_RE.search(p.name)
        if not m:
            continue
        episodes.append((int(m.group(1)), p))
    episodes.sort(key=lambda x: x[0])
    return episodes


def resolve_fps(explicit_fps, config_token, repo_root: Path):
    if explicit_fps is not None:
        return explicit_fps
    if config_token and repo_root:
        yml_path = repo_root / "task_config" / f"demo_{config_token}.yml"
        if yml_path.exists():
            try:
                with open(yml_path) as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict) and "save_freq" in cfg:
                    logger.info(f"Auto-derived fps={cfg['save_freq']} from {yml_path}")
                    return cfg["save_freq"]
            except Exception as e:
                logger.warning(f"Failed to read fps from {yml_path}: {e}")
    logger.info(f"Falling back to default fps={DEFAULT_FPS}")
    return DEFAULT_FPS


def load_instruction(src_path: Path, ep_idx: int, task_name: str, instruction_set: str) -> str:
    instr_path = src_path / "instructions" / f"episode{ep_idx}.json"
    if not instr_path.exists():
        logger.warning(f"Missing {instr_path}; falling back to task name as instruction")
        return task_name.replace("_", " ")
    import json

    with open(instr_path) as f:
        data = json.load(f)
    choices = data.get(instruction_set)
    if not choices:
        logger.warning(f"{instr_path} has no '{instruction_set}' instructions; falling back to task name")
        return task_name.replace("_", " ")
    return random.choice(choices)


def convert_task_folder(
    src_path: Path,
    output_path: Path,
    fps=None,
    repo_id=None,
    robot_type=None,
    task_name=None,
    cameras=DEFAULT_CAMERAS,
    image_layout="auto",
    instruction_set="seen",
    instruction=None,
    max_episodes=None,
    overwrite=False,
    push_to_hub=False,
    repo_root=None,
    head_camera_source="head",
):
    lb = import_lerobot()
    layout = image_layout if image_layout != "auto" else lb.default_image_layout

    parsed_task_name, parsed_embodiment, config_token, _ = parse_task_folder_name(src_path.name)
    task_name = task_name or parsed_task_name or src_path.name
    robot_type = robot_type or parsed_embodiment or "unknown"
    repo_root = repo_root or Path(__file__).resolve().parents[2]

    fps = resolve_fps(fps, config_token, repo_root)
    repo_id = repo_id or f"local/{task_name}"

    episodes = discover_episodes(src_path)
    if not episodes:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {src_path / 'data'}")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    logger.info(f"Found {len(episodes)} episode(s) in {src_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists (pass --overwrite to replace it)")
        import shutil

        shutil.rmtree(output_path)

    # Probe the first episode to determine image shape and build the features dict.
    first_idx, first_path = episodes[0]
    probe_frames, image_shape = load_raw_episode(first_path, cameras, layout, head_camera_source)
    if probe_frames is None:
        raise RuntimeError(f"First episode {first_path} has too few frames to probe image shape from")

    features = build_features(image_shape, cameras, layout)
    logger.info(
        f"Using image layout={layout}, image_shape(H,W,C)={image_shape}, fps={fps}, robot_type={robot_type}, "
        f"head_camera_source={head_camera_source}"
    )

    dataset = lb.LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )

    n_converted = 0
    for ep_idx, ep_path in tqdm(episodes, desc=f"Converting {src_path.name}"):
        if ep_idx == first_idx:
            frames, ep_image_shape = probe_frames, image_shape
        else:
            frames, ep_image_shape = load_raw_episode(ep_path, cameras, layout, head_camera_source)
        if frames is None:
            continue
        if ep_image_shape != image_shape:
            raise RuntimeError(
                f"{ep_path}: image shape {ep_image_shape} differs from the shape probed from episode "
                f"{first_idx} ({image_shape}); all episodes in a task folder must share the same resolution"
            )

        ep_instruction = instruction if instruction is not None else load_instruction(src_path, ep_idx, task_name, instruction_set)
        for frame in frames:
            dataset.add_frame(frame, task=ep_instruction)
        dataset.save_episode()
        n_converted += 1

    logger.info(f"Converted {n_converted}/{len(episodes)} episode(s) into {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["LeRobot", "robotwin", robot_type, task_name],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-path", type=Path, required=True, help="RoboTwin task folder, e.g. data/grab_roller_aloha-agilex_clean_50")
    parser.add_argument("--output-path", type=Path, required=True, help="destination LeRobot dataset root directory")
    parser.add_argument("--fps", type=int, default=None, help="default: auto-derive from task_config/demo_<config>.yml, else 15")
    parser.add_argument("--repo-id", type=str, default=None, help="default: local/<task_name>")
    parser.add_argument("--robot-type", type=str, default=None, help="default: parsed from folder name (embodiment)")
    parser.add_argument("--task-name", type=str, default=None, help="default: parsed from folder name")
    parser.add_argument("--cameras", type=str, nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--image-layout", type=str, choices=["auto", "hwc", "chw"], default="auto")
    parser.add_argument("--instruction-set", type=str, choices=["seen", "unseen"], default="seen")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="use this single fixed instruction string for every episode/frame in the dataset, instead of "
        "randomly sampling a different one per episode from instructions/episode<N>.json",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None, help="RoboTwin repo root, for fps auto-lookup (default: two parents up from this script)")
    parser.add_argument(
        "--head-camera-source",
        type=str,
        choices=["head", "third_view"],
        default="head",
        help="'third_view' populates the 'head_camera' feature from the top-level third_view_rgb HDF5 "
        "dataset (RoboTwin's third-person observer camera, task_config data_type.third_view: true) "
        "instead of observation/head_camera/rgb -- see module docstring.",
    )
    args = parser.parse_args()

    convert_task_folder(
        src_path=args.src_path,
        output_path=args.output_path,
        fps=args.fps,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        task_name=args.task_name,
        cameras=tuple(args.cameras),
        image_layout=args.image_layout,
        instruction_set=args.instruction_set,
        instruction=args.instruction,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        repo_root=args.repo_root,
        head_camera_source=args.head_camera_source,
    )


if __name__ == "__main__":
    main()
