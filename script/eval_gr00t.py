#!/usr/bin/env python
"""Closed-loop GR00T evaluation for a RoboTwin task.

Runs a finetuned GR00T checkpoint against a RoboTwin task for a fixed number of trials
starting at a given seed, with a fixed instruction override, logging a per-trial and
aggregate success rate and saving one video per trial.

Must be run from the RoboTwin repo root (relative paths below mirror script/eval_policy.py):

    python script/eval_gr00t.py --model-path /path/to/checkpoint --task stack_bowls_three \\
        --instruction "stack the bowls" --num-trials 20 --seed 0
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append(str(Path(__file__).resolve().parent.parent / "policy" / "GR00T"))

import yaml  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from eval_policy import class_decorator, get_camera_config, get_embodiment_config  # noqa: E402
from gr00t_adapter import GR00TRoboTwinAdapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Path to the finetuned GR00T checkpoint")
    parser.add_argument("--task", required=True, help="RoboTwin task/module name, e.g. stack_bowls_three")
    parser.add_argument("--instruction", required=True, help="Fixed language instruction for every trial")
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Base seed; trial i uses seed + i")
    parser.add_argument("--task-config", default="demo_randomized", help="task_config/<name>.yml")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Receding-horizon steps executed per model query, out of the predicted action chunk",
    )
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--output-dir", default=None, help="Defaults to eval_result/<task>/GR00T/<task_config>/<timestamp>")
    return parser.parse_args()


def load_task_args(task_name: str, task_config: str) -> tuple[dict, dict]:
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args["embodiment"]
    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    camera_config = get_camera_config(args["camera"]["head_camera_type"])
    args["head_camera_h"] = camera_config["h"]
    args["head_camera_w"] = camera_config["w"]
    args["eval_mode"] = True
    return args, camera_config


def start_video_writer(task_env, episode_idx: int, video_size: str) -> None:
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", video_size,
            "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "23",
            f"{task_env.eval_video_path}/episode{episode_idx}.mp4",
        ],
        stdin=subprocess.PIPE,
    )
    task_env._set_eval_video_ffmpeg(ffmpeg)


def run_trial(
    task_env,
    args: dict,
    adapter: GR00TRoboTwinAdapter,
    trial_idx: int,
    seed: int,
    instruction: str,
    n_action_steps: int,
    camera_name: str,
    video_size: str,
) -> bool:
    task_env.setup_demo(now_ep_num=trial_idx, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=instruction)

    if task_env.eval_video_path is not None:
        start_video_writer(task_env, trial_idx, video_size)

    adapter.reset()
    success = False
    while task_env.take_action_cnt < task_env.step_lim:
        obs = task_env.get_obs()
        gr00t_obs = adapter.build_observation(obs, instruction)
        action, _ = adapter.policy.get_action(gr00t_obs)

        extrinsic_cv = obs["observation"][camera_name]["extrinsic_cv"]
        waypoints = adapter.decode_action_chunk(action, extrinsic_cv)

        for waypoint in waypoints[:n_action_steps]:
            task_env.take_action(waypoint, action_type="ee")
            if task_env.eval_success:
                success = True
                break
        if success:
            break

    if task_env.eval_video_path is not None:
        task_env._del_eval_video_ffmpeg()

    task_env.close_env(clear_cache=((trial_idx + 1) % args.get("clear_cache_freq", 5) == 0))
    return success


def main():
    cli = parse_args()

    from test_render import Sapien_TEST

    Sapien_TEST()

    args, camera_config = load_task_args(cli.task, cli.task_config)
    video_size = f"{camera_config['w']}x{camera_config['h']}"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(cli.output_dir)
        if cli.output_dir
        else Path(f"eval_result/{cli.task}/GR00T/{cli.task_config}/{current_time}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args["eval_video_save_dir"] = output_dir

    print(f"\033[34mTask Name: {cli.task}\033[0m")
    print(f"\033[34mModel Path: {cli.model_path}\033[0m")
    print(f"\033[34mInstruction: {cli.instruction}\033[0m")

    task_env = class_decorator(cli.task)
    adapter = GR00TRoboTwinAdapter(
        model_path=cli.model_path,
        embodiment_tag=cli.embodiment_tag,
        device=cli.device,
        camera_name=cli.camera_name,
    )

    trials = []
    for i in range(cli.num_trials):
        seed_i = cli.seed + i
        try:
            success = run_trial(
                task_env,
                args,
                adapter,
                i,
                seed_i,
                cli.instruction,
                cli.n_action_steps,
                cli.camera_name,
                video_size,
            )
        except Exception as e:
            print(f"\033[91mTrial {i} (seed {seed_i}) raised {type(e).__name__}: {e}\033[0m")
            task_env.close_env()
            success = False

        trials.append({"trial": i, "seed": seed_i, "success": success})
        status = "\033[92mSuccess\033[0m" if success else "\033[91mFail\033[0m"
        print(f"Trial {i + 1}/{cli.num_trials} (seed {seed_i}): {status}")

    successes = sum(t["success"] for t in trials)
    success_rate = successes / cli.num_trials
    results = {
        "task": cli.task,
        "task_config": cli.task_config,
        "model_path": cli.model_path,
        "instruction": cli.instruction,
        "num_trials": cli.num_trials,
        "seed": cli.seed,
        "success_rate": success_rate,
        "trials": trials,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccess rate: {successes}/{cli.num_trials} => {success_rate * 100:.1f}%")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
