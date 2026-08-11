#!/usr/bin/env python
"""Closed-loop GR00T evaluation for a RoboTwin task -- identical to
script/eval_gr00t_endpose.py except the video key(s) normally sourced from the
real head camera are instead sourced from RoboTwin's third-person "observer"
camera (obs["third_view_rgb"], populated when task_config's data_type.third_view
is true; see envs/_base_task.py:get_obs). Uses RoboTwin's native world-frame,
absolute endpose representation (no camera-frame reprojection, no rot6d).

Pair with a checkpoint trained on a dataset built via
script/lerobot/convert_robotwin_to_lerobot.py --head-camera-source third_view
(see convert_data_third_view.sh) -- that converter stores third-view frames
under the SAME "observation.images.head_camera" feature name a head-camera
dataset would use, so no separate --data-config is needed here; the checkpoint
just learned different pixel content for that slot.

Talks to gr00t.model.policy.Gr00tPolicy (Isaac-GR00T N1.5's classical API, as
found in /Users/lochathien/Documents/Code/Isaac-GR00T) via
policy/GR00T/gr00t_adapter_endpose.py (GR00TRoboTwinEndposeAdapter(third_view=True)).
This requires <model_path>/experiment_cfg/metadata.json (produced automatically
by the finetuning pipeline's checkpoint-save step) and a matching --data-config
(default "equi_robotwin_2hand_quat").

For the head-camera variant of this script, use script/eval_gr00t_endpose.py
instead -- this script is purely additive and doesn't change anything about
that one. For a checkpoint trained on the older MotionTrans/camera-frame/rot6d
schema, use script/eval_gr00t.py.

Runs a finetuned GR00T checkpoint against a RoboTwin task for a fixed number of trials
starting at a given seed, with a fixed instruction override, logging a per-trial and
aggregate success rate and saving one video per trial.

Must be run from the RoboTwin repo root (relative paths below mirror script/eval_policy.py).

If the gr00t package (ego_gr00t repo) is not installed into this environment, point
--gr00t-path (or the GR00T_REPO_PATH env var) at a conda env's site-packages dir
that already has gr00t (editable) + its dependencies installed:

    python script/eval_gr00t_endpose_thirdview.py --model-path /path/to/checkpoint --task stack_bowls_three \\
        --instruction "stack the bowls" --num-trials 20 --seed 0 \\
        --gr00t-path /home/user/miniconda3/envs/ego_gr00t/lib/python3.10/site-packages
"""

import os
import sys


def _reexec_with_libxcb_preload() -> None:
    """Some gr00t-env package pulled in transitively while constructing Gr00tPolicy
    (video loading in the checkpoint's data-transform pipeline -- decord/av/opencv
    wheels all vendor this way) ships an auditwheel-vendored copy of libxcb whose
    embedded DT_SONAME is still the plain "libxcb.so.1", not its hash-suffixed
    filename on disk. Once dlopen'd, the dynamic linker treats "libxcb.so.1" as
    already satisfied process-wide, so any *later* dependency resolution of that
    soname -- e.g. Mesa's llvmpipe Vulkan ICD resolving libxcb when SAPIEN/GLFW probes
    X11 window-surface support while constructing the *second* SapienRenderer of the
    run (the task_env's own, in run_trial -> setup_demo -> setup_scene; the first, in
    test_render.Sapien_TEST(), runs before gr00t is ever imported and is unaffected)
    -- silently gets the vendored copy instead of the real one, and segfaults deep in
    that .so with no Python-catchable exception the moment SAPIEN makes its first xcb
    call. Diagnosed via `gdb -batch -ex run -ex bt` around a real
    `Segmentation fault (core dumped)` that happened right after checkpoint loading,
    every time, 100% reproducible.

    Preloading the system libxcb fixes it, but ONLY via LD_PRELOAD specifically --
    glibc's ld.so consults LD_PRELOAD at process *exec* time, before the interpreter
    or any extension module has loaded anything, giving it priority in the soname
    race; a plain ctypes.CDLL(..., mode=RTLD_GLOBAL) issued after the process is
    already running does not (verified empirically: it still segfaults). Since we're
    already inside the process by the time this runs, the only way to actually get
    LD_PRELOAD applied is to re-exec ourselves once with it set -- which is exactly
    what this does, guarded by an env var so it only happens once.
    """
    if not sys.platform.startswith("linux") or os.environ.get("_ROBOTWIN_LIBXCB_PRELOADED"):
        return
    candidates = ("/usr/lib/x86_64-linux-gnu/libxcb.so.1", "/usr/lib64/libxcb.so.1", "/usr/lib/libxcb.so.1")
    libxcb = next((p for p in candidates if os.path.exists(p)), None)
    if libxcb is None:
        return  # no known system libxcb found; run un-preloaded and hope for the best
    env = dict(os.environ)
    existing = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{libxcb}:{existing}" if existing else libxcb
    env["_ROBOTWIN_LIBXCB_PRELOADED"] = "1"
    print(f"eval_gr00t_endpose_thirdview: re-exec'ing with LD_PRELOAD={libxcb} "
          "(works around a gr00t-env package's vendored libxcb crashing SAPIEN's "
          "renderer init -- see _reexec_with_libxcb_preload's docstring)")
    sys.stdout.flush()  # execve() replaces the process image without flushing buffered stdout
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_reexec_with_libxcb_preload()

import argparse
import json
import site
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append(str(Path(__file__).resolve().parent.parent / "policy" / "GR00T"))

import yaml  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from eval_policy import class_decorator, get_camera_config, get_embodiment_config  # noqa: E402
from gr00t_adapter_endpose import GR00TRoboTwinEndposeAdapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Path to the finetuned GR00T checkpoint")
    parser.add_argument("--task", required=True, help="RoboTwin task/module name, e.g. stack_bowls_three")
    parser.add_argument("--instruction", required=True, help="Fixed language instruction for every trial")
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Base seed; trial i uses seed + i")
    parser.add_argument("--task-config", default="demo_randomized", help="task_config/<name>.yml")
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help=(
            "Must match a gr00t.data.embodiment_tags.EmbodimentTag *value* exactly "
            "(lowercase: gr1/oxe_droid/agibot_genie1/new_embodiment on this branch of "
            "Isaac-GR00T) -- it's looked up by value, not by enum member name."
        ),
    )
    parser.add_argument(
        "--data-config",
        default="equi_robotwin_2hand_quat",
        help="Name passed to gr00t.experiment.data_config.load_data_config(); must match how the checkpoint was trained",
    )
    parser.add_argument("--denoising-steps", type=int, default=None, help="Override the checkpoint's default flow-matching denoising steps")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Receding-horizon steps executed per model query, out of the predicted action chunk",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to eval_result/<task>/GR00T-Endpose-ThirdView/<task_config>/<timestamp>",
    )
    parser.add_argument(
        "--gr00t-path",
        default=os.environ.get("GR00T_REPO_PATH"),
        help=(
            "Path prepended to sys.path (right before the GR00T policy is constructed, "
            "after RoboTwin's own env is already loaded) so 'import gr00t' resolves without "
            "installing the package here. Point it at another conda env's site-packages dir "
            "(e.g. /home/user/miniconda3/envs/ego_gr00t/lib/python3.10/site-packages) to pull "
            "in gr00t's dependencies (torch, transformers, ...) too, not just the ego_gr00t "
            "repo root -- the repo root alone only makes the gr00t package's own source "
            "importable, its dependencies still need to be present here some other way. "
            "Defaults to the GR00T_REPO_PATH env var."
        ),
    )
    return parser.parse_args()


def load_task_args(task_name: str, task_config: str) -> tuple:
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

    # Force on regardless of what task_config/<task_config>.yml itself sets -- this script's
    # whole point is feeding the policy third_view_rgb (envs/_base_task.py:get_obs), so it must
    # always be populated no matter which task_config the caller picked (e.g. demo_randomized,
    # whose default is data_type.third_view: false).
    args.setdefault("data_type", {})["third_view"] = True
    return args, camera_config


def prioritize_sitedir(path: str) -> None:
    """Add `path` to sys.path via site.addsitedir (so any .pth files in it -- including
    PEP 660 editable-install finders, which register via a .pth 'import' line and are
    invisible to a plain sys.path.insert -- actually get processed), then move every path
    entry addsitedir just added to the front of sys.path so they're resolved before
    whatever RoboTwin's own env already has under the same package names (addsitedir only
    appends by default)."""
    before = list(sys.path)
    site.addsitedir(path)
    new_entries = [p for p in sys.path if p not in before]
    remaining = [p for p in sys.path if p not in new_entries]
    sys.path[:] = new_entries + remaining


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
    adapter: GR00TRoboTwinEndposeAdapter,
    trial_idx: int,
    seed: int,
    instruction: str,
    n_action_steps: int,
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
        action = adapter.policy.get_action(gr00t_obs)  # gr00t.model.policy: single dict, not a (action, info) tuple
        waypoints = adapter.decode_action_chunk(action)

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
        # "GR00T-Endpose-ThirdView" is just this script's own output-path convention
        # (distinct from eval_gr00t_endpose.py's "GR00T-Endpose" and eval_gr00t.py's "GR00T"
        # segments) -- rename freely, nothing downstream depends on it.
        else Path(f"eval_result/{cli.task}/GR00T-Endpose-ThirdView/{cli.task_config}/{current_time}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args["eval_video_save_dir"] = output_dir

    print(f"\033[34mTask Name: {cli.task}\033[0m")
    print(f"\033[34mModel Path: {cli.model_path}\033[0m")
    print(f"\033[34mInstruction: {cli.instruction}\033[0m")

    task_env = class_decorator(cli.task)

    # Deferred until after RoboTwin's own env (sapien, mplib, numpy, ...) has already
    # imported and cached its own package versions in sys.modules -- inserting an external
    # conda env's site-packages any earlier risks shadowing RoboTwin's own numpy/opencv/etc.
    # with incompatible versions for anything RoboTwin imports afterward. Packages already
    # cached in sys.modules by this point are unaffected either way (Python won't
    # re-resolve an already-imported module), so this only changes resolution for
    # packages RoboTwin's env hasn't touched yet (transformers, torch, ...).
    if cli.gr00t_path:
        prioritize_sitedir(cli.gr00t_path)

    adapter = GR00TRoboTwinEndposeAdapter(
        model_path=cli.model_path,
        embodiment_tag=cli.embodiment_tag,
        data_config=cli.data_config,
        device=cli.device,
        denoising_steps=cli.denoising_steps,
        third_view=True,
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
