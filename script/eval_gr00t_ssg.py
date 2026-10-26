#!/usr/bin/env python
"""Shared-scene-graph (SSG) P1 / P5 eval: eval_gr00t_parallel.py plus the GT scene graph of the
current frame as observation["graph"].

The env workers turn on data_type.scene_graph (envs/utils/scene_graph_logger.py), so get_obs
carries object poses, head-camera masks and finger contacts; the main process keeps the last
few observations of every env and builds the graph with the converter's own
data_loader_for_robotwin_ssg.build_graph -- the code that labelled the training data --
taking its last row. Observations come once per action chunk, so the frame-rate constants of
build_graph are rescaled to chunk rate (held_by debounce 1, activity per chunk).

Extra flags (everything else as eval_gr00t_parallel.py):
  --graph-mode M     oracle (default): GT graph each call; none: no graph key
  --action-space S   ee (default): camera-frame EEF checkpoints (GR00TRoboTwinAdapter);
                     joint: 14-D joint checkpoints (gr00t_joint_adapter.py), take_action("qpos")
  --ssg-converter D  lerobot-convert checkout with the SSG loader
                     (default /mnt/data/sftp/data/locht1/workspace/lerobot-convert_ssg)
Run with PYTHONPATH pointing at the GR00T checkout whose Gr00tPolicy accepts "graph".
"""
"""Parallel-env, batched-inference variant of script/eval_gr00t.py.

Runs --num-envs RoboTwin environments, each in its own subprocess, and a single GR00T
policy in this (main) process. Every env that is waiting for an action is served by ONE
batched Gr00tPolicy.get_action call, so the GPU runs one B-sized forward instead of B
separate ones, and the physics/rendering of the other envs overlaps with inference.

Why subprocesses rather than several envs in one process: Base_Task.setup_demo reseeds the
GLOBAL np.random / torch RNGs with the trial seed, and close_env(clear_cache=True) clears
SAPIEN's process-wide render cache -- interleaving several envs in one process would let
them clobber each other.

Same CLI, output layout and results.json schema as eval_gr00t.py (so eval_arm.sh's
completion check and resume work unchanged), plus --num-envs. Seed semantics match too:
seeds are dispensed in increasing order, unstable seeds are skipped without consuming a
trial, so the evaluated set is exactly the first num_trials stable seeds from --seed, and
trial i is the i-th of them. Trials finish out of order across envs, but results.json only
ever holds the contiguous seed-ordered prefix of finished trials, so a crash loses at most
the in-flight ones and a rerun resumes exactly like eval_gr00t.py.

Policy noise. In eval_gr00t.py, setup_demo's torch.manual_seed(trial seed) also fixes the
flow-matching noise the policy samples, so trial s always gets the same noise stream and
every checkpoint is scored on the same (scene, noise) pairs. With --no-per-trial-noise this
script samples the noise of the whole batch from ONE torch RNG seeded
once with --seed, so a trial's noise depends on which other envs share its batch: reruns
differ and arms are not paired (the protocol before 2026-09-28). The default
(--per-trial-noise) restores the sequential semantics: each seed gets its own CUDA generator
(seeded with the seed at reset), and the initial noise of every batched get_action is built
row by row from the generator of that row's trial. Outcomes are then reproducible and paired
across checkpoints, but still NOT bit-identical to eval_gr00t.py (batch-8 vs batch-1 kernels
round differently, and the sim amplifies that), so compare rates, not individual trials.

    python script/eval_gr00t_parallel.py --model-path /path/to/checkpoint \\
        --task press_stapler --instruction "Use one arm to press the stapler." \\
        --num-trials 100 --seed 0 --num-envs 8 --task-config demo_clean_bg16 \\
        --gr00t-path /home/user/miniconda3/envs/ego_gr00t/lib/python3.10/site-packages
"""

import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from multiprocessing.connection import wait
from pathlib import Path

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "policy" / "GR00T"))

import numpy as np  # noqa: E402

import eval_gr00t as seq  # noqa: E402


def parse_args():
    # Reuse eval_gr00t.py's parser so the two scripts can never drift apart.
    argv, num_envs = list(sys.argv[1:]), 4
    graph_mode = _pop_flag(argv, "--graph-mode", "oracle")
    action_space = _pop_flag(argv, "--action-space", "ee")
    if action_space not in ("ee", "joint"):
        raise SystemExit("--action-space ee|joint")
    ssg_converter = _pop_flag(argv, "--ssg-converter", SSG_CONVERTER)
    # --lean-env: joint-space envs skip the CuRobo planners (GPU memory -> more envs) and
    # envs without --graph-mode oracle skip the scene-graph renders; actions are unchanged
    lean_env = "--lean-env" in argv
    while "--lean-env" in argv:
        argv.remove("--lean-env")
    if graph_mode not in ("oracle", "none"):
        raise SystemExit("--graph-mode oracle|none")
    if "--num-envs" in argv:
        i = argv.index("--num-envs")
        num_envs = int(argv[i + 1])
        del argv[i : i + 2]
    per_trial_noise = True  # default since 2026-09-28; --no-per-trial-noise = old protocol
    for flag in ("--per-trial-noise", "--no-per-trial-noise"):
        while flag in argv:
            argv.remove(flag)
            per_trial_noise = flag == "--per-trial-noise"
    sys.argv[1:] = argv
    cli = seq.parse_args()
    cli.num_envs = num_envs
    cli.per_trial_noise = per_trial_noise
    cli.graph_mode, cli.ssg_converter, cli.action_space = graph_mode, ssg_converter, action_space
    cli.lean_env = lean_env
    if cli.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")
    return cli


# ----------------------------------------------------------------------------- worker side


SSG_CONVERTER = "/mnt/data/sftp/data/locht1/workspace/lerobot-convert_ssg"
GRAPH_HISTORY = 8  # observations (action chunks) kept per env for build_graph


def _pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i : i + 2]
        return val
    return default


def _slim_obs(obs: dict, camera_name: str) -> dict:
    """Only what GR00TRoboTwinAdapter.build_observation reads -- keeps pipe traffic to the
    head frame (+ the wrist frames, which 3-camera checkpoints read as camera1/camera2)
    and a few poses instead of every camera/pointcloud get_obs produced."""
    cam = obs["observation"][camera_name]
    out = {camera_name: {"rgb": cam["rgb"], "extrinsic_cv": cam["extrinsic_cv"]}}
    for wrist in ("right_camera", "left_camera"):
        if wrist in obs["observation"] and "rgb" in obs["observation"][wrist]:
            out[wrist] = {"rgb": obs["observation"][wrist]["rgb"]}
    out[camera_name]["intrinsic_cv"] = cam["intrinsic_cv"]
    return {
        "observation": out,
        "endpose": obs["endpose"],
        "joint_action": {"vector": obs["joint_action"]["vector"]},
        "scene_graph": obs.get("scene_graph"),
    }


def _start_video(task_env, video_path: str, video_size: str) -> None:
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", video_size,
            "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "23", video_path,
        ],
        stdin=subprocess.PIPE,
    )
    task_env._set_eval_video_ffmpeg(ffmpeg)


def env_worker(conn, task: str, task_config: str, instruction: str, n_action_steps: int,
               camera_name: str, video_dir: str | None, action_type: str = "ee",
               lean_env: bool = False, graph_mode: str = "oracle") -> None:
    """One RoboTwin env. Protocol (main -> worker):
        ("reset", seed)          -> ("obs", obs) | ("unstable", msg) | ("fatal", err)
        ("step", waypoints)      -> ("obs", obs) | ("done", success, err_or_None) | ("fatal", err)
        ("close",)               -> exits
    A "fatal" reply means the env itself is broken (setup failed, GPU OOM, render buffer
    allocation) -- not a policy outcome -- and the main process aborts rather than record
    it as a failed trial.
    """
    # take_action prints a \r progress counter every step; with N envs that is N x the
    # log volume for nothing. Errors still go to stderr / back over the pipe.
    sys.stdout = open(os.devnull, "w")
    from envs.utils.create_actor import UnStableError
    from eval_policy import class_decorator

    args, camera_config = seq.load_task_args(task, task_config)
    args.setdefault("data_type", {})["scene_graph"] = not (lean_env and graph_mode != "oracle")
    if lean_env and action_type == "qpos":
        args["skip_curobo"] = True
    video_size = f"{camera_config['w']}x{camera_config['h']}"
    args["eval_video_save_dir"] = video_dir
    task_env = class_decorator(task)
    n_run = 0
    has_video = False

    def finish(success: bool, err):
        nonlocal n_run, has_video
        try:
            if has_video:
                task_env._del_eval_video_ffmpeg()
        finally:
            has_video = False
            n_run += 1
            task_env.close_env(clear_cache=(n_run % args.get("clear_cache_freq", 5) == 0))
        conn.send(("done", success, err))

    while True:
        msg = conn.recv()
        if msg[0] == "close":
            return
        try:
            if msg[0] == "reset":
                seed = msg[1]
                try:
                    task_env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **args)
                except UnStableError as e:
                    task_env.close_env()
                    conn.send(("unstable", str(e)))
                    continue
                task_env.set_instruction(instruction=instruction)
                if video_dir is not None:
                    _start_video(task_env, f"{video_dir}/pending_seed{seed}.mp4", video_size)
                    has_video = True
            else:  # step
                for waypoint in msg[1][:n_action_steps]:
                    task_env.take_action(waypoint, action_type=action_type)
                    if task_env.eval_success:
                        break
                if task_env.eval_success:
                    finish(True, None)
                    continue
            if task_env.take_action_cnt >= task_env.step_lim:
                finish(False, None)
            else:
                conn.send(("obs", _slim_obs(task_env.get_obs(), camera_name)))
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            # A setup that dies halfway leaves the env half-built (e.g. Robot without
            # left_planner after a CuRobo OOM), so every later seed on it fails too -- one
            # OOM once turned a whole cell into 100 recorded failures in two minutes.
            if msg[0] == "reset" or _is_infra_error(e):
                conn.send(("fatal", err))
                return
            try:
                finish(False, err)
            except Exception:
                conn.send(("done", False, err))


def _is_infra_error(e: Exception) -> bool:
    text = f"{type(e).__name__}: {e}".lower()
    return "out of memory" in text or "outofmemory" in text or "cannot create buffer" in text


# ------------------------------------------------------------------------------- main side


class PerRowNoise:
    """Context manager: during one batched get_action, the model's
    torch.randn(size=(B, H, D)) for the initial flow-matching noise returns rows drawn from
    per-trial generators instead of the global RNG. Any other randn call passes through.
    Raises if the initial-noise call does not happen exactly once, so a model change that
    moves or adds sampling cannot silently fall back to unseeded noise."""

    def __init__(self, torch, generators):
        self.torch, self.gens, self.hits = torch, generators, 0

    def __enter__(self):
        torch, orig = self.torch, self.torch.randn
        self.orig = orig

        def randn(*args, **kwargs):
            size = kwargs.get("size", args[0] if len(args) == 1 else args)
            size = tuple(size)
            # The initial noise is always drawn on an explicit device; parameter inits
            # (e.g. a lazily built submodule's torch.randn(1, W, D)) never pass one.
            if (len(size) == 3 and size[0] == len(self.gens) and "generator" not in kwargs
                    and kwargs.get("device") is not None):
                self.hits += 1
                dtype, device = kwargs.get("dtype"), kwargs.get("device")
                return torch.cat(
                    [orig((1, *size[1:]), generator=g, dtype=dtype, device=device) for g in self.gens]
                )
            return orig(*args, **kwargs)

        torch.randn = randn
        return self

    def __exit__(self, exc_type, *_):
        self.torch.randn = self.orig
        if exc_type is None and self.hits != 1:
            raise RuntimeError(
                f"--per-trial-noise: expected exactly one initial-noise torch.randn of batch "
                f"size {len(self.gens)} per get_action, saw {self.hits}; the model's sampling "
                f"changed -- update PerRowNoise")


def stack_observations(obs_list: list[dict]) -> dict:
    """B single-env (B=1) GR00T observations -> one (B, ...) batch."""
    first = obs_list[0]
    return {
        "video": {k: np.concatenate([o["video"][k] for o in obs_list], 0) for k in first["video"]},
        "state": {k: np.concatenate([o["state"][k] for o in obs_list], 0) for k in first["state"]},
        "language": {k: sum((o["language"][k] for o in obs_list), []) for k in first["language"]},
        **({"graph": {k: np.concatenate([o["graph"][k] for o in obs_list], 0) for k in first["graph"]}}
           if "graph" in first else {}),
    }


class GraphHistory:
    """Per-env window of recent observations, viewed as the HDF5 layout build_graph reads."""

    def __init__(self, task: str, camera_name: str, converter: str, n_action_steps: int):
        sys.path.insert(0, converter)
        import data_loader_for_robotwin_ssg as loader

        # build_graph's constants are per 30 fps frame; here one row is one action chunk
        loader.HOLD_DEBOUNCE = 1
        loader.ACTIVE_WINDOW = 1
        loader.ACTIVE_SPEED = loader.ACTIVE_SPEED * n_action_steps
        self.build_graph = loader.build_graph
        self.task, self.camera = task, camera_name
        self.rows: dict[int, list[dict]] = {}

    def reset(self, w: int) -> None:
        self.rows[w] = []

    def graph(self, w: int, obs: dict) -> dict:
        cam = obs["observation"][self.camera]
        row = {f"scene_graph/{k}": np.asarray(v) for k, v in obs["scene_graph"].items()}
        row["observation/head_camera/intrinsic_cv"] = np.asarray(cam["intrinsic_cv"])
        row["observation/head_camera/extrinsic_cv"] = np.asarray(cam["extrinsic_cv"])
        for side in ("left", "right"):
            row[f"endpose/{side}_endpose"] = np.asarray(obs["endpose"][f"{side}_endpose"])
            row[f"endpose/{side}_gripper"] = np.asarray(obs["endpose"][f"{side}_gripper"])
        rows = self.rows.setdefault(w, [])
        rows.append(row)
        del rows[:-GRAPH_HISTORY]
        stacked = _H5View({k: np.stack([r[k] for r in rows]) for k in row})
        cols, _ = self.build_graph(stacked, self.task)
        # [B=1, T=1, width], the layout Gr00tPolicy unbatches
        return {k[len("graph."):]: v[-1:][None] for k, v in cols.items()}


class _H5View(dict):
    """dict with h5py-style group access: f["scene_graph"]["obj_pose"] == f["scene_graph/obj_pose"]."""

    def __getitem__(self, key):
        if key in self.keys():
            return dict.__getitem__(self, key)
        prefix = key + "/"
        return _H5View({k[len(prefix):]: v for k, v in self.items() if k.startswith(prefix)})


def main():
    cli = parse_args()

    from test_render import Sapien_TEST

    Sapien_TEST()

    output_dir = Path(cli.output_dir) if cli.output_dir else None
    if output_dir is None:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(f"eval_result/{cli.task}/GR00T/{cli.task_config}/{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("pending_seed*.mp4"):  # in-flight videos of a crashed run
        stale.unlink()

    results_path = output_dir / "results.json"
    trials = [] if cli.fresh else seq.load_resumable_trials(results_path, cli)
    if len(trials) >= cli.num_trials:
        _, successes, success_rate = seq.write_results(cli, results_path, trials)
        print(f"\033[92mAll {cli.num_trials} trials already completed in {results_path}; nothing to do "
              f"(pass --fresh to rerun from scratch).\033[0m")
        print(f"\nSuccess rate: {successes}/{cli.num_trials} => {success_rate * 100:.1f}%")
        return

    print(f"\033[34mTask Name: {cli.task}\033[0m")
    print(f"\033[34mModel Path: {cli.model_path}\033[0m")
    print(f"\033[34mInstruction: {cli.instruction}\033[0m")
    print(f"\033[34mParallel envs: {cli.num_envs}\033[0m")
    print(f"\033[34mPer-trial policy noise: {'on' if cli.per_trial_noise else 'off'}\033[0m")
    if trials:
        print(f"\033[36mResuming from {results_path}: {len(trials)}/{cli.num_trials} trials already done.\033[0m")

    remaining = cli.num_trials - len(trials)
    num_envs = min(cli.num_envs, remaining)

    # Workers are spawned BEFORE the gr00t site-packages go on sys.path (and before torch
    # touches CUDA), so they import only RoboTwin's own env -- see eval_gr00t.py main().
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for _ in range(num_envs):
        parent, child = ctx.Pipe()
        p = ctx.Process(
            target=env_worker,
            args=(child, cli.task, cli.task_config, cli.instruction, cli.n_action_steps,
                  cli.camera_name, str(output_dir), "qpos" if cli.action_space == "joint" else "ee",
                  cli.lean_env, cli.graph_mode),
            daemon=True,
        )
        p.start()
        child.close()
        conns.append(parent)
        procs.append(p)

    if cli.gr00t_path:
        seq.prioritize_sitedir(cli.gr00t_path)
    import torch

    torch.manual_seed(cli.seed)
    history = GraphHistory(cli.task, cli.camera_name, cli.ssg_converter, cli.n_action_steps)
    adapter_cls = seq.GR00TRoboTwinAdapter
    if cli.action_space == "joint":
        from gr00t_joint_adapter import GR00TRoboTwinJointAdapter as adapter_cls
    adapter = adapter_cls(
        model_path=cli.model_path,
        embodiment_tag=cli.embodiment_tag,
        device=cli.device,
        camera_name=cli.camera_name,
    )

    next_seed = trials[-1]["seed"] + 1 if trials else cli.seed
    accepted = 0           # seeds whose setup did not raise UnStableError
    pending_setup = 0      # seeds dispatched whose stability is not known yet
    resolved = {}          # seed -> "unstable" | "running" | {"success":..}
    seed_of = {}           # worker idx -> seed it is running
    idle = set(range(num_envs))
    first_open_seed = next_seed  # every seed below this is committed or unstable
    noise_gen = {}         # seed -> its own CUDA generator (--per-trial-noise)

    def dispatch(w):
        nonlocal next_seed, pending_setup
        if cli.per_trial_noise:
            noise_gen[next_seed] = torch.Generator(device=cli.device).manual_seed(next_seed)
        seed_of[w] = next_seed
        resolved[next_seed] = "setup"
        history.reset(w)
        conns[w].send(("reset", next_seed))
        next_seed += 1
        pending_setup += 1
        idle.discard(w)

    def fill():
        for w in sorted(idle):
            if accepted + pending_setup >= remaining:
                return
            dispatch(w)

    def commit():
        """Append finished trials to results.json in seed order, as far as the prefix is
        fully resolved. Trial numbers / video names are only assigned here, once every
        smaller seed is known to be stable-and-finished or unstable."""
        nonlocal first_open_seed
        wrote = False
        while first_open_seed in resolved:
            state = resolved[first_open_seed]
            if state == "unstable":
                pass
            elif isinstance(state, dict):
                i = len(trials)
                pending = output_dir / f"pending_seed{first_open_seed}.mp4"
                if pending.exists():
                    pending.replace(output_dir / f"episode{i}.mp4")
                trials.append({"trial": i, "seed": first_open_seed, "success": state["success"]})
                status = "\033[92mSuccess\033[0m" if state["success"] else "\033[91mFail\033[0m"
                print(f"Trial {i + 1}/{cli.num_trials} (seed {first_open_seed}): {status}", flush=True)
                wrote = True
            else:
                break
            del resolved[first_open_seed]
            first_open_seed += 1
        if wrote:
            seq.write_results(cli, results_path, trials)

    fill()
    try:
        while len(trials) < cli.num_trials:
            busy = [conns[w] for w in range(num_envs) if w not in idle]
            if not busy:
                raise RuntimeError("no env running but trials remain -- scheduler bug")
            ready = wait(busy)
            batch = []  # (worker, obs)
            for c in ready:
                w = conns.index(c)
                try:
                    msg = c.recv()
                except EOFError:
                    raise RuntimeError(f"env worker {w} (seed {seed_of.get(w)}) died, "
                                       f"exitcode={procs[w].exitcode}") from None
                seed = seed_of[w]
                kind = msg[0]
                if resolved.get(seed) == "setup" and kind != "unstable":
                    pending_setup -= 1
                    accepted += 1
                    resolved[seed] = "running"
                if kind == "obs":
                    batch.append((w, msg[1]))
                elif kind == "fatal":
                    raise RuntimeError(
                        f"env worker {w} broke on seed {seed} -- aborting without recording it "
                        f"(results.json keeps only real trials; rerun to resume):\n{msg[1]}")
                elif kind == "unstable":
                    pending_setup -= 1
                    resolved[seed] = "unstable"
                    print(f"\033[93mSeed {seed} unstable, skipping: {msg[1]}\033[0m", flush=True)
                    noise_gen.pop(seed, None)
                    idle.add(w)
                else:  # done
                    _, success, err = msg
                    if err:
                        print(f"\033[91mSeed {seed} raised {err}\033[0m", flush=True)
                    resolved[seed] = {"success": success}
                    noise_gen.pop(seed, None)
                    idle.add(w)

            if batch:
                obs_list = []
                for w, o in batch:
                    g = adapter.build_observation(o, cli.instruction)
                    if cli.graph_mode == "oracle":
                        g["graph"] = history.graph(w, o)
                    obs_list.append(g)
                gr00t_obs = stack_observations(obs_list)
                if cli.per_trial_noise:
                    with PerRowNoise(torch, [noise_gen[seed_of[w]] for w, _ in batch]):
                        action, _ = adapter.policy.get_action(gr00t_obs)
                else:
                    action, _ = adapter.policy.get_action(gr00t_obs)
                for b, (w, o) in enumerate(batch):
                    extrinsic_cv = o["observation"][cli.camera_name]["extrinsic_cv"]
                    conns[w].send(("step", adapter.decode_action_chunk(action, extrinsic_cv, b)))

            commit()
            fill()
    finally:
        for c in conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    _, successes, success_rate = seq.write_results(cli, results_path, trials)
    print(f"\nSuccess rate: {successes}/{cli.num_trials} => {success_rate * 100:.1f}%")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
