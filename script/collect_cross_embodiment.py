"""Collect every task for every embodiment, restarting the worker when it crashes or hangs.

Each (task, embodiment) runs script/collect_data_resumable.py in its own process group and writes to
<data_root>/<embodiment>/<task>/. The worker updates status.json at the start of every
seed/episode; if one step runs longer than --episode_timeout the whole process group is killed and
restarted, and the worker skips / replaces the seed that hung. Finished runs get a `.done` marker,
so re-running this script simply picks up where it left off.
"""
import glob
import json
import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

DEFAULT_EMBODIMENTS = ["aloha-agilex", "franka-panda", "ARX-X5", "ur5-wsg"]
EXIT_DONE, EXIT_RETRY, EXIT_GAVE_UP = 0, 3, 4


def log(msg, log_file=None):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(f"\033[96m{line}\033[0m", flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def count_progress(save_path):
    """Number of successful seeds + collected episodes; used to detect restarts that make no progress."""
    seeds = 0
    seed_file = os.path.join(save_path, "seed.txt")
    if os.path.exists(seed_file):
        with open(seed_file) as f:
            seeds = len(f.read().split())
    data_dir = os.path.join(save_path, "data")
    episodes = len([f for f in os.listdir(data_dir) if f.endswith(".hdf5")]) if os.path.isdir(data_dir) else 0
    return seeds + episodes


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_worker(cmd, save_path, worker_log, a):
    """Run one worker process; return its exit code, or None if the watchdog killed it."""
    status_path = os.path.join(save_path, "status.json")
    if os.path.exists(status_path):
        os.remove(status_path)
    start = time.time()
    with open(worker_log, "a") as out:
        out.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)} =====\n")
        out.flush()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "PYTHONWARNINGS": "ignore::UserWarning"})
        while True:
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            status = read_json(status_path)
            if status is None:
                if time.time() - start > a.startup_timeout:
                    reason = f"no heartbeat {a.startup_timeout}s after start"
                    break
                continue
            inflight = status.get("inflight")
            if inflight and time.time() - inflight["t"] > a.episode_timeout:
                reason = f"{inflight['phase']} step (episode {inflight['episode']}, seed {inflight['seed']}) " \
                         f"stuck > {a.episode_timeout}s"
                break
            if not inflight and time.time() - status["t"] > a.episode_timeout:
                reason = f"idle > {a.episode_timeout}s (merging / instruction generation stuck?)"
                break
    kill_group(proc)
    return reason


def main():
    p = ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names, or a path to a .txt file (one per line). Default: every task in envs/")
    p.add_argument("--embodiments", nargs="+", default=DEFAULT_EMBODIMENTS)
    p.add_argument("--task_config", default="demo_clean_seg_depth")
    p.add_argument("--data_root", default="./data", help="Output: <data_root>/<embodiment>/<task>/")
    p.add_argument("-n", "--episode_num", type=int, default=None)
    p.add_argument("--episode_timeout", type=int, default=900, help="Kill worker if one seed/episode exceeds this (s)")
    p.add_argument("--startup_timeout", type=int, default=600, help="Kill worker if it does not start within (s)")
    p.add_argument("--max_no_progress_restarts", type=int, default=8,
                   help="Give up a (task, embodiment) after this many consecutive restarts without new data")
    p.add_argument("--max_replay_attempts", type=int, default=2)
    p.add_argument("--max_seed_factor", type=int, default=20)
    p.add_argument("--retry_gave_up", action="store_true", help="Retry (task, embodiment) pairs marked .gave_up")
    a = p.parse_args()

    tasks = a.tasks
    if not tasks:
        tasks = sorted(os.path.basename(f)[:-3] for f in glob.glob("envs/*.py")
                       if not os.path.basename(f).startswith("_"))
    elif len(tasks) == 1 and os.path.isfile(tasks[0]):
        with open(tasks[0]) as f:
            tasks = [t.strip() for t in f if t.strip() and not t.startswith("#")]

    os.makedirs("logs", exist_ok=True)
    main_log = os.path.join("logs", f"collect_{a.task_config}.log")
    summary = {}
    log(f"{len(a.embodiments)} embodiments x {len(tasks)} tasks -> {a.data_root}/<embodiment>/<task>", main_log)

    for embodiment in a.embodiments:
        for task in tasks:
            save_path = os.path.join(a.data_root, embodiment, task)
            os.makedirs(save_path, exist_ok=True)
            done_marker = os.path.join(save_path, ".done")
            gave_up_marker = os.path.join(save_path, ".gave_up")
            key = f"{task} | {embodiment}"

            if os.path.exists(done_marker):
                summary[key] = "done (skipped)"
                continue
            if os.path.exists(gave_up_marker):
                if not a.retry_gave_up:
                    summary[key] = "gave up (skipped)"
                    continue
                os.remove(gave_up_marker)

            cmd = [sys.executable, "script/collect_data_resumable.py", task, a.task_config, embodiment,
                   "--save_path", save_path, "--max_replay_attempts", str(a.max_replay_attempts),
                   "--max_seed_factor", str(a.max_seed_factor)]
            if a.episode_num is not None:
                cmd += ["-n", str(a.episode_num)]
            worker_log = os.path.join(save_path, "collect.log")

            no_progress, attempt = 0, 0
            log(f"START {key} -> {save_path}", main_log)
            while True:
                attempt += 1
                before = count_progress(save_path)
                result = run_worker(cmd, save_path, worker_log, a)
                after = count_progress(save_path)

                if result == EXIT_DONE:
                    open(done_marker, "w").close()
                    summary[key] = "done"
                    log(f"DONE  {key} (attempt {attempt})", main_log)
                    break
                if result == EXIT_GAVE_UP:
                    open(gave_up_marker, "w").close()
                    summary[key] = "gave up: too many failed seeds"
                    log(f"GAVE UP {key}: too many failed seeds", main_log)
                    break

                why = result if isinstance(result, str) else f"exit code {result}"
                no_progress = 0 if after > before else no_progress + 1
                log(f"RESTART {key} (attempt {attempt}): {why}; progress {before}->{after}, "
                    f"{no_progress}/{a.max_no_progress_restarts} restarts without progress", main_log)
                if no_progress >= a.max_no_progress_restarts:
                    open(gave_up_marker, "w").close()
                    summary[key] = f"gave up: no progress ({why})"
                    log(f"GAVE UP {key}: no progress after {no_progress} restarts", main_log)
                    break
                # clear the renderer's stale episode cache before restarting
                subprocess.run(["rm", "-rf", os.path.join(save_path, ".cache")])
                time.sleep(3)

            subprocess.run(["rm", "-rf", os.path.join(save_path, ".cache")])

    log("===== SUMMARY =====", main_log)
    for key, state in summary.items():
        log(f"{key}: {state}", main_log)


if __name__ == "__main__":
    main()
