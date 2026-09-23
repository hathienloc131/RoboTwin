"""Crash/hang-safe version of collect_data.py for a single (task, embodiment).

Every step is persisted to <save_path>/progress.json before it starts, so after a crash or a
watchdog kill (see collect_cross_embodiment.py) the next run knows exactly what was in flight:
  - a seed that was being planned when the process died is skipped for good,
  - an episode whose replay died / failed `max_replay_attempts` times has its seed replaced.

Exit codes: 0 = done, 3 = replay failed (restart me), 4 = gave up (too many failed seeds).
"""
import sys

sys.path.append("./")

import json
import os
import time
from argparse import ArgumentParser

import yaml

from envs import *
from script.collect_data import class_decorator, get_embodiment_config

EXIT_DONE, EXIT_RETRY, EXIT_GAVE_UP = 0, 3, 4


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


class Collector:

    def __init__(self, task_name, task_config, embodiment, save_path, episode_num, max_replay_attempts,
                 max_seed_factor):
        with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        if episode_num is not None:
            args["episode_num"] = episode_num

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        robot_file = embodiment_types[embodiment]["file_path"]

        args["task_name"] = task_name
        args["embodiment"] = [embodiment]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
        args["left_embodiment_config"] = get_embodiment_config(robot_file)
        args["right_embodiment_config"] = get_embodiment_config(robot_file)
        args["embodiment_name"] = embodiment
        args["task_config"] = task_config
        args["save_path"] = save_path

        self.args = args
        self.task_name = task_name
        self.embodiment = embodiment
        self.save_path = args["save_path"]
        self.max_replay_attempts = max_replay_attempts
        self.max_seed_tries = args["episode_num"] * max_seed_factor
        os.makedirs(self.save_path, exist_ok=True)

        self.seed_path = os.path.join(self.save_path, "seed.txt")
        self.progress_path = os.path.join(self.save_path, "progress.json")
        self.status_path = os.path.join(self.save_path, "status.json")
        self.info_path = os.path.join(self.save_path, "scene_info.json")

        self.seed_list = self._load_seeds()
        self.progress = load_json(self.progress_path, {})
        self.progress.setdefault("next_seed", max(self.seed_list) + 1 if self.seed_list else 0)
        self.progress.setdefault("skip_seeds", [])
        self.progress.setdefault("replay_fail", {})
        self.progress.setdefault("replaced_seeds", [])
        self.progress.setdefault("inflight", None)

        self.task_env = class_decorator(task_name)

    # ---------------------------------------------------------------- persistence
    def _load_seeds(self):
        if not os.path.exists(self.seed_path):
            return []
        with open(self.seed_path, "r") as f:
            return [int(s) for s in f.read().split()]

    def _save_seeds(self):
        atomic_write_text(self.seed_path, "".join(f"{s} " for s in self.seed_list))

    def _save_progress(self):
        atomic_write_json(self.progress_path, self.progress)

    def _set_inflight(self, phase, seed, episode):
        inflight = {"phase": phase, "seed": seed, "episode": episode, "t": time.time()} if phase else None
        self.progress["inflight"] = inflight
        self._save_progress()
        # heartbeat for the watchdog
        atomic_write_json(self.status_path, {"pid": os.getpid(), "inflight": inflight, "t": time.time()})

    def _hdf5_path(self, idx):
        return os.path.join(self.save_path, "data", f"episode{idx}.hdf5")

    def _traj_path(self, idx):
        return os.path.join(self.save_path, "_traj_data", f"episode{idx}.pkl")

    # ---------------------------------------------------------------- recovery
    def recover(self):
        """Account for whatever was running when the previous process died."""
        for sub in ["data", "video"]:  # half-written outputs from an interrupted merge
            d = os.path.join(self.save_path, sub)
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    if ".tmp" in fname:
                        os.remove(os.path.join(d, fname))

        inflight = self.progress["inflight"]
        if inflight is not None:
            if inflight["phase"] == "seed":
                seed = inflight["seed"]
                print(f"\033[93m[Resume] seed {seed} crashed/hung during planning, skipping it\033[0m")
                if seed not in self.seed_list and seed not in self.progress["skip_seeds"]:
                    self.progress["skip_seeds"].append(seed)
                self.progress["next_seed"] = max(self.progress["next_seed"], seed + 1)
            elif inflight["phase"] == "data":
                key = str(inflight["episode"])
                self.progress["replay_fail"][key] = self.progress["replay_fail"].get(key, 0) + 1
                print(f"\033[93m[Resume] episode {key} crashed/hung during replay "
                      f"({self.progress['replay_fail'][key]} time(s))\033[0m")
            self.progress["inflight"] = None

        # replace seeds whose replay keeps failing
        for key, count in sorted(self.progress["replay_fail"].items(), key=lambda kv: -int(kv[0])):
            idx = int(key)
            if count >= self.max_replay_attempts and idx < len(self.seed_list) and not os.path.exists(
                    self._hdf5_path(idx)):
                self._replace_seed(idx)
        self._save_seeds()
        self._save_progress()

    def _replace_seed(self, idx):
        """Drop seed at idx; move the last seed into its slot (same as data/process_stuck.py)."""
        last = len(self.seed_list) - 1
        bad_seed = self.seed_list[idx]
        if idx != last and os.path.exists(self._hdf5_path(last)):
            print(f"\033[91m[Resume] cannot replace episode {idx}: episode {last} is already collected\033[0m")
            return
        print(f"\033[91m[Resume] replacing episode {idx} (seed {bad_seed}): replay failed too often\033[0m")
        self.progress["replaced_seeds"].append(bad_seed)
        if bad_seed not in self.progress["skip_seeds"]:
            self.progress["skip_seeds"].append(bad_seed)
        if idx != last:
            os.replace(self._traj_path(last), self._traj_path(idx))
            self.seed_list[idx] = self.seed_list[last]
            last_fail = self.progress["replay_fail"].pop(str(last), None)
            self.progress["replay_fail"].pop(str(idx), None)
            if last_fail is not None:
                self.progress["replay_fail"][str(idx)] = last_fail
        else:
            if os.path.exists(self._traj_path(idx)):
                os.remove(self._traj_path(idx))
            self.progress["replay_fail"].pop(str(idx), None)
        self.seed_list.pop()

    # ---------------------------------------------------------------- phases
    def collect_seeds(self):
        args = self.args
        env = self.task_env
        args["need_plan"] = True
        args["save_data"] = False
        episode_num = args["episode_num"]

        while len(self.seed_list) < episode_num:
            seed = self.progress["next_seed"]
            if seed >= self.max_seed_tries:
                print(f"\033[91mGave up: {seed} seeds tried, only {len(self.seed_list)} succeeded\033[0m")
                return False
            self.progress["next_seed"] = seed + 1
            if seed in self.progress["skip_seeds"] or seed in self.seed_list:
                continue

            suc_num = len(self.seed_list)
            self._set_inflight("seed", seed, suc_num)
            try:
                env.setup_demo(now_ep_num=suc_num, seed=seed, **args)
                env.play_once()
                if env.plan_success and env.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {seed})")
                    env.save_traj_data(suc_num)
                    self.seed_list.append(seed)
                    self._save_seeds()
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {seed})")
                env.close_env()
            except Exception as e:
                print(f" -------------\nsimulate data episode {suc_num} fail! (seed = {seed})\nError: {e}\n -------------")
                try:
                    env.close_env()
                except Exception:
                    pass
                time.sleep(1 if not isinstance(e, UnStableError) else 0.3)
            self._set_inflight(None, None, None)
        return True

    def _dump_seg_id_map(self, idx):
        """Map raw segmentation ids to names: actor level = entity.per_scene_id, mesh level = render shape id."""
        actors, meshes = {}, {}
        # robot links are entities too: tag them so a robot-arm mask = isin(actor_seg, robot ids)
        robot_part = {}
        try:
            robot = self.task_env.robot
            same = robot.left_entity is robot.right_entity  # e.g. aloha-agilex: one urdf for both arms
            for side, art in [("left", robot.left_entity), ("right", robot.right_entity)]:
                for link in art.get_links():
                    robot_part[link.entity.per_scene_id] = "robot" if same else f"robot_{side}"
        except Exception as e:
            print(f"\033[93mWarning: could not find robot links: {e}\033[0m")
        robot_actor_ids, robot_mesh_ids = {}, {}
        try:
            for entity in self.task_env.scene.entities:
                part = robot_part.get(entity.per_scene_id, "object")
                actors[str(entity.per_scene_id)] = {"name": entity.name, "part": part}
                if part != "object":
                    robot_actor_ids.setdefault(part, []).append(entity.per_scene_id)
                for comp in entity.components:
                    for i, shape in enumerate(getattr(comp, "render_shapes", []) or []):
                        sid = getattr(shape, "per_scene_id", None)
                        if sid is not None:
                            meshes[str(sid)] = {"actor": entity.name, "actor_id": entity.per_scene_id, "shape": i,
                                                "part": part}
                            if part != "object":
                                robot_mesh_ids.setdefault(part, []).append(sid)
        except Exception as e:
            print(f"\033[93mWarning: could not build segmentation id map: {e}\033[0m")
        os.makedirs(os.path.join(self.save_path, "seg_id_map"), exist_ok=True)
        atomic_write_json(os.path.join(self.save_path, "seg_id_map", f"episode{idx}.json"), {
            "robot_actor_ids": robot_actor_ids,
            "robot_mesh_ids": robot_mesh_ids,
            "actor_segmentation": actors,
            "mesh_segmentation": meshes,
        })

    def collect_data(self):
        args = self.args
        env = self.task_env
        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True
        clear_cache_freq = args["clear_cache_freq"]

        for idx in range(args["episode_num"]):
            if os.path.exists(self._hdf5_path(idx)):
                continue
            print(f"\033[34mTask: {self.task_name} | {self.embodiment} | episode {idx}\033[0m")
            self._set_inflight("data", self.seed_list[idx], idx)
            try:
                env.setup_demo(now_ep_num=idx, seed=self.seed_list[idx], **args)
                traj_data = env.load_tran_data(idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                env.set_path_lst(args)
                self._dump_seg_id_map(idx)

                info = env.play_once()
                success = env.check_success()
                env.close_env(clear_cache=((idx + 1) % clear_cache_freq == 0))
                if not success:
                    raise RuntimeError("replay did not succeed")

                info_db = load_json(self.info_path, {})
                info_db[f"episode_{idx}"] = info
                atomic_write_json(self.info_path, info_db)

                env.merge_pkl_to_hdf5_video()
                env.remove_data_cache()
            except Exception as e:
                print(f"\033[91mReplay of episode {idx} failed: {e}\033[0m")
                self.progress["replay_fail"][str(idx)] = self.progress["replay_fail"].get(str(idx), 0) + 1
                self._set_inflight(None, None, None)
                return False  # restart in a fresh process; recover() replaces the seed if needed
            self.progress["replay_fail"].pop(str(idx), None)
            self._set_inflight(None, None, None)
        return True

    def run(self):
        self.recover()
        print(f"\033[93m[{self.task_name} | {self.embodiment}] seeds: {len(self.seed_list)}/{self.args['episode_num']}, "
              f"next seed: {self.progress['next_seed']}\033[0m")
        if not self.collect_seeds():
            return EXIT_GAVE_UP
        if not self.args["collect_data"]:
            return EXIT_DONE
        if not self.collect_data():
            return EXIT_RETRY

        if not self.progress.get("instructions_done"):
            self.generate_instructions()
            self.progress["instructions_done"] = True
            self._save_progress()
        return EXIT_DONE

    def generate_instructions(self):
        """Same as description/gen_episode_instructions.sh, but reads/writes self.save_path."""
        sys.path.append("./description/utils")
        from generate_episode_instructions import extract_episodes_from_scene_info, generate_episode_descriptions

        episodes = extract_episodes_from_scene_info(load_json(self.info_path, {}))
        results = generate_episode_descriptions(self.task_name, episodes, self.args["language_num"])
        out_dir = os.path.join(self.save_path, "instructions")
        os.makedirs(out_dir, exist_ok=True)
        for desc in results:
            atomic_write_json(os.path.join(out_dir, f"episode{desc['episode_index']}.json"), {
                "seen": desc.get("seen", []),
                "unseen": desc.get("unseen", []),
            })


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("embodiment", type=str)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Output folder (default: <save_path in yml>/<embodiment>/<task>)")
    parser.add_argument("-n", "--episode_num", type=int, default=None)
    parser.add_argument("--max_replay_attempts", type=int, default=2,
                        help="Replace an episode's seed after its replay fails/hangs this many times")
    parser.add_argument("--max_seed_factor", type=int, default=20,
                        help="Give up after episode_num * factor seeds tried")
    a = parser.parse_args()

    if a.save_path is None:
        with open(f"./task_config/{a.task_config}.yml", "r", encoding="utf-8") as f:
            a.save_path = os.path.join(yaml.safe_load(f)["save_path"], a.embodiment, a.task_name)
    collector = Collector(a.task_name, a.task_config, a.embodiment, a.save_path, a.episode_num,
                          a.max_replay_attempts, a.max_seed_factor)
    sys.exit(collector.run())
