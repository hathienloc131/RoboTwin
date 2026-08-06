"""Bridges RoboTwin's TASK_ENV observation/action format to
gr00t.model.policy.Gr00tPolicy (Isaac-GR00T N1.5's classical API, in
/Users/lochathien/Documents/Code/Isaac-GR00T on this machine) for checkpoints
finetuned on RoboTwin's plain world-frame, absolute, native endpose
representation -- i.e. Isaac-GR00T's "equi_robotwin_2hand_quat" data config
(EquiRoboTwin_2Hand_Quat_Config in gr00t/experiment/data_config.py), whose
flat modality keys are {state,action}.{l_xyz, l_quat, r_xyz, r_quat, l_gripper,
r_gripper} and video.{image, left_image, right_image}. This is exactly the
schema script/lerobot/convert_robotwin_to_lerobot.py (this repo) produces from
RoboTwin's raw HDF5: per-arm [x, y, z, qw, qx, qy, qz, gripper], no camera
reprojection.

Contrast with gr00t_adapter.py (GR00TRoboTwinAdapter): that adapter targets a
*different* GR00T policy module (gr00t.policy.gr00t_policy.Gr00tPolicy, a nested
video/state/language dict API that only exists on some Isaac-GR00T branches/
installs -- NOT the one checked out at /Users/lochathien/Documents/Code/Isaac-GR00T,
which only has gr00t.model.policy.Gr00tPolicy) and a bespoke "MotionTrans"
schema whose EEF poses are reprojected into the head camera's own frame at every
timestep and represented as column-convention rot6d. This module needs none of
that: RoboTwin's native endpose is already world-frame and absolute, and its
quaternion order (wxyz) passes straight through -- EquiRoboTwin_2Hand_Quat_Config's
transform() applies no rotation-representation conversion to l_quat/r_quat (no
target_rotations entry for them), so whatever order the training LeRobot dataset
stored is what the checkpoint learned; script/lerobot/convert_robotwin_to_lerobot.py
writes RoboTwin's native wxyz unreordered, so this adapter does the same.

Unlike gr00t.policy.gr00t_policy.Gr00tPolicy, gr00t.model.policy.Gr00tPolicy:
  - requires an explicit modality_config + modality_transform, built from a
    "data config" object (gr00t.experiment.data_config.load_data_config(name)) --
    it does not introspect these from the checkpoint itself.
  - requires <model_path>/experiment_cfg/metadata.json (normalization stats,
    keyed by embodiment_tag.value), produced automatically by the finetuning
    pipeline's checkpoint-save step.
  - takes/returns FLAT dicts with the modality keys' full "video."/"state."/
    "action."/"annotation." prefixes intact (e.g. "state.l_xyz"), not the
    prefix-stripped bare names gr00t.policy.gr00t_policy.Gr00tPolicy exposes,
    and no batch dimension for a single unbatched observation: video is
    (T, H, W, C), state is (T, D), annotation is (T,). get_action() returns a
    single dict (not a (action, info) tuple).
  - EmbodimentTag values on this branch are lowercase ("new_embodiment", not
    "NEW_EMBODIMENT") -- gr00t.data.embodiment_tags.EmbodimentTag(embodiment_tag)
    looks up by value, so passing the wrong case raises ValueError.

Pick this adapter for a checkpoint trained on data from
script/lerobot/convert_robotwin_to_lerobot.py; pick gr00t_adapter.GR00TRoboTwinAdapter
only if you actually have the other gr00t.policy.gr00t_policy-based install/branch.
"""

import numpy as np

_ARM_ALIASES = {"l": "left", "left": "left", "r": "right", "right": "right"}
_FIELD_ALIASES = {
    "xyz": "xyz",
    "pos": "xyz",
    "position": "xyz",
    "quat": "quat",
    "quaternion": "quat",
    "rot": "quat",
    "gripper": "gripper",
}
_VIDEO_CAMERA_ALIASES = {
    "image": "head_camera",
    "head_image": "head_camera",
    "head_camera": "head_camera",
    "left_image": "left_camera",
    "left_wrist_image": "left_camera",
    "left_camera": "left_camera",
    "right_image": "right_camera",
    "right_wrist_image": "right_camera",
    "right_camera": "right_camera",
}
# Modality-key namespace prefixes gr00t.model.policy's flat dict format uses
# (e.g. "state.l_xyz") -- stripped before alias lookup, but the ORIGINAL prefixed
# string is always what's used as the actual observation/action dict key.
_KNOWN_PREFIXES = ("video.", "state.", "action.")


def _strip_known_prefix(key: str) -> str:
    for prefix in _KNOWN_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _parse_state_key(key: str) -> tuple:
    """'state.l_xyz' -> ('left', 'xyz'); 'action.right_gripper' -> ('right', 'gripper');
    etc. Tolerant of both short (l_/r_) and long (left_/right_) arm prefixes, and of a
    few common synonyms for each field, since there's no checkpoint available in this
    environment to verify the exact key names against -- fail loudly and specifically
    (naming the offending key) rather than silently mis-mapping."""
    bare = _strip_known_prefix(key)
    if "_" not in bare:
        raise NotImplementedError(f"Unrecognized state/action key {key!r}: expected '<arm>_<field>'")
    head, rest = bare.split("_", 1)
    arm = _ARM_ALIASES.get(head)
    if arm is None:
        raise NotImplementedError(
            f"Unrecognized arm prefix in state/action key {key!r}: {head!r}; expected one of l_/left_/r_/right_"
        )
    field = _FIELD_ALIASES.get(rest)
    if field is None:
        raise NotImplementedError(
            f"Unrecognized field in state/action key {key!r}: {rest!r}; "
            "expected one of xyz/pos/position, quat/quaternion/rot, gripper"
        )
    return arm, field


def _resolve_camera(video_key: str) -> str:
    bare = _strip_known_prefix(video_key)
    camera = _VIDEO_CAMERA_ALIASES.get(bare)
    if camera is None:
        raise NotImplementedError(
            f"Unrecognized video key {video_key!r}; expected one of {sorted(_VIDEO_CAMERA_ALIASES)}"
        )
    return camera


class GR00TRoboTwinEndposeAdapter:
    """Wraps a gr00t.model.policy.Gr00tPolicy and translates to/from RoboTwin's
    TASK_ENV.get_obs() / take_action(action_type='ee') formats using RoboTwin's
    native world-frame endpose directly -- no camera-frame reprojection, no rot6d.
    Supports 1-3 video keys (mapped to head_camera/left_camera/right_camera) and
    tolerates minor key-naming variation (l_/left_ etc.) in the data config.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        data_config: str = "equi_robotwin_2hand_quat",
        device: str = "cuda:0",
        denoising_steps: int = None,
    ):
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy

        cfg = load_data_config(data_config)
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=cfg.modality_config(),
            modality_transform=cfg.transform(),
            denoising_steps=denoising_steps,
            device=device,
        )

        modality_config = self.policy.get_modality_config()
        self.state_keys = modality_config["state"].modality_keys
        self.action_keys = modality_config["action"].modality_keys
        video_keys = modality_config["video"].modality_keys
        self.language_key = modality_config["language"].modality_keys[0]

        if not video_keys:
            raise NotImplementedError("GR00TRoboTwinEndposeAdapter: no video keys in modality_config")

        self.state_key_info = {key: _parse_state_key(key) for key in self.state_keys}
        self.action_key_info = {key: _parse_state_key(key) for key in self.action_keys}
        self.video_camera_map = {key: _resolve_camera(key) for key in video_keys}

        mapped_cameras = list(self.video_camera_map.values())
        if len(set(mapped_cameras)) != len(mapped_cameras):
            print(
                f"WARNING: GR00TRoboTwinEndposeAdapter: multiple video keys map to the same "
                f"RoboTwin camera: {self.video_camera_map}"
            )

        state_pairs = set(self.state_key_info.values())
        action_pairs = set(self.action_key_info.values())
        if state_pairs != action_pairs:
            raise NotImplementedError(
                "GR00TRoboTwinEndposeAdapter: state and action modality keys don't cover the same "
                f"(arm, field) pairs -- state={state_pairs}, action={action_pairs}"
            )

        # Inverse of action_key_info, built once since decode_action_chunk runs in the hot trial loop.
        self._action_lookup = {pair: key for key, pair in self.action_key_info.items()}

    def reset(self) -> None:
        """Clears Gr00tPolicy's internal temporal-ensembling buffers between episodes.
        Harmless/no-op in the default (no smoothing) configuration, but real
        Gr00tPolicy.reset() state exists (unlike gr00t.policy.gr00t_policy's fully
        stateless single-shot API), so this is a genuine call, not a stub."""
        self.policy.reset()

    def build_observation(self, obs: dict, instruction: str) -> dict:
        """Returns a FLAT dict (gr00t.model.policy.Gr00tPolicy's unbatched format):
        each value shaped (T=1, ...) with no leading batch dimension."""
        observation = {}
        for key, (arm, field) in self.state_key_info.items():
            if field == "gripper":
                value = np.array([obs["endpose"][f"{arm}_gripper"]], dtype=np.float32)
            else:
                endpose = np.asarray(obs["endpose"][f"{arm}_endpose"], dtype=np.float32)  # [x,y,z,qw,qx,qy,qz]
                value = endpose[:3] if field == "xyz" else endpose[3:7]
            observation[key] = value[None, :].astype(np.float32)  # (T=1, D)

        for key, camera in self.video_camera_map.items():
            rgb = np.asarray(obs["observation"][camera]["rgb"], dtype=np.uint8)
            observation[key] = rgb[None, :, :, :]  # (T=1, H, W, 3), no crop/resize

        observation[self.language_key] = [instruction]  # (T=1,)
        return observation

    def decode_action_chunk(self, action: dict) -> list:
        """Returns a list of (16,) RoboTwin take_action(action_type='ee') vectors, one per
        predicted timestep, in [left_xyz(3), left_quat(4), left_gripper(1), right_xyz(3),
        right_quat(4), right_gripper(1)] order (envs/_base_task.py:1497-1521 is left-first).
        No extrinsic_cv needed: state/action are already world-frame absolute poses.

        `action` is Gr00tPolicy.get_action()'s return value directly: a flat dict keyed by
        the same prefixed action keys (e.g. "action.l_xyz"), each shaped (horizon, D) --
        no batch dimension (gr00t.model.policy squeezes it off for unbatched input)."""
        horizon = action[self.action_keys[0]].shape[0]
        waypoints = []
        for t in range(horizon):
            parts = []
            for arm in ("left", "right"):
                xyz = np.asarray(action[self._action_lookup[(arm, "xyz")]][t], dtype=np.float32)
                quat = np.asarray(action[self._action_lookup[(arm, "quat")]][t], dtype=np.float32)
                gripper = np.atleast_1d(
                    np.asarray(action[self._action_lookup[(arm, "gripper")]][t], dtype=np.float32)
                )
                parts.append(xyz)
                parts.append(quat)
                parts.append(gripper)
            waypoints.append(np.concatenate(parts).astype(np.float32))
        return waypoints
