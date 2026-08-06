"""Bridges RoboTwin's TASK_ENV observation/action format to
gr00t.policy.gr00t_policy.Gr00tPolicy for checkpoints finetuned on RoboTwin's plain
world-frame, absolute, native endpose representation -- i.e. Isaac-GR00T's
"equi_robotwin_2hand_quat" data config (EquiRoboTwin_2Hand_Quat_Config in
gr00t/experiment/data_config.py), whose bare modality keys are
{l_xyz, l_quat, r_xyz, r_quat, l_gripper, r_gripper} (state and action) and
{image, left_image, right_image} (video). This is exactly the schema
script/lerobot/convert_robotwin_to_lerobot.py (this repo) produces from RoboTwin's
raw HDF5: per-arm [x, y, z, qw, qx, qy, qz, gripper], no camera reprojection.

Contrast with gr00t_adapter.py (GR00TRoboTwinAdapter): that adapter targets a
different, bespoke "MotionTrans" schema whose EEF poses are reprojected into the
head camera's own frame at every timestep and represented as column-convention
rot6d, requiring nontrivial world<->camera matrix math on every step. This module
needs none of that: RoboTwin's native endpose is already world-frame and absolute,
and its quaternion order (wxyz) already matches the convention GR00T's own
pytorch3d-based rotation transforms use internally, so state/action pass through
untouched -- no scipy.spatial.transform.Rotation reordering, no extrinsic_cv.

Pick this adapter for a checkpoint trained on data from
script/lerobot/convert_robotwin_to_lerobot.py; pick gr00t_adapter.GR00TRoboTwinAdapter
for a checkpoint trained on the MotionTrans/camera-frame/rot6d schema instead.
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


def _parse_state_key(key: str) -> tuple:
    """'l_xyz' -> ('left', 'xyz'); 'right_gripper' -> ('right', 'gripper'); etc.
    Tolerant of both short (l_/r_) and long (left_/right_) arm prefixes, and of a
    few common synonyms for each field, since there's no checkpoint available in
    this environment to verify the exact key names against -- fail loudly and
    specifically (naming the offending key) rather than silently mis-mapping."""
    if "_" not in key:
        raise NotImplementedError(f"Unrecognized state/action key {key!r}: expected '<arm>_<field>'")
    head, rest = key.split("_", 1)
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
    camera = _VIDEO_CAMERA_ALIASES.get(video_key)
    if camera is None:
        raise NotImplementedError(
            f"Unrecognized video key {video_key!r}; expected one of {sorted(_VIDEO_CAMERA_ALIASES)}"
        )
    return camera


class GR00TRoboTwinEndposeAdapter:
    """Wraps a Gr00tPolicy and translates to/from RoboTwin's TASK_ENV.get_obs() /
    take_action(action_type='ee') formats using RoboTwin's native world-frame endpose
    directly -- no camera-frame reprojection, no rot6d. Supports 1-3 video keys
    (mapped to head_camera/left_camera/right_camera) and tolerates minor key-naming
    variation (l_/left_ etc.) in the checkpoint's modality config.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "cuda:0",
    ):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)

        self.state_keys = self.policy.modality_configs["state"].modality_keys
        self.action_keys = self.policy.modality_configs["action"].modality_keys
        video_keys = self.policy.modality_configs["video"].modality_keys
        self.language_key = self.policy.language_key

        if not video_keys:
            raise NotImplementedError("GR00TRoboTwinEndposeAdapter: no video keys in modality_configs")

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
        """No-op: Gr00tPolicy.get_action is single-shot / stateless per call."""

    def build_observation(self, obs: dict, instruction: str) -> dict:
        state = {}
        for key, (arm, field) in self.state_key_info.items():
            if field == "gripper":
                value = np.array([obs["endpose"][f"{arm}_gripper"]], dtype=np.float32)
            else:
                endpose = np.asarray(obs["endpose"][f"{arm}_endpose"], dtype=np.float32)  # [x,y,z,qw,qx,qy,qz]
                value = endpose[:3] if field == "xyz" else endpose[3:7]
            state[key] = value[None, None, :].astype(np.float32)  # (B=1, T=1, D)

        video = {}
        for key, camera in self.video_camera_map.items():
            rgb = np.asarray(obs["observation"][camera]["rgb"], dtype=np.uint8)
            video[key] = rgb[None, None, :, :, :]  # (B=1, T=1, H, W, 3), no crop/resize

        language = {self.language_key: [[instruction]]}
        return {"video": video, "state": state, "language": language}

    def decode_action_chunk(self, action: dict) -> list:
        """Returns a list of (16,) RoboTwin take_action(action_type='ee') vectors, one per
        predicted timestep, in [left_xyz(3), left_quat(4), left_gripper(1), right_xyz(3),
        right_quat(4), right_gripper(1)] order (envs/_base_task.py:1497-1521 is left-first).
        No extrinsic_cv needed: state/action are already world-frame absolute poses."""
        horizon = action[self.action_keys[0]].shape[1]
        waypoints = []
        for t in range(horizon):
            parts = []
            for arm in ("left", "right"):
                xyz = np.asarray(action[self._action_lookup[(arm, "xyz")]][0, t], dtype=np.float32)
                quat = np.asarray(action[self._action_lookup[(arm, "quat")]][0, t], dtype=np.float32)
                gripper = np.atleast_1d(
                    np.asarray(action[self._action_lookup[(arm, "gripper")]][0, t], dtype=np.float32)
                )
                parts.append(xyz)
                parts.append(quat)
                parts.append(gripper)
            waypoints.append(np.concatenate(parts).astype(np.float32))
        return waypoints
