"""Bridges RoboTwin's TASK_ENV observation/action format to
gr00t.policy.gr00t_policy.Gr00tPolicy for checkpoints finetuned with
examples/MotionTrans/motiontrans_config.py (in the ego_gr00t repo): bimanual
[right_eef_9d, right_gripper, left_eef_9d, left_gripper] state/action groups, camera-frame
EEF poses, single "camera0" video key.

Checkpoints trained on this schema store EEF state/action relative to the head camera's
own pose at each timestep, not RoboTwin's native SAPIEN world frame (see
data_loader_for_robotwin_motiontrans.py in the lerobot-convert repo, the converter that
produced this training data -- it is the ground truth for every convention below). This
module performs the reprojection in both directions: RoboTwin's world-frame endpose ->
camera-frame model input, and the model's camera-frame predicted action -> world-frame
target pose for TASK_ENV.take_action(action_type='ee').

IMPORTANT: the 9-D EEF representation here is [xyz, rot6d] where rot6d is the first two
COLUMNS of the rotation matrix (data_loader_for_robotwin_motiontrans.py::mat_to_pos_rot6d).
This is the opposite convention from gr00t.data.state_action.pose.EndEffectorPose /
gr00t.data.state_action.camera_projection.rot6d_to_matrix, which both use the first two
ROWS. Do not reuse gr00t's row-convention helpers here -- GR00T's own pipeline reinterprets
these floats consistently (always as rows) both when building training targets and when
decoding predicted actions, so its internal math is self-consistent regardless of the "true"
geometric convention; but this module is the one place that must recover the actual SAPIEN
pose those floats encode, so it must match the converter's column convention exactly.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# RoboTwin head-camera preprocessing must match
# data_loader_for_robotwin_motiontrans.py::center_crop_resize exactly: center-crop to
# (CROP_H, CROP_W) -- a no-op when the camera is already this size (Large_D435, 640x480) --
# then resize to (OUT_H, OUT_W).
_CROP_H, _CROP_W = 480, 640
_OUT_H, _OUT_W = 600, 960

_BIMANUAL_STATE_KEYS = {"right_eef_9d", "right_gripper", "left_eef_9d", "left_gripper"}


def _extrinsic_to_mat4(extrinsic_cv: np.ndarray) -> np.ndarray:
    """(3, 4) world->camera OpenCV extrinsic -> (4, 4)."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = np.asarray(extrinsic_cv, dtype=np.float64)
    return mat


def _endpose_to_mat4(endpose: np.ndarray) -> np.ndarray:
    """RoboTwin endpose [x, y, z, qw, qx, qy, qz] (SAPIEN world frame) -> (4, 4)."""
    endpose = np.asarray(endpose, dtype=np.float64)
    pos, quat_wxyz = endpose[:3], endpose[3:7]
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = pos
    return mat


def _mat4_to_endpose(mat: np.ndarray) -> np.ndarray:
    """(4, 4) -> RoboTwin endpose [x, y, z, qw, qx, qy, qz]."""
    quat_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([mat[:3, 3], quat_wxyz]).astype(np.float32)


def mat_to_pos_colrot6d(mat: np.ndarray) -> np.ndarray:
    """(4, 4) -> (9,) [x, y, z, col0(3), col1(3)] -- column-convention rot6d, matching
    data_loader_for_robotwin_motiontrans.py::mat_to_pos_rot6d exactly."""
    pos = mat[:3, 3]
    col0, col1 = mat[:3, 0], mat[:3, 1]
    return np.concatenate([pos, col0, col1]).astype(np.float32)


def pos_colrot6d_to_mat(pose9: np.ndarray) -> np.ndarray:
    """Inverse of mat_to_pos_colrot6d: (9,) -> (4, 4) via Gram-Schmidt on the two columns."""
    pose9 = np.asarray(pose9, dtype=np.float64)
    pos, col0, col1 = pose9[:3], pose9[3:6], pose9[6:9]
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 - np.dot(col0, col1) * col0
    col1 = col1 / np.linalg.norm(col1)
    col2 = np.cross(col0, col1)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.stack([col0, col1, col2], axis=1)
    mat[:3, 3] = pos
    return mat


def world_endpose_to_state9(endpose_xyzquat: np.ndarray, extrinsic_cv: np.ndarray) -> np.ndarray:
    """RoboTwin world-frame endpose + (3, 4) OpenCV world->camera extrinsic -> (9,)
    camera-frame [xyz, colrot6d] state, matching the distribution
    data_loader_for_robotwin_motiontrans.py::raw_data_loader built for training
    (robotN_in_cam = cam_extrinsic @ world_pose)."""
    T_world = _endpose_to_mat4(endpose_xyzquat)
    T_cam = _extrinsic_to_mat4(extrinsic_cv)
    T_cam_frame = T_cam @ T_world
    return mat_to_pos_colrot6d(T_cam_frame)


def cam_action9_to_world_xyzquat(action9: np.ndarray, extrinsic_cv: np.ndarray) -> np.ndarray:
    """Inverse: (9,) camera-frame predicted EEF pose + (3, 4) extrinsic -> RoboTwin
    world-frame [x, y, z, qw, qx, qy, qz], ready for
    TASK_ENV.take_action(action_type='ee')."""
    T_cam_frame = pos_colrot6d_to_mat(action9)
    T_cam = _extrinsic_to_mat4(extrinsic_cv)
    T_world = np.linalg.inv(T_cam) @ T_cam_frame
    return _mat4_to_endpose(T_world)


def preprocess_head_camera(rgb: np.ndarray) -> np.ndarray:
    """Center-crop then resize to match the resolution GR00T was trained on."""
    ih, iw = rgb.shape[:2]
    y0 = (ih - _CROP_H) // 2
    x0 = (iw - _CROP_W) // 2
    cropped = rgb[y0 : y0 + _CROP_H, x0 : x0 + _CROP_W]
    return cv2.resize(cropped, (_OUT_W, _OUT_H), interpolation=cv2.INTER_LINEAR)


class GR00TRoboTwinAdapter:
    """Wraps a Gr00tPolicy and translates to/from RoboTwin's TASK_ENV.get_obs() /
    take_action(action_type='ee') formats, including the camera<->world EEF reprojection
    above. Bimanual-only (right_/left_ prefixed groups) -- for a single-arm checkpoint
    (bare eef_9d/gripper groups, see the note in motiontrans_config.py), extend
    build_observation/decode_action_chunk rather than repurposing this class.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "cuda:0",
        camera_name: str = "head_camera",
    ):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        self.camera_name = camera_name

        self.state_keys = self.policy.modality_configs["state"].modality_keys
        self.action_keys = self.policy.modality_configs["action"].modality_keys
        video_keys = self.policy.modality_configs["video"].modality_keys
        if len(video_keys) != 1:
            raise NotImplementedError(
                f"GR00TRoboTwinAdapter only supports a single video key, got {video_keys}"
            )
        self.video_key = video_keys[0]
        self.language_key = self.policy.language_key

        if set(self.state_keys) != _BIMANUAL_STATE_KEYS or set(self.action_keys) != _BIMANUAL_STATE_KEYS:
            raise NotImplementedError(
                "GR00TRoboTwinAdapter only supports the bimanual MotionTrans schema "
                f"{_BIMANUAL_STATE_KEYS}, got state={self.state_keys}, action={self.action_keys}. "
                "See the single-arm note in examples/MotionTrans/motiontrans_config.py."
            )

    def reset(self) -> None:
        """No-op: Gr00tPolicy.get_action is single-shot / stateless per call."""

    def build_observation(self, obs: dict, instruction: str) -> dict:
        extrinsic_cv = obs["observation"][self.camera_name]["extrinsic_cv"]
        rgb = preprocess_head_camera(obs["observation"][self.camera_name]["rgb"])

        state = {}
        for key in self.state_keys:
            arm = "right" if key.startswith("right_") else "left"
            if key.endswith("_eef_9d"):
                endpose = np.asarray(obs["endpose"][f"{arm}_endpose"], dtype=np.float32)
                value = world_endpose_to_state9(endpose, extrinsic_cv)
            else:  # *_gripper
                value = np.array([obs["endpose"][f"{arm}_gripper"]], dtype=np.float32)
            state[key] = value[None, None, :]  # (B=1, T=1, D)

        video = {self.video_key: rgb[None, None, :, :, :].astype(np.uint8)}  # (B=1, T=1, H, W, 3)
        language = {self.language_key: [[instruction]]}  # (B=1, T=1)

        return {"video": video, "state": state, "language": language}

    def decode_action_chunk(self, action: dict, extrinsic_cv) -> list:
        """Returns a list of (16,) RoboTwin take_action(action_type='ee') vectors, one per
        predicted timestep, in [left_xyzquat(7), left_gripper(1), right_xyzquat(7),
        right_gripper(1)] order (envs/_base_task.py:1497-1520 is left-first)."""
        horizon = action[self.action_keys[0]].shape[1]
        waypoints = []
        for t in range(horizon):
            parts = []
            for arm in ("left", "right"):
                world_pose = cam_action9_to_world_xyzquat(action[f"{arm}_eef_9d"][0, t], extrinsic_cv)
                gripper = float(action[f"{arm}_gripper"][0, t, 0])
                parts.append(world_pose)
                parts.append(np.array([gripper], dtype=np.float32))
            waypoints.append(np.concatenate(parts).astype(np.float32))
        return waypoints
