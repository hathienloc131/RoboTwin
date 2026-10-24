"""GR00T <-> RoboTwin in JOINT space, for checkpoints trained on the SSG robot data
(lerobot-convert data_loader_for_robotwin_ssg.py, ego_gr00t
examples/MotionTrans/scene_graph_3cam_joint_config.py).

State and action are the 14-D joint vector in RoboTwin's order -- left arm 6, left gripper,
right arm 6, right gripper -- split into the groups left_arm_joints / left_gripper_joint /
right_arm_joints / right_gripper_joint. The model predicts absolute next joints, which go to
TASK_ENV.take_action(action_type="qpos") unchanged. Images are passed at RoboTwin's native
480x640, exactly as the converter stored them (no crop / resize).
"""

import numpy as np

JOINT_SLICES = {
    "left_arm_joints": slice(0, 6),
    "left_gripper_joint": slice(6, 7),
    "right_arm_joints": slice(7, 13),
    "right_gripper_joint": slice(13, 14),
}
CAMERAS = {"camera0": "head_camera", "camera1": "right_camera", "camera2": "left_camera"}


class GR00TRoboTwinJointAdapter:
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
        if set(self.state_keys) != set(JOINT_SLICES) or set(self.action_keys) != set(JOINT_SLICES):
            raise NotImplementedError(
                f"joint adapter expects groups {list(JOINT_SLICES)}, got state={self.state_keys}, "
                f"action={self.action_keys}"
            )
        video_keys = self.policy.modality_configs["video"].modality_keys
        cameras = dict(CAMERAS, camera0=camera_name)
        unknown = [k for k in video_keys if k not in cameras]
        if unknown or not video_keys:
            raise NotImplementedError(f"joint adapter supports video keys {list(cameras)}, got {video_keys}")
        self.video_cameras = {k: cameras[k] for k in video_keys}
        self.language_key = self.policy.language_key

    def reset(self) -> None:
        """No-op: Gr00tPolicy.get_action is stateless per call."""

    def build_observation(self, obs: dict, instruction: str) -> dict:
        joints = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        state = {k: joints[s][None, None, :] for k, s in JOINT_SLICES.items()}  # (B=1, T=1, D)
        video = {
            k: np.asarray(obs["observation"][cam]["rgb"], dtype=np.uint8)[None, None]
            for k, cam in self.video_cameras.items()
        }
        language = {self.language_key: [[instruction]]}
        return {"video": video, "state": state, "language": language}

    def decode_action_chunk(self, action: dict, extrinsic_cv=None, batch_idx: int = 0) -> list:
        """List of (14,) take_action(action_type="qpos") vectors, one per predicted step."""
        horizon = action[self.action_keys[0]].shape[1]
        order = list(JOINT_SLICES)
        return [
            np.concatenate([np.asarray(action[k][batch_idx, t], dtype=np.float32) for k in order])
            for t in range(horizon)
        ]
