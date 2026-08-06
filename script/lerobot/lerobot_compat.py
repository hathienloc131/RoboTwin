"""
Import shim for the `lerobot` package.

Different `lerobot` versions live at different import paths:
  - newer releases (e.g. lerobot>=0.3, this repo's `lerobot` conda env):
        lerobot.datasets.lerobot_dataset / lerobot.datasets.utils
    and expect HWC (H, W, C) numpy image arrays in `add_frame`.
  - older commits (e.g. the one pinned by policy/pi0's pyproject.toml,
    and used by policy/GO1/scripts/convert_aloha_data_to_lerobot_robotwin.py):
        lerobot.common.datasets.lerobot_dataset / lerobot.common.datasets.utils
    and expect CHW (C, H, W) numpy image arrays.

`import_lerobot()` tries the new path first and falls back to the old one, so
the converter scripts in this directory work against either install without
the caller needing to know in advance which one is on PYTHONPATH.
"""

import logging

logger = logging.getLogger(__name__)


class LerobotBindings:
    """Small holder for whatever we managed to import, plus which variant it is."""

    def __init__(self, variant, LeRobotDataset, LeRobotDatasetMetadata, write_episode, write_episode_stats, write_info, write_task):
        self.variant = variant
        self.LeRobotDataset = LeRobotDataset
        self.LeRobotDatasetMetadata = LeRobotDatasetMetadata
        self.write_episode = write_episode
        self.write_episode_stats = write_episode_stats
        self.write_info = write_info
        self.write_task = write_task

    @property
    def default_image_layout(self):
        return "hwc" if self.variant == "new" else "chw"


def import_lerobot() -> LerobotBindings:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import write_episode, write_episode_stats, write_info, write_task

        logger.info("lerobot_compat: using NEW import path (lerobot.datasets.*), default image layout = hwc")
        return LerobotBindings("new", LeRobotDataset, LeRobotDatasetMetadata, write_episode, write_episode_stats, write_info, write_task)
    except ImportError:
        pass

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.common.datasets.utils import write_episode, write_episode_stats, write_info, write_task

        logger.info("lerobot_compat: using OLD import path (lerobot.common.datasets.*), default image layout = chw")
        return LerobotBindings("old", LeRobotDataset, LeRobotDatasetMetadata, write_episode, write_episode_stats, write_info, write_task)
    except ImportError as e:
        raise ImportError(
            "Could not import `lerobot` from either `lerobot.datasets.*` or `lerobot.common.datasets.*`. "
            "Install it first, e.g.:\n"
            "  conda activate lerobot  # or: pip install -r script/lerobot/requirements.txt\n"
            "then re-run this script."
        ) from e
