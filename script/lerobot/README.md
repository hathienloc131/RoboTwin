# RoboTwin → LeRobot converter

Converts a RoboTwin task folder (e.g. `data/grab_roller_aloha-agilex_clean_50/`,
as produced by `script/collect_data.py`) directly into a LeRobot v2.1 dataset —
no intermediate ALOHA-style HDF5 step needed (contrast with
`policy/GO1/scripts/process_data.py` + `convert_aloha_data_to_lerobot_robotwin.py`,
which does require that intermediate format).

## Dataset schema

- **Action / state = end-effector pose (`endpose`)**, not joint angles. Both
  `observation.state` and `action` are a 16-dim float32 vector:
  ```
  [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
   right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
  ```
  Quaternions are **wxyz** (RoboTwin's native order) — kept as-is, not reordered to
  xyzw. Watch out if you consume this with `scipy.spatial.transform.Rotation`,
  which defaults to xyzw.
- `action[t]` = `state[t+1]`, i.e. the *next* frame's absolute pose — this mirrors
  how RoboTwin's own policy converters (`policy/DP/process_data.py`,
  `policy/GO1/scripts/process_data.py`) build action labels from RoboTwin's
  inherently-absolute recorded data. The last frame of each episode has no
  successor and is dropped, so each converted episode has `T-1` frames.
- **Cameras**: `head_camera`, `left_camera`, `right_camera` only.
  `front_camera` is present in the raw HDF5 but dropped here, matching what
  every existing RoboTwin policy converter uses downstream.
- **Images**: stored in LeRobot "video" mode (mp4-encoded per episode/camera),
  at RoboTwin's native collected resolution (no resizing) — read from the data
  itself, so it works for both the default D435 (320×240) and Large_D435
  (640×480) camera configs.

## Usage

Run with an environment that has `lerobot` installed — on this machine that's
the dedicated `lerobot` conda env:

```bash
conda activate lerobot   # or: pip install -r script/lerobot/requirements.txt in your own env

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/grab_roller_aloha-agilex_clean_50 \
    --output-path /tmp/lerobot_out/grab_roller
```

Useful flags:
- `--max-episodes N` — convert only the first N episodes (for a quick smoke test).
- `--fps` — defaults to auto-detecting `save_freq` from `task_config/demo_<config>.yml`,
  falling back to 15.
- `--image-layout {auto,hwc,chw}` — defaults to `auto`, which picks HWC for the
  new `lerobot.datasets.*` import path (this repo's `lerobot` env) or CHW for the
  older `lerobot.common.datasets.*` path (e.g. `policy/pi0`'s pinned lerobot commit).
- `--instruction-set {seen,unseen}` — which half of each episode's
  `instructions/episode{N}.json` to sample the language instruction from
  (ignored if `--instruction` is given).
- `--instruction "..."` — use this single fixed instruction string for every
  episode/frame in the dataset, instead of randomly sampling a different one
  per episode.
- `--overwrite` — replace an existing `--output-path` (off by default).
- `--push-to-hub` — push the converted dataset to the HF Hub afterwards.

### Combining multiple task folders

Each run of `convert_robotwin_to_lerobot.py` converts one task folder into one
LeRobot dataset. To merge several already-converted datasets into one combined
dataset:

```bash
python script/lerobot/aggregate_lerobot_datasets.py \
    --input-dirs /tmp/lerobot_out/grab_roller /tmp/lerobot_out/move_stapler_pub \
    --output-dir /tmp/lerobot_out/aggregated
```

All input datasets must share the same fps / robot_type / features schema
(guaranteed if they were all produced by `convert_robotwin_to_lerobot.py` with
the same `--cameras`/`--image-layout`). Note that the aggregated dataset's
`total_tasks` count deduplicates identical instruction strings across inputs,
so it isn't simply the sum of each input's task count.

## Compatibility

`lerobot_compat.py` transparently supports both:
- the newer `lerobot.datasets.*` import path (HWC images) — this repo's
  `lerobot` conda env (`lerobot==0.3.4`).
- the older `lerobot.common.datasets.*` path (CHW images) — used by
  `policy/GO1/scripts/convert_aloha_data_to_lerobot_robotwin.py` and the
  `lerobot` commit pinned by `policy/pi0/pyproject.toml`.
