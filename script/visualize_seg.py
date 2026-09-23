"""Render segmentation / depth from collected episodes into an mp4 for a quick visual check.

Each camera becomes one row:  RGB | RGB + actor seg (with names) | robot vs object mask | mesh seg | depth
(panels whose data is missing are skipped; pick panels with --panels, e.g. `--panels rgb depth`).
Depth uses the turbo colormap (red = near, blue = far) with a meter scale bar; black = no depth.

Usage:
    python script/visualize_seg.py data/aloha-agilex/adjust_bottle                 # episode 0
    python script/visualize_seg.py data/aloha-agilex/adjust_bottle -e 0 3 7
    python script/visualize_seg.py data/aloha-agilex/adjust_bottle -e all --cameras head_camera
    python script/visualize_seg.py data/aloha-agilex/adjust_bottle/data/episode0.hdf5
    python script/visualize_seg.py data/aloha-agilex/adjust_bottle --panels rgb depth   # depth only
Output: <task_dir>/seg_vis/episode<i>.mp4 (or --out_dir)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

import cv2
import h5py
import numpy as np

ROBOT_COLOR = np.array([40, 120, 255], dtype=np.uint8)  # RGB
OBJECT_COLOR = np.array([255, 150, 30], dtype=np.uint8)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANELS = ["rgb", "actor", "robot", "mesh", "depth"]


def id_colors(max_id):
    """Stable, well separated color per id; id 0 (background) is black."""
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 256, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors


def decode_rgb(frames):
    # rgb is stored as padded jpeg bytes; the encoder got RGB arrays, so decoding gives RGB back
    return np.stack([cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in frames])


def colorize_seg(seg, colors):
    if seg.ndim == 4:  # legacy palette mode (segmentation_raw_id: false) - already colored
        return seg.astype(np.uint8)
    return colors[seg]


def colorize_depth(depth):
    depth = depth.astype(np.float32)
    valid = depth > 0
    if not valid.any():
        return np.zeros(depth.shape + (3, ), np.uint8)
    lo, hi = np.percentile(depth[valid], [1, 99])  # fixed range over the whole episode -> no flicker
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.stack([cv2.applyColorMap((255 - norm[i] * 255).astype(np.uint8), cv2.COLORMAP_TURBO) for i in range(len(norm))])
    out = out[..., ::-1].copy()  # BGR -> RGB
    out[~valid] = 0
    add_depth_colorbar(out, lo, hi)
    return out


def add_depth_colorbar(frames, lo_mm, hi_mm):
    """Vertical turbo bar on the right edge: red = near (lo), blue = far (hi), labeled in meters."""
    h = frames.shape[1]
    bar_h, bar_w = int(h * 0.6), 10
    y0, x0 = (h - bar_h) // 2, frames.shape[2] - bar_w - 48
    ramp = np.linspace(255, 0, bar_h).astype(np.uint8)[:, None].repeat(bar_w, 1)
    bar = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[..., ::-1]
    for f in frames:
        f[y0:y0 + bar_h, x0:x0 + bar_w] = bar
        for y, v in [(y0 + 4, lo_mm), (y0 + bar_h, hi_mm)]:
            txt = f"{v / 1000:.2f}m"
            cv2.putText(f, txt, (x0 + bar_w + 3, y), FONT, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(f, txt, (x0 + bar_w + 3, y), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def label(img, text):
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 9 * len(text) + 8), 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_names(img, seg, names, min_area):
    """Write the actor name at the center of each visible region."""
    ids, counts = np.unique(seg, return_counts=True)
    for i, c in zip(ids, counts):
        if i == 0 or c < min_area or str(i) not in names:
            continue
        ys, xs = np.nonzero(seg == i)
        x, y = int(np.median(xs)), int(np.median(ys))
        cv2.putText(img, names[str(i)], (x, y), FONT, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, names[str(i)], (x, y), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def load_id_map(task_dir, ep):
    path = os.path.join(task_dir, "seg_id_map", f"episode{ep}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def render_camera(cam, cam_name, id_map, alpha, label_every, min_area, want):
    """Return list of panels, each (T, H, W, 3) uint8. `want` = set of panel names to draw."""
    rgb = decode_rgb(cam["rgb"][:])
    panels = []
    if "rgb" in want:
        panels.append(np.stack([label(f.copy(), f"{cam_name} rgb") for f in rgb]))

    if "actor_segmentation" in cam and want & {"actor", "robot"}:
        seg = cam["actor_segmentation"][:]
        raw = seg.ndim == 3
        colors = id_colors(int(seg.max()) if raw else 0)
        seg_rgb = colorize_seg(seg, colors)
        mask = (seg > 0) if raw else seg.any(-1)
        overlay = rgb.copy()
        overlay[mask] = (alpha * seg_rgb[mask] + (1 - alpha) * rgb[mask]).astype(np.uint8)
        names = {k: v["name"] if isinstance(v, dict) else v
                 for k, v in (id_map or {}).get("actor_segmentation", {}).items()}
        for t in range(len(overlay)):
            if raw and names and t % label_every == 0:
                draw_names(overlay[t], seg[t], names, min_area)
            label(overlay[t], "actor seg")
        if "actor" in want:
            panels.append(overlay)

        robot_ids = sum((id_map or {}).get("robot_actor_ids", {}).values(), [])
        if raw and robot_ids and "robot" in want:
            robot = np.isin(seg, robot_ids)
            ro = np.zeros_like(rgb)
            ro[mask & ~robot] = OBJECT_COLOR
            ro[robot] = ROBOT_COLOR
            ro = (0.3 * rgb + 0.7 * ro).astype(np.uint8)
            panels.append(np.stack([label(f, "robot (blue) / object (orange)") for f in ro]))

    if "mesh_segmentation" in cam and "mesh" in want:
        seg = cam["mesh_segmentation"][:]
        colors = id_colors(int(seg.max()) if seg.ndim == 3 else 0)
        panels.append(np.stack([label(f.copy(), "mesh seg") for f in colorize_seg(seg, colors)]))

    if "depth" in cam and "depth" in want:
        panels.append(np.stack([label(f, f"{cam_name} depth") for f in colorize_depth(cam["depth"][:])]))
    elif "depth" in want and "depth" not in cam:
        print(f"  {cam_name}: no depth in file (collected with depth: false?)")
    return panels


def write_video(frames, path, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames.shape[1:3]
    frames = frames[:, :h - h % 2, :w - w % 2]  # yuv420p needs even size
    if shutil.which("ffmpeg"):
        proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{frames.shape[2]}x{frames.shape[1]}", "-framerate", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "20", path
        ], stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(frames).tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
        for f in frames:
            writer.write(f[..., ::-1])
        writer.release()


def visualize_episode(hdf5_path, task_dir, ep, a):
    id_map = load_id_map(task_dir, ep)
    with h5py.File(hdf5_path, "r") as f:
        obs = f["observation"]
        cams = a.cameras or [c for c in ["head_camera", "left_camera", "right_camera"] if c in obs] + \
            [c for c in obs.keys() if c not in ("head_camera", "left_camera", "right_camera")]
        rows = []
        for cam in cams:
            if cam not in obs or "rgb" not in obs[cam]:
                print(f"  skip {cam}: no rgb")
                continue
            panels = render_camera(obs[cam], cam, id_map, a.alpha, a.label_every, a.min_area, set(a.panels))
            if not panels:
                continue
            h = max(p.shape[1] for p in panels)
            panels = [p if p.shape[1] == h else np.stack([cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))
                                                          for x in p]) for p in panels]
            rows.append(np.concatenate(panels, axis=2))
    if not rows:
        print(f"  nothing to draw in {hdf5_path}")
        return
    width = max(r.shape[2] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, 0), (0, width - r.shape[2]), (0, 0))) for r in rows]
    frames = np.concatenate(rows, axis=1)
    if a.scale != 1:
        frames = np.stack([cv2.resize(x, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA) for x in frames])

    out = os.path.join(a.out_dir or os.path.join(task_dir, "seg_vis"), f"episode{ep}.mp4")
    write_video(frames, out, a.fps)
    if id_map is None:
        print("  (no seg_id_map found: names / robot mask not shown)")
    print(f"  -> {out}  ({len(frames)} frames, {frames.shape[2]}x{frames.shape[1]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Task folder (e.g. data/aloha-agilex/adjust_bottle) or one episode .hdf5")
    p.add_argument("-e", "--episodes", nargs="+", default=["0"], help="Episode indices, or 'all'")
    p.add_argument("--cameras", nargs="+", default=None, help="Default: every camera in the file")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--panels", nargs="+", default=PANELS, choices=PANELS,
                   help="Which panels to draw, e.g. `--panels rgb depth` for a depth-only check")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.55, help="Segmentation overlay opacity")
    p.add_argument("--scale", type=float, default=1.0, help="Resize the final video")
    p.add_argument("--label_every", type=int, default=1, help="Draw actor names every N frames (slow on big videos)")
    p.add_argument("--min_area", type=int, default=150, help="Min pixel area of a region to get a name label")
    a = p.parse_args()

    if a.path.endswith(".hdf5"):
        ep = int(re.search(r"episode(\d+)\.hdf5$", a.path).group(1))
        jobs = [(a.path, os.path.dirname(os.path.dirname(os.path.abspath(a.path))), ep)]
    else:
        files = glob.glob(os.path.join(a.path, "data", "episode*.hdf5"))
        available = sorted(int(re.search(r"episode(\d+)\.hdf5$", f).group(1)) for f in files)
        eps = available if a.episodes == ["all"] else [int(e) for e in a.episodes]
        jobs = [(os.path.join(a.path, "data", f"episode{e}.hdf5"), a.path, e) for e in eps]

    for hdf5_path, task_dir, ep in jobs:
        if not os.path.exists(hdf5_path):
            print(f"missing {hdf5_path}")
            continue
        print(f"episode {ep}: {hdf5_path}")
        visualize_episode(hdf5_path, task_dir, ep, a)


if __name__ == "__main__":
    main()
