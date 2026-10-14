"""Build the `human-panda` embodiment: a Franka Panda whose *visual* meshes are replaced by a
skin-coloured human arm + hand. Collision meshes, kinematics and curobo configs are untouched, so the
planner and grasp logic behave exactly like franka-panda.

Also swaps head_camera for an egocentric (eye-level) view.

    python script/make_human_panda.py [--skin 0.87 0.69 0.56]
"""
import argparse
import os
import re
import shutil

import numpy as np
import trimesh
import yaml

SRC = "assets/embodiments/franka-panda"
DST = "assets/embodiments/human-panda"
ENTRY = "task_config/_embodiment_config.yml"


def sphere(c, r):
    return trimesh.creation.uv_sphere(radius=r, count=[16, 16]).apply_translation(c)


def limb(p0, r0, p1, r1):
    """Tapered capsule: convex hull of two spheres (r0 at p0, r1 at p1)."""
    pts = np.concatenate([sphere(p0, r0).vertices, sphere(p1, r1).vertices])
    return trimesh.convex.convex_hull(pts)


def rbox(center, size, round_r):
    """Box with rounded edges/corners (hull of 8 corner spheres)."""
    half = np.array(size) / 2 - round_r
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(sphere(np.array(center) + half * [sx, sy, sz], round_r).vertices)
    return trimesh.convex.convex_hull(np.concatenate(pts))


def chain(joints, radii):
    """Smooth finger/thumb: tapered capsules between consecutive joints."""
    return [limb(joints[i], radii[i], joints[i + 1], radii[i + 1]) for i in range(len(joints) - 1)]


# Gripper frame conventions (from the Panda collision meshes): the pad contact plane of both fingers is
# local y = 0; the left finger body lies at +y, the right finger at -y, z runs towards the fingertips.
# So the thumb (left finger) sits at +y and the four fingers (right finger) at -y, tips curl towards y=0.
FINGERS = [  # (x, length scale): index, middle, ring, little (x grows towards the ulnar side)
    (-0.0095, 0.95), (0.0095, 1.05), (0.0285, 0.98), (0.0465, 0.80),
]
PALM_X = 0.0185


def finger_meshes():
    out = []
    for x, k in FINGERS:
        z0 = -0.014
        zs = np.cumsum([0, 0.034 * k, 0.022 * k, 0.018 * k]) + z0
        ys = [-0.0185, -0.0165, -0.0125, -0.0085]  # base stands off the contact plane, tip touches it
        joints = [np.array([x, y, z]) for y, z in zip(ys, zs)]
        out += chain(joints, [0.0092, 0.0085, 0.0078, 0.0070])
        out.append(sphere(joints[-1], 0.0070))
    return out


def thumb_meshes():
    joints = [np.array(p) for p in ([-0.004, 0.017, -0.040], [-0.004, 0.0150, -0.012],
                                    [-0.004, 0.0130, 0.016], [-0.004, 0.0105, 0.040])]
    out = chain(joints, [0.0125, 0.0108, 0.0095, 0.0082])
    out.append(sphere(joints[-1], 0.0082))
    return out


# Panda link frames -> human arm parts (frame offsets taken from panda.urdf joint origins)
def link_parts():
    return {
        # link0/link1 (shoulder column) are intentionally left without visuals
        "link2": [sphere([0, 0, 0], 0.058), limb([0, 0, 0], 0.052, [0, -0.316, 0], 0.042)],     # shoulder -> upper arm
        "link3": [sphere([0, 0, 0], 0.044), limb([0, 0, 0], 0.044, [0.0825, 0, 0], 0.042)],     # elbow
        "link4": [sphere([0, 0, 0], 0.042), limb([0, 0, 0], 0.042, [-0.0825, 0.384, 0], 0.029)],  # forearm
        "link5": [sphere([0, 0, 0], 0.029)],                                                    # wrist
        "link6": [sphere([0, 0, 0], 0.029), limb([0, 0, 0], 0.029, [0.088, 0, 0], 0.029)],
        "link7": [limb([0, 0, 0], 0.029, [PALM_X * 0.6, -0.022, 0.092], 0.034)],                # wrist -> palm
        "hand": [rbox([PALM_X, -0.027, 0.020], [0.092, 0.040, 0.078], 0.012),                   # palm
                 limb([-0.011, -0.003, 0.002], 0.021, [-0.011, -0.003, 0.040], 0.013)],         # thenar pad
        "leftfinger": thumb_meshes(),
        "rightfinger": finger_meshes(),
    }


def save_glb(parts, rgba, path):
    mesh = trimesh.util.concatenate(parts)
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8))
    mesh.export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skin", nargs=3, type=float, default=[0.87, 0.69, 0.56])
    a = ap.parse_args()
    rgba = [*a.skin, 1.0]

    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    vis = os.path.join(DST, "franka_description/meshes/visual")
    for f in os.listdir(vis):
        os.remove(os.path.join(vis, f))

    parts = link_parts()
    for name, ps in parts.items():
        save_glb(ps, rgba, os.path.join(vis, f"human_{name}.glb"))

    # --- URDF: point visuals to the human meshes, drop link0/1 and camera visuals
    urdf_path = os.path.join(DST, "panda.urdf")
    urdf = open(urdf_path).read()

    def repl_link(match):
        link, body = match.group(1), match.group(2)
        key = {"panda_hand": "hand", "panda_leftfinger": "leftfinger", "panda_rightfinger": "rightfinger"}.get(
            link, link.replace("panda_", ""))
        if key in parts:
            body = re.sub(r"<mesh filename=\"franka_description/meshes/visual/[^\"]+\"\s*/>",
                          f'<mesh filename="franka_description/meshes/visual/human_{key}.glb"/>', body)
            # the meshes are already modelled in the link frame (right finger is symmetric)
            body = re.sub(r"(<visual>)\s*<origin[^>]*/>", r"\1", body, count=1)
        else:  # no human visual for this link
            body = re.sub(r"<visual>.*?</visual>", "", body, flags=re.S)
        return f'<link name="{link}">{body}</link>'

    urdf = re.sub(r'<link name="([^"]+)">(.*?)</link>', repl_link, urdf, flags=re.S)
    open(urdf_path, "w").write(urdf)

    # --- config: egocentric head camera
    cfg_path = os.path.join(DST, "config.yml")
    cfg = open(cfg_path).read()
    cfg = re.sub(r"static_camera_list:.*?(?=\n# - name|\Z)", """static_camera_list:
# same as franka-panda's head camera, raised to z=1.45
- name: head_camera
  position:
  - -0.032
  - -0.45
  - 1.45
  forward:
  - 0
  - 0.6
  - -0.8
  left:
  - -1
  - 0
  - 0
""", cfg, flags=re.S)
    open(cfg_path, "w").write(cfg)

    # --- register
    entry = open(ENTRY).read()
    if "human-panda:" not in entry:
        entry = entry.rstrip("\n") + '\n\nhuman-panda:\n  file_path: "./assets/embodiments/human-panda"\n'
        open(ENTRY, "w").write(entry)
    print(f"wrote {DST}; registered human-panda in {ENTRY}")


if __name__ == "__main__":
    main()
