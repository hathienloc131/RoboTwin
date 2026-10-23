"""Per-frame scene-graph ground truth for the shared-scene-graph (SSG) co-training work.

Enabled with ``data_type.scene_graph: true`` in a task config; get_obs then adds
``pkl_dic["scene_graph"]`` with fixed-size arrays over MAX_OBJECTS slots (unused slots NaN /
-1), which pkl2hdf5 stores as ``scene_graph/<key>`` datasets of shape (T, ...):

- ``obj_pose`` (N, 7): world xyz + quaternion wxyz
- ``obj_aabb_local`` (N, 6): actor-frame AABB min xyz, max xyz (from collision shapes)
- ``obj_seg_id`` (N,): actor per_scene_id (matches head-camera actor segmentation)
- ``head_mask_px`` (N,): visible pixels of the object in the head camera
- ``head_mask_bbox`` (N, 4): visible-mask box u0, v0, u1, v1 (pixels)
- ``head_mask_centroid`` (N, 2): visible-mask centroid u, v (pixels)
- ``gripper_contact`` (N, 2, 2): object touches [left, right] arm x [finger link7, link8] (0 / 1)

Slot order and names come from SSG_OBJECTS; the converter
(lerobot-convert scene_graph core) holds the matching vocabulary.
"""

import numpy as np

MAX_OBJECTS = 3

# task -> [(slot_name, env attribute)]; slot order is the graph's node order
SSG_OBJECTS = {
    "adjust_bottle": [("bottle", "bottle")],
    "blocks_ranking_rgb": [("red_block", "block1"), ("green_block", "block2"), ("blue_block", "block3")],
    "click_bell": [("bell", "bell")],
    "move_stapler_pad": [("stapler", "stapler"), ("pad", "pad")],
    "press_stapler": [("stapler", "stapler")],
    "stack_blocks_two": [("red_block", "block1"), ("green_block", "block2")],
    "stack_blocks_three": [("red_block", "block1"), ("green_block", "block2"), ("blue_block", "block3")],
    "stack_bowls_two": [("bowl1", "bowl1"), ("bowl2", "bowl2")],
    "stack_bowls_three": [("bowl1", "bowl1"), ("bowl2", "bowl2"), ("bowl3", "bowl3")],
}

LEFT_FINGER_LINKS = ("fl_link7", "fl_link8")
RIGHT_FINGER_LINKS = ("fr_link7", "fr_link8")
CONTACT_SEPARATION = 0.002  # m


def _entity(actor):
    return actor.actor if hasattr(actor, "actor") else actor


def _local_aabb(entity) -> np.ndarray:
    """Actor-frame AABB from the collision shapes (box half sizes or scaled convex hulls)."""
    pts = []
    for comp in entity.get_components():
        if not hasattr(comp, "get_collision_shapes"):
            continue
        for shape in comp.get_collision_shapes():
            if hasattr(shape, "half_size"):
                h = np.asarray(shape.half_size, dtype=np.float64)
                local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            elif hasattr(shape, "vertices"):
                local = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale, dtype=np.float64)
            else:
                continue
            T = shape.local_pose.to_transformation_matrix()
            pts.append(local @ T[:3, :3].T + T[:3, 3])
    if not pts:
        return np.full(6, np.nan)
    pts = np.concatenate(pts)
    return np.concatenate([pts.min(0), pts.max(0)])


def scene_graph_obs(task, head_camera) -> dict:
    """Ground-truth object state for one frame. ``head_camera`` must have a fresh picture."""
    slots = SSG_OBJECTS.get(task.task_name)
    if slots is None:
        raise KeyError(f"scene_graph: no SSG_OBJECTS entry for task {task.task_name!r}")

    n = MAX_OBJECTS
    out = {
        "obj_pose": np.full((n, 7), np.nan),
        "obj_aabb_local": np.full((n, 6), np.nan),
        "obj_seg_id": np.full(n, -1, dtype=np.int64),
        "head_mask_px": np.zeros(n, dtype=np.int64),
        "head_mask_bbox": np.full((n, 4), np.nan),
        "head_mask_centroid": np.full((n, 2), np.nan),
        "gripper_contact": np.zeros((n, 2, 2), dtype=np.int64),
    }

    # the AABB is constant within an episode; cache it on the task object
    cache = getattr(task, "_ssg_aabb_cache", None)
    if cache is None:
        cache = task._ssg_aabb_cache = {}

    seg_ids = head_camera.get_picture("Segmentation")[..., 1]
    id_to_slot = {}
    for i, (_, attr) in enumerate(slots):
        ent = _entity(getattr(task, attr))
        pose = ent.get_pose()
        out["obj_pose"][i] = np.concatenate([pose.p, pose.q])
        if attr not in cache or cache[attr][0] is not ent:  # new episode, new actor
            cache[attr] = (ent, _local_aabb(ent))
        out["obj_aabb_local"][i] = cache[attr][1]
        sid = int(ent.per_scene_id)
        out["obj_seg_id"][i] = sid
        id_to_slot[sid] = i

        vv, uu = np.nonzero(seg_ids == sid)
        out["head_mask_px"][i] = len(uu)
        if len(uu):
            out["head_mask_bbox"][i] = [uu.min(), vv.min(), uu.max(), vv.max()]
            out["head_mask_centroid"][i] = [uu.mean(), vv.mean()]

    for contact in task.scene.get_contacts():
        if not any(p.separation < CONTACT_SEPARATION for p in contact.points):
            continue
        a, b = (body.entity for body in contact.bodies)
        for obj, other in ((a, b), (b, a)):
            slot = id_to_slot.get(int(obj.per_scene_id))
            if slot is None:
                continue
            name = other.get_name()
            for arm, links in enumerate((LEFT_FINGER_LINKS, RIGHT_FINGER_LINKS)):
                if name in links:
                    out["gripper_contact"][slot, arm, links.index(name)] = 1
    return out
