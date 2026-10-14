import numpy as np
from scipy.ndimage import gaussian_filter1d

SIM_HZ = 250  # matches scene timestep / planner interpolation_dt


def _blend_rest_phases(pos, min_speed_frac):
    """Re-time a trajectory so its speed never drops below `min_speed_frac` * (peak speed).

    The planner brings every segment to rest at both ends, so consecutive segments (approach -> grasp
    -> lift) look like stop-and-go. Compressing the slow head/tail in *time* (the path is untouched, so
    both end poses are still reached exactly) lets neighbouring segments run into each other at speed.
    """
    n = pos.shape[0]
    ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # joint-space step length
    if n < 8 or ds.sum() < 1e-3:
        return pos
    v = ds * SIM_HZ
    vmin = min_speed_frac * np.percentile(v, 95)
    if vmin <= 0:
        return pos
    # steps with ds == 0 take no time; everything else advances by ds / max(v, vmin)
    t =np.concatenate([[0.0], np.cumsum(np.where(ds > 0, ds / np.maximum(v, vmin), 0.0))])
    grid = np.arange(0.0, t[-1] + 0.5 / SIM_HZ, 1.0 / SIM_HZ)
    grid[-1] = t[-1]
    return np.stack([np.interp(grid, t, pos[:, j]) for j in range(pos.shape[1])], axis=1)


def humanize_arm_traj(arm, cfg):
    """Make a planned joint trajectory look less robotic.

    arm: {"position": (N, J), "velocity": (N, J)}; returns a new dict, input is left untouched.
    cfg keys (all optional):
        min_speed_frac: speed floor as a fraction of peak speed, applied to the head/tail of the
                        segment so segments blend instead of stopping (0 disables)
        smooth_sigma:   Gaussian sigma in sim steps (1 step = 4 ms), 0 disables smoothing
        noise_std:      peak joint noise in rad, 0 disables noise
        noise_sigma:    noise correlation length in sim steps
    Both end poses are preserved exactly, so grasp/place targets are still reached. The segment may
    end with non-zero velocity; the caller must zero it before the arm idles (see Base_Task).
    """
    pos = np.asarray(arm["position"], dtype=np.float64)
    if pos.shape[0] < 3:
        return arm
    first, last = pos[0].copy(), pos[-1].copy()

    out = pos
    frac = cfg.get("min_speed_frac", 0)
    if frac > 0:
        out = _blend_rest_phases(out, frac)
    n = out.shape[0]

    sigma = cfg.get("smooth_sigma", 0)
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=min(sigma, n / 4), axis=0, mode="nearest")

    std = cfg.get("noise_std", 0)
    if std > 0:
        corr = max(cfg.get("noise_sigma", 40), 1)
        noise = np.random.randn(n, out.shape[1])
        noise = gaussian_filter1d(noise, sigma=min(corr, n / 4), axis=0, mode="nearest")
        noise *= std / (noise.std(axis=0, keepdims=True) + 1e-9)
        out = out + noise * np.sin(np.pi * np.linspace(0, 1, n))[:, None]  # zero at both ends

    # pin endpoints (smoothing with mode="nearest" shifts them slightly)
    t = np.linspace(0, 1, n)[:, None]
    out = out - ((out[0] - first) * (1 - t) + (out[-1] - last) * t)

    res = dict(arm)
    res["position"] = out.astype(arm["position"].dtype)
    res["velocity"] = (np.gradient(out, axis=0) * SIM_HZ).astype(arm["velocity"].dtype)
    return res


def humanize_gripper(grip, cfg):
    """Ease a planned gripper command (linear ramp) like fingers closing: slow-fast-slow, optionally slower.

    grip: {"num_step", "per_step", "result"} as returned by Planner.plan_grippers (+ hold padding).
    cfg keys: gripper_stretch (>=1 lengthens the ramp, 0/absent disables easing).
    """
    stretch = cfg.get("gripper_stretch", 0)
    if not stretch or grip is None:
        return grip
    vals = np.asarray(grip["result"], dtype=np.float64)
    a, b = vals[0], vals[-1]
    if vals.shape[0] < 4 or abs(b - a) < 1e-4:
        return grip
    moving = int(np.argmax(np.abs(vals - b) < 1e-9)) + 1  # ramp length; the rest is hold padding
    ramp = max(int(moving * max(stretch, 1.0)), 4)
    s = np.linspace(0, 1, ramp)
    ease = s**3 * (s * (6 * s - 15) + 10)  # smootherstep
    new = np.concatenate([a + (b - a) * ease, np.full(vals.shape[0] - moving, b)])
    res = dict(grip)
    res["result"] = new
    res["num_step"] = new.shape[0]
    res["per_step"] = 2.0 * (b - a) / ramp  # Robot.set_gripper rate-limit; smootherstep peaks at 1.875x linear
    return res
