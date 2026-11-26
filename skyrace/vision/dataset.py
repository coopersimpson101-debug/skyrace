"""Synthetic training / evaluation renders with exact corner labels.

Each sample: random gate layout (6 gates scattered over a 60 m field, random
heights), random scene texture (``scene_seeds``), a camera pose that is
usually "approaching a gate" (distance 1.5-16 m, heading error ~14 deg rms,
racing pitch -70..+10 deg, roll up to +-85 deg in banked turns) and sometimes fully random. Gate
colour, exposure, sensor noise and motion blur are randomised.

Labels: for every gate, its 4 inner corners TL, TR, BR, BL in the gate's own
frame (left/right as seen from the approach side, up = away from the legs;
mirrored when seen from behind), visibility and depth. Under a banked turn the
labels rotate with the image, so the network has to use the gate's legs and the
horizon, not just "top-left in the image".
"""
from __future__ import annotations

import numpy as np

from ..so3 import euler_to_rot
from ..sim.camera import CameraModel, Renderer, Scene
from ..sim.track import Gate

MAX_GATES = 8


def random_layout(rng, n=6):
    gates = []
    for _ in range(n):
        for _try in range(50):
            c = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), -rng.uniform(1.3, 4.5)])
            if all(np.linalg.norm(c[:2] - g.center[:2]) > 5.0 for g in gates):
                break
        gates.append(Gate(c, rng.uniform(-np.pi, np.pi)))
    return gates


def random_view(rng, gates):
    if rng.random() < 0.85:
        g = gates[rng.integers(len(gates))]
        d = rng.uniform(1.5, 16.0)
        side = 1.0 if rng.random() < 0.9 else -1.0     # mostly approach from the front
        lat = rng.uniform(-0.45, 0.45) * d
        vert = rng.uniform(-0.25, 0.25) * d
        p = g.center + g.R @ np.array([-side * d, lat, vert])
        p[2] = min(p[2], -0.3)
        to = g.center - p
        yaw = np.arctan2(to[1], to[0]) + rng.normal(0, np.radians(14))
        # FPV racing attitudes: nose down up to 70 deg, banked turns up to ~85 deg of roll
        pitch = rng.uniform(np.radians(-70), np.radians(10))
        if rng.random() < 0.5:
            roll = rng.normal(0, np.radians(20))
        else:
            roll = rng.uniform(np.radians(-85), np.radians(85))
    else:
        p = np.array([rng.uniform(-35, 35), rng.uniform(-35, 35), -rng.uniform(0.5, 6)])
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi), rng.uniform(-0.8, 0.3), rng.normal(0, 0.3)
    return euler_to_rot(roll, pitch, yaw), p


def augment(rng, img):
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.3:
        k = int(rng.integers(2, 4))
        ax = int(rng.integers(0, 2))
        img = sum(np.roll(img, s, axis=ax) for s in range(k)) / k
    img = img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape)
    return np.clip(img, 0, 1)


def render_sample(seed: int, scene_seeds=(0, 1, 2, 3, 4, 5), cam: CameraModel | None = None,
                  gates=None, scenes_cache={}):
    rng = np.random.default_rng(seed)
    ss = int(scene_seeds[rng.integers(len(scene_seeds))])
    if ss not in scenes_cache:
        scenes_cache[ss] = Scene(seed=ss)
    cam = cam or CameraModel()
    r = Renderer(cam, scenes_cache[ss])
    gates = gates if gates is not None else random_layout(rng, int(rng.integers(3, 7)))
    R, p = random_view(rng, gates)
    hue = np.array([1.0, 0.45, 0.08]) + rng.normal(0, [0.04, 0.08, 0.04])
    tint = [np.clip(hue + rng.normal(0, 0.03, 3), 0, 1) for _ in gates]
    img = augment(rng, r.render(R, p, gates, gate_tint=tint))
    lab = np.zeros((MAX_GATES, 4, 4), np.float32)  # u, v, visible, depth
    for i, (uv, vis, mir, z) in enumerate(r.corner_labels(R, p, gates)[:MAX_GATES]):
        lab[i, :, :2] = uv
        lab[i, :, 2] = vis
        lab[i, :, 3] = z
    return (img * 255).astype(np.uint8), lab, R, p


def _worker(args):
    seed, scene_seeds = args
    img, lab, _, _ = render_sample(seed, scene_seeds)
    return img, lab


def make_dataset(n: int, seed0: int, scene_seeds, workers: int = 6):
    jobs = [(seed0 + i, tuple(scene_seeds)) for i in range(n)]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_worker, jobs, chunksize=16)
    else:
        res = [_worker(j) for j in jobs]
    imgs = np.stack([r[0] for r in res])
    labs = np.stack([r[1] for r in res])
    return imgs, labs
