"""Gate measurement front-end: corners -> association with the predicted gate -> PnP."""
from __future__ import annotations

import os

import numpy as np

from . import classical
from .cnn import GateNetNumpy, decode_heatmaps
from .pnp import camera_pose_to_body, solve_pnp_planar

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "gatenet.npz")


class CornerDetector:
    """``kind`` = "cnn" (GateNet, numpy inference) or "classical" (colour baseline)."""

    def __init__(self, kind: str = "cnn", weights: str = DEFAULT_WEIGHTS, thresh: float = 0.3):
        self.kind = kind
        self.typed = kind == "cnn"   # classical corners are typed by image position only
        self.thresh = thresh
        self.net = GateNetNumpy.load(weights) if kind == "cnn" else None

    def __call__(self, img: np.ndarray):
        if self.kind == "cnn":
            return decode_heatmaps(self.net(img), self.thresh)
        return classical.detect_corners(img)


def associate(cands, pred_uv, radius: float, typed: bool = True):
    """Pick, per corner type, the strongest-then-nearest candidate within ``radius``.

    ``typed=False`` pools all candidates and matches each predicted corner to the
    nearest unused one (for detectors that cannot tell corner types apart)."""
    uv = np.full((4, 2), np.nan)
    ok = np.zeros(4, bool)
    if not typed:
        nonempty = [c for c in cands if len(c)]
        pool = np.concatenate(nonempty, 0) if nonempty else np.zeros((0, 3))
        used = np.zeros(len(pool), bool)
        for c in range(4):
            if len(pool) == 0 or not np.all(np.isfinite(pred_uv[c])):
                continue
            d = np.linalg.norm(pool[:, :2] - pred_uv[c], axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= radius:
                uv[c] = pool[j, :2]
                ok[c] = True
                used[j] = True
        return uv, ok
    for c in range(4):
        cc = cands[c]
        if len(cc) == 0 or not np.all(np.isfinite(pred_uv[c])):
            continue
        d = np.linalg.norm(cc[:, :2] - pred_uv[c], axis=1)
        j = int(np.argmin(d - 2.0 * cc[:, 2]))   # mild preference for confident peaks
        if d[j] <= radius:
            uv[c] = cc[j, :2]
            ok[c] = True
    return uv, ok


def predicted_corners(cam, R_wb, p_wb, gate):
    mirrored = bool(np.dot(p_wb - gate.center, gate.normal) > 0)
    uv, z = cam.project(R_wb, p_wb, gate.inner_corners_world(mirrored))
    uv[z < 0.3] = np.nan
    return uv, mirrored, z


def measure_gate(cands, cam, gate, R_prior, p_prior, radius=14.0, sigma_px=1.5, max_rms=3.0, typed=True):
    """Returns a measurement dict (world pose of the body + covariance) or None."""
    pred, mirrored, z = predicted_corners(cam, R_prior, p_prior, gate)
    if not np.all(np.isfinite(pred)):
        return None
    uv, ok = associate(cands, pred, radius, typed)
    if ok.sum() < 4:
        return None
    X = gate.inner_corners_local(mirrored)
    sol = solve_pnp_planar(X, uv, cam.K, sigma_px=sigma_px)
    if sol is None or not sol["ok"] or sol["rms"] > max_rms:
        return None
    p, R, cov = camera_pose_to_body(sol, gate, cam.R_bc)
    return dict(p=p, R=R, cov=cov, rms=sol["rms"], uv=uv, pred=pred, dist=float(np.linalg.norm(sol["t"])),
                X_w=gate.inner_corners_world(mirrored))
