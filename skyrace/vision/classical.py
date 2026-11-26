"""Classical gate-corner baseline: colour threshold + hole extreme points.

1. Orange mask in a chromaticity test (r > g > b, saturated, bright enough).
2. Connected components (4-neighbour) by vectorised min-label propagation.
3. For every component large enough: fill its bounding box, find the
   background region(s) enclosed by the component (connected components of
   the inverted mask that do not touch the bounding-box border) = the gate
   opening.
4. The opening's corners are its extreme points along +-(x + y) and
   +-(x - y): TL = argmin(x + y), BR = argmax(x + y), TR = argmax(x - y),
   BL = argmin(x - y). Sub-pixel: average of the 3 most extreme pixels.

Known weaknesses (measured in results/detector_eval.json): fails when the
opening is not closed in the image (gate partly out of view, or overlapping
another gate), when roll is near +-45 deg (corner ordering ambiguous), and on
orange clutter; the CNN does not need any of those assumptions.
"""
from __future__ import annotations

import numpy as np


def orange_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(-1)
    mn = img.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    return (r > g * 1.25) & (g > b * 1.1) & (sat > 0.55) & (r > 0.35)


def label_components(mask: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """4-connected labels (0 = background) by iterated min-pooling of pixel ids."""
    H, W = mask.shape
    big = H * W + 1
    lab = np.where(mask, np.arange(1, H * W + 1).reshape(H, W), big)
    for _ in range(max_iter):
        p = np.pad(lab, 1, constant_values=big)
        nb = np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        nb = np.where(mask, nb, big)
        if np.array_equal(nb, lab):
            break
        lab = nb
    lab[~mask] = 0
    return lab


def _extreme(xs, ys, key, largest):
    k = 3 if len(xs) >= 3 else len(xs)
    idx = np.argsort(key)[-k:] if largest else np.argsort(key)[:k]
    return float(xs[idx].mean()), float(ys[idx].mean())


def detect_gates(img: np.ndarray, min_area: int = 12):
    """Returns a list of gates, each a 4x2 array of (u, v) for TL, TR, BR, BL."""
    mask = orange_mask(img)
    lab = label_components(mask)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    H, W = mask.shape
    gates = []
    for i, c in zip(ids, counts):
        if c < min_area:
            continue
        ys, xs = np.nonzero(lab == i)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) < 4 or (x1 - x0) < 4:
            continue
        comp = lab[y0:y1, x0:x1] == i
        bg = ~comp
        hl = label_components(bg)
        border = np.unique(np.r_[hl[0], hl[-1], hl[:, 0], hl[:, -1]])
        hids, hc = np.unique(hl[(hl > 0) & ~np.isin(hl, border)], return_counts=True)
        if len(hids) == 0:
            continue
        hole = hl == hids[np.argmax(hc)]
        hy, hx = np.nonzero(hole)
        if len(hx) < 4:
            continue
        # pixel centres; opening boundary sits half a pixel outside the hole pixels
        u = hx + x0 + 0.5
        v = hy + y0 + 0.5
        s, d = u + v, u - v
        tl = _extreme(u, v, s, False)
        br = _extreme(u, v, s, True)
        tr = _extreme(u, v, d, True)
        bl = _extreme(u, v, d, False)
        pts = np.array([tl, tr, br, bl])
        pts += np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        gates.append(pts)
    return gates


def detect_corners(img: np.ndarray):
    """Same output format as the CNN decoder: list of 4 arrays of (u, v, score)."""
    out = [[] for _ in range(4)]
    for g in detect_gates(img):
        for c in range(4):
            out[c].append((g[c, 0], g[c, 1], 1.0))
    return [np.array(o).reshape(-1, 3) for o in out]
