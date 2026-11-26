"""Train GateNet with the hand-written numpy backprop in :mod:`skyrace.vision.cnn`.

Loss: CenterNet penalty-reduced focal loss (Zhou et al. 2019, "Objects as
Points") on Gaussian corner heatmaps (sigma = 1 heatmap cell); corners
further than ``MAX_DEPTH`` are "don't care". Optimiser: Adam with a short
linear warm-up and cosine decay. Augmentation: horizontal flip (which swaps
TL<->TR and BL<->BR) on top of the render-time augmentation.
"""
from __future__ import annotations

import time

import numpy as np

from .cnn import STRIDE, backward, focal_loss_grad, forward, init_params

MAX_DEPTH = 22.0


def heatmap_targets(labels, H, W, sigma=1.0):
    """labels: N x G x 4 x 4 (u, v, vis, depth) -> (targets, weights), both N x 4 x H x W."""
    N = labels.shape[0]
    tgt = np.zeros((N, 4, H, W), np.float32)
    wgt = np.ones((N, 4, H, W), np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    for n in range(N):
        for g in range(labels.shape[1]):
            for c in range(4):
                u, v, vis, z = labels[n, g, c]
                if vis < 0.5:
                    continue
                cx, cy = u / STRIDE - 0.5, v / STRIDE - 0.5
                if not (-1 <= cx <= W and -1 <= cy <= H):
                    continue
                d2 = (xs - cx) ** 2 + (ys - cy) ** 2
                if z > MAX_DEPTH:
                    wgt[n, c][d2 < 4.0] = 0.0
                    continue
                np.maximum(tgt[n, c], np.exp(-d2 / (2 * sigma ** 2)), out=tgt[n, c])
    return tgt, wgt


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in params:
            g = grads[k] + self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def train(imgs, labels, epochs=8, batch=16, lr=2e-3, seed=0, log=print, params=None):
    rng = np.random.default_rng(seed)
    N, H, W, _ = imgs.shape
    Ho, Wo = H // STRIDE, W // STRIDE
    tgt, wgt = heatmap_targets(labels, Ho, Wo)
    P = params or init_params(seed)
    opt = Adam(P, lr)
    steps = epochs * ((N + batch - 1) // batch)
    warm = max(1, steps // 30)
    hist = []
    t0 = time.time()
    k = 0
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            x = imgs[idx].astype(np.float32).transpose(3, 0, 1, 2) / 255.0     # (3, B, H, W)
            tb = tgt[idx].transpose(1, 0, 2, 3)                                  # (4, B, h, w)
            wb = wgt[idx].transpose(1, 0, 2, 3)
            if rng.random() < 0.5:
                x = x[..., ::-1]
                tb = tb[[1, 0, 3, 2]][..., ::-1]
                wb = wb[[1, 0, 3, 2]][..., ::-1]
            x = np.ascontiguousarray(x)
            logits, cache = forward(P, x, cache=True)
            loss, dl = focal_loss_grad(logits, np.ascontiguousarray(tb), np.ascontiguousarray(wb))
            g = backward(P, cache, dl)
            lr_k = lr * min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * k / steps))
            opt.step(P, g, lr_k)
            tot += loss * len(idx)
            k += 1
        hist.append(tot / N)
        log(f"epoch {ep + 1}/{epochs}  loss {tot / N:.4f}  ({time.time() - t0:.0f} s)")
    return P, hist
