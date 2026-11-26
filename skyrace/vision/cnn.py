"""GateNet: a small fully-convolutional corner-heatmap network, in plain numpy.

Input  3 x 144 x 192 RGB in [0, 1].
Output 4 x 36 x 48 heatmaps (stride 4), one channel per inner-corner type
(TL, TR, BR, BL as seen in the image), CenterNet-style Gaussian targets.

    c1  conv3x3  3->16 s2          72x96
    c2  conv3x3 16->16             72x96
    c3  conv3x3 16->32 s2          36x48   (skip)
    c4  conv3x3 32->32             36x48
    c5  conv3x3 32->48 s2          18x24
    c6  conv3x3 48->48 dilation 2  18x24   (context)
        nearest x2 upsample, concat skip -> 80 ch @ 36x48
    c7  conv3x3 80->32             36x48
    head conv1x1 32->4 (logits; sigmoid at inference)

Every conv is conv + bias + ReLU. Forward *and backward* passes are written
here by hand (batched im2col with ``as_strided`` + one GEMM per layer;
col2im as 9 strided accumulations), so training needs nothing but numpy.
Tensors are laid out (C, B, H, W) so the GEMMs need no transposes.
"""
from __future__ import annotations

import numpy as np

# (name, cin, cout, stride, dilation)
LAYERS = [
    ("c1", 3, 16, 2, 1),
    ("c2", 16, 16, 1, 1),
    ("c3", 16, 32, 2, 1),
    ("c4", 32, 32, 1, 1),
    ("c5", 32, 48, 2, 1),
    ("c6", 48, 48, 1, 2),
    ("c7", 80, 32, 1, 1),
]
HEAD = ("head", 32, 4)
STRIDE = 4
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _out_size(n, stride):
    return (n - 1) // stride + 1


def im2col(x, stride=1, dilation=1):
    """x: (C, B, H, W) -> cols (C*9, B*Ho*Wo), padding = dilation."""
    C, B, H, W = x.shape
    p = dilation
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho, Wo = _out_size(H, stride), _out_size(W, stride)
    s0, s1, s2, s3 = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(C, 3, 3, B, Ho, Wo),
        strides=(s0, s2 * dilation, s3 * dilation, s1, s2 * stride, s3 * stride))
    return cols.reshape(C * 9, B * Ho * Wo), (Ho, Wo)


def col2im(dcols, shape, stride=1, dilation=1):
    """Adjoint of :func:`im2col`."""
    C, B, H, W = shape
    p = dilation
    Ho, Wo = _out_size(H, stride), _out_size(W, stride)
    d = dcols.reshape(C, 3, 3, B, Ho, Wo)
    dxp = np.zeros((C, B, H + 2 * p, W + 2 * p), dcols.dtype)
    for ky in range(3):
        for kx in range(3):
            y0, x0 = ky * dilation, kx * dilation
            dxp[:, :, y0:y0 + stride * (Ho - 1) + 1:stride, x0:x0 + stride * (Wo - 1) + 1:stride] += d[:, ky, kx]
    return dxp[:, :, p:p + H, p:p + W]


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    P = {}
    for name, cin, cout, stride, dil in LAYERS:
        P[name + ".w"] = (rng.normal(0, np.sqrt(2.0 / (cin * 9)), (cout, cin, 3, 3))).astype(np.float32)
        P[name + ".b"] = np.zeros(cout, np.float32)
    P["head.w"] = (rng.normal(0, 0.01, (4, 32, 1, 1))).astype(np.float32)
    P["head.b"] = np.full(4, -2.2, np.float32)   # prior ~10 % so early training is stable
    return P


def forward(P, x, cache=False):
    """x: (3, B, H, W) float32 -> logits (4, B, H/4, W/4) [, cache]."""
    cs = {}
    skip = None
    for name, cin, cout, stride, dil in LAYERS:
        if name == "c7":
            up = x.repeat(2, axis=2).repeat(2, axis=3)
            x = np.concatenate([up, skip], 0)
        cols, (Ho, Wo) = im2col(x, stride, dil)
        z = P[name + ".w"].reshape(cout, -1) @ cols
        z += P[name + ".b"][:, None]
        a = np.maximum(z, 0.0).reshape(cout, x.shape[1], Ho, Wo)
        if cache:
            cs[name] = (cols, x.shape, z > 0)
        x = a
        if name == "c4":
            skip = x
    C, B, H, W = x.shape
    feat = x.reshape(C, -1)
    logits = P["head.w"].reshape(4, C) @ feat + P["head.b"][:, None]
    logits = logits.reshape(4, B, H, W)
    if cache:
        cs["head"] = feat
        return logits, cs
    return logits


def backward(P, cs, dlogits):
    """Gradients of all parameters given dL/dlogits (4, B, H, W)."""
    g = {}
    _, B, H, W = dlogits.shape
    dl = dlogits.reshape(4, -1)
    g["head.w"] = (dl @ cs["head"].T).reshape(P["head.w"].shape)
    g["head.b"] = dl.sum(1)
    dx = (P["head.w"].reshape(4, -1).T @ dl).reshape(32, B, H, W)
    dskip = None
    for name, cin, cout, stride, dil in reversed(LAYERS):
        cols, xshape, mask = cs[name]
        dz = dx.reshape(cout, -1) * mask
        g[name + ".w"] = (dz @ cols.T).reshape(P[name + ".w"].shape)
        g[name + ".b"] = dz.sum(1)
        dx = col2im(P[name + ".w"].reshape(cout, -1).T @ dz, xshape, stride, dil)
        if name == "c7":
            dup, dskip = dx[:48], dx[48:]
            C, Bb, Hh, Ww = dup.shape
            dx = dup.reshape(C, Bb, Hh // 2, 2, Ww // 2, 2).sum((3, 5))
        if name == "c5":
            dx = dx + dskip          # c4's output feeds both c5 and the skip
    return g


def focal_loss_grad(logits, target, weight=None, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss and its gradient w.r.t. the logits."""
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    pos = target >= 0.97
    npos = max(1.0, float(pos.sum()))
    lp, l1p = np.log(p), np.log(1 - p)
    nw = (1 - target) ** beta
    if weight is not None:
        nw = nw * weight
    loss_pos = -((1 - p) ** alpha * lp)[pos].sum()
    loss_neg = -(nw * p ** alpha * l1p)[~pos].sum()
    g_pos = alpha * p * (1 - p) ** alpha * lp - (1 - p) ** (alpha + 1)
    g_neg = -nw * (alpha * p ** alpha * (1 - p) * l1p - p ** (alpha + 1))
    grad = np.where(pos, g_pos, g_neg) / npos
    return float((loss_pos + loss_neg) / npos), grad.astype(np.float32)


class GateNetNumpy:
    """Inference wrapper: H x W x 3 image -> 4 x H/4 x W/4 heatmaps in (0, 1)."""

    def __init__(self, weights: dict):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[:, None], dtype=np.float32)
        lg = forward(self.w, x)[:, 0]
        return 1.0 / (1.0 + np.exp(-lg))


def decode_heatmaps(hm: np.ndarray, thresh: float = 0.3, max_per_class: int = 12):
    """Local maxima (3x3) above ``thresh`` with 3x3 centroid refinement.

    Returns a list of 4 arrays (one per corner type) of rows (u, v, score) in
    input-image pixel coordinates (continuous, pixel i spans [i, i+1)).
    """
    C, H, W = hm.shape
    pad = np.pad(hm, ((0, 0), (1, 1), (1, 1)), constant_values=0.0)
    nb = np.stack([pad[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
    peak = (hm >= nb.max(0)) & (hm > thresh)
    out = []
    for c in range(C):
        ys, xs = np.nonzero(peak[c])
        sc = hm[c, ys, xs]
        order = np.argsort(-sc)[:max_per_class]
        rows = []
        for k in order:
            y, x = ys[k], xs[k]
            win = pad[c, y:y + 3, x:x + 3]
            wsum = win.sum()
            dy = (win.sum(1) @ np.array([-1.0, 0.0, 1.0])) / wsum
            dx = (win.sum(0) @ np.array([-1.0, 0.0, 1.0])) / wsum
            rows.append((STRIDE * (x + dx + 0.5), STRIDE * (y + dy + 0.5), float(sc[k])))
        out.append(np.array(rows).reshape(-1, 3))
    return out
