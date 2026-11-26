"""Pinhole FPV camera and a small numpy rasteriser.

Scene: textured ground plane (procedural albedo tile, bilinear sampled per
pixel ray), sky gradient with fog towards the horizon, and the gates as flat
quads (frame bars + legs) filled with half-plane tests after near-plane
clipping, painted far-to-near. Optional 2x supersampling gives anti-aliased
edges so that sub-pixel corner accuracy is meaningful.

Camera frame: x right, y down, z forward (OpenCV). It is mounted on the FRD
body with an up-tilt, like an FPV camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GATE_COLOURS = {0: np.array([1.0, 0.45, 0.08]), 1: np.array([0.18, 0.18, 0.2])}


@dataclass
class CameraModel:
    width: int = 192
    height: int = 144
    hfov_deg: float = 90.0
    uptilt_deg: float = 35.0

    @property
    def fx(self) -> float:
        return 0.5 * self.width / np.tan(np.radians(0.5 * self.hfov_deg))

    @property
    def K(self) -> np.ndarray:
        f = self.fx
        return np.array([[f, 0.0, 0.5 * self.width], [0.0, f, 0.5 * self.height], [0.0, 0.0, 1.0]])

    @property
    def R_bc(self) -> np.ndarray:
        """Camera -> body rotation (columns = camera axes in FRD body)."""
        t = np.radians(self.uptilt_deg)
        return np.array([[0.0, np.sin(t), np.cos(t)], [1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)]])

    def project(self, R_wb, p_wb, pts_w):
        """World points -> (pixels Nx2, depth N)."""
        R_wc = R_wb @ self.R_bc
        Xc = (np.asarray(pts_w) - p_wb) @ R_wc
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * Xc[:, 0] / z + 0.5 * self.width
            v = self.fx * Xc[:, 1] / z + 0.5 * self.height
        return np.stack([u, v], 1), z


def _value_noise(rng, n, octaves=(4, 8, 16, 32, 64)):
    img = np.zeros((n, n))
    amp = 1.0
    for o in octaves:
        g = rng.random((o + 1, o + 1))
        g[-1, :] = g[0, :]
        g[:, -1] = g[:, 0]
        x = np.linspace(0, o, n, endpoint=False)
        i = x.astype(int)
        f = x - i
        f = f * f * (3 - 2 * f)
        a = g[i][:, i] * (1 - f)[None, :] + g[i][:, i + 1] * f[None, :]
        b = g[i + 1][:, i] * (1 - f)[None, :] + g[i + 1][:, i + 1] * f[None, :]
        img += amp * (a * (1 - f)[:, None] + b * f[:, None])
        amp *= 0.55
    img -= img.min()
    return img / img.max()


class Scene:
    """Ground texture + sky. ``distractors`` adds orange-ish blobs on the ground."""

    def __init__(self, seed: int = 0, tile_m: float = 24.0, n: int = 512, distractors: int = 6):
        rng = np.random.default_rng(seed)
        self.tile_m = tile_m
        self.n = n
        a = _value_noise(rng, n)
        b = _value_noise(rng, n, octaves=(16, 32, 64, 128))
        grass = np.array([0.22, 0.42, 0.16])
        dirt = np.array([0.45, 0.36, 0.25])
        mix = np.clip((a - 0.45) * 3.0, 0, 1)[..., None]
        tex = grass * (1 - mix) + dirt * mix
        tex *= (0.75 + 0.5 * b)[..., None]
        tex *= (0.85 + 0.3 * rng.random((n, n)))[..., None]  # fine grain
        # faint mowing stripes / track markings
        yy, xx = np.mgrid[0:n, 0:n]
        tex *= (1.0 - 0.06 * ((xx // 16) % 2))[..., None]
        for _ in range(distractors):
            cx, cy, r = rng.integers(0, n), rng.integers(0, n), rng.integers(4, 14)
            col = np.array([rng.uniform(0.7, 1.0), rng.uniform(0.2, 0.6), rng.uniform(0.0, 0.2)])
            dx = np.minimum((xx - cx) % n, (cx - xx) % n)
            dy = np.minimum((yy - cy) % n, (cy - yy) % n)
            tex[dx * dx + dy * dy < r * r] = col
        self.tex = np.clip(tex, 0, 1).astype(np.float32)
        self.sky_top = np.array([0.35, 0.55, 0.85])
        self.sky_hor = np.array([0.78, 0.84, 0.9])
        self.sun = np.array([0.4, -0.3, -0.85])
        self.sun /= np.linalg.norm(self.sun)

    def ground(self, x, y):
        s = self.n / self.tile_m
        i = (x * s).astype(np.int64) % self.n
        j = (y * s).astype(np.int64) % self.n
        return self.tex[j, i]


def _clip_near(P, znear=0.05):
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        ain, bin_ = a[2] >= znear, b[2] >= znear
        if ain:
            out.append(a)
        if ain != bin_:
            t = (znear - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


class Renderer:
    def __init__(self, cam: CameraModel | None = None, scene: Scene | None = None, supersample: int = 3):
        self.cam = cam or CameraModel()
        self.scene = scene or Scene()
        self.ss = supersample
        W, H = self.cam.width, self.cam.height
        f = self.cam.fx
        u = (np.arange(W) + 0.5) - 0.5 * W
        v = (np.arange(H) + 0.5) - 0.5 * H
        uu, vv = np.meshgrid(u, v)
        rays = np.stack([uu / f, vv / f, np.ones_like(uu)], -1)
        self.rays_c = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3)
        # sub-pixel sample offsets for gate coverage (ss x ss grid inside each pixel)
        o = (np.arange(self.ss) + 0.5) / self.ss
        self.sub = np.stack(np.meshgrid(o, o), -1).reshape(-1, 2)

    def render(self, R_wb, p_wb, gates, gate_tint=None) -> np.ndarray:
        """Returns float32 HxWx3 in [0, 1]."""
        cam, sc = self.cam, self.scene
        W, H = cam.width, cam.height
        R_wc = R_wb @ cam.R_bc
        rays = self.rays_c @ R_wc.T
        rz = rays[:, 2]
        pz = p_wb[2]
        elev = np.clip(-rz, 0, 1)[:, None]
        img = (sc.sky_hor * (1 - elev) + sc.sky_top * elev).astype(np.float32)
        g = rz > 1e-4
        t = -pz / rz[g]
        col = sc.ground(p_wb[0] + t * rays[g, 0], p_wb[1] + t * rays[g, 1])
        fog = np.clip(t / 120.0, 0, 1)[:, None]
        img[g] = col * (1 - fog) + sc.sky_hor * fog
        img = img.reshape(H, W, 3)
        quads = []
        for gi, gate in enumerate(gates):
            for P, mat in gate.quads_world():
                quads.append((np.linalg.norm(P.mean(0) - p_wb), P, mat, gi))
        quads.sort(key=lambda q: -q[0])
        f = cam.fx
        sub = self.sub
        for dist, P, mat, gi in quads:
            Xc = (P - p_wb) @ R_wc
            if np.all(Xc[:, 2] < 0.05):
                continue
            Xc = _clip_near(Xc)
            if len(Xc) < 3:
                continue
            u = f * Xc[:, 0] / Xc[:, 2] + 0.5 * W
            v = f * Xc[:, 1] / Xc[:, 2] + 0.5 * H
            x0, x1 = int(max(0, np.floor(u.min()))), int(min(W, np.ceil(u.max())))
            y0, y1 = int(max(0, np.floor(v.min()))), int(min(H, np.ceil(v.max())))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            px = xx[..., None] + sub[:, 0]
            py = yy[..., None] + sub[:, 1]
            area = np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)
            sgn = 1.0 if area > 0 else -1.0
            inside = np.ones(px.shape, bool)
            nv = len(u)
            for i in range(nv):
                ax, ay, bx, by = u[i], v[i], u[(i + 1) % nv], v[(i + 1) % nv]
                inside &= sgn * ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) >= 0
            cov = inside.mean(-1, dtype=np.float32)
            if not cov.any():
                continue
            n = np.cross(P[1] - P[0], P[2] - P[0])
            n /= np.linalg.norm(n) + 1e-12
            shade = 0.55 + 0.45 * abs(float(n @ sc.sun))
            base = GATE_COLOURS[mat] if (mat != 0 or gate_tint is None) else gate_tint[gi]
            fg = min(1.0, dist / 120.0)
            colr = (base * shade * (1 - fg) + sc.sky_hor * fg).astype(np.float32)
            blk = img[y0:y1, x0:x1]
            blk += cov[..., None] * (colr - blk)
        return img

    def corner_labels(self, R_wb, p_wb, gates, margin: float = 0.0):
        """Per gate: (pixels 4x2, visible mask 4, mirrored flag, depth 4)."""
        cam = self.cam
        out = []
        for gate in gates:
            mirrored = bool(np.dot(p_wb - gate.center, gate.normal) > 0)
            pts = gate.inner_corners_world(mirrored)
            uv, z = cam.project(R_wb, p_wb, pts)
            vis = (z > 0.3) & (uv[:, 0] >= -margin) & (uv[:, 0] < cam.width + margin) &                   (uv[:, 1] >= -margin) & (uv[:, 1] < cam.height + margin)
            out.append((uv, vis, mirrored, z))
        return out
