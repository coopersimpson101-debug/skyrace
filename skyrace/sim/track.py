"""Race gates and tracks (NED world frame, metres)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INNER = 1.4    # inner opening (m), square
OUTER = 1.9    # outer frame size (m)


@dataclass
class Gate:
    center: np.ndarray            # NED
    yaw: float                    # rad, direction of travel through the gate
    inner: float = INNER
    outer: float = OUTER

    @property
    def R(self) -> np.ndarray:
        """Gate frame -> world. Gate x = through-direction, y = right, z = down."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def normal(self) -> np.ndarray:
        return self.R[:, 0]

    def inner_corners_local(self, mirrored: bool = False) -> np.ndarray:
        """TL, TR, BR, BL of the opening as seen when flying through (+x)."""
        h = 0.5 * self.inner
        pts = np.array([[0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h]], dtype=float)
        if mirrored:  # seen from behind: image-left is gate +y
            pts = pts[[1, 0, 3, 2]]
        return pts

    def inner_corners_world(self, mirrored: bool = False) -> np.ndarray:
        return self.center + self.inner_corners_local(mirrored) @ self.R.T

    def to_local(self, p_world) -> np.ndarray:
        return (np.asarray(p_world) - self.center) @ self.R

    def quads_world(self):
        """Frame bars (4) and legs (2) as world-space quads + a material id."""
        h, H = 0.5 * self.inner, 0.5 * self.outer
        d = 0.06  # half depth of the frame (thin)
        bars = [
            [(-H, -H), (H, -H), (H, -h), (-H, -h)],     # top (y, z)
            [(-H, h), (H, h), (H, H), (-H, H)],         # bottom
            [(-H, -h), (-h, -h), (-h, h), (-H, h)],     # left
            [(h, -h), (H, -h), (H, h), (h, h)],         # right
        ]
        out = []
        for b in bars:
            for x in (-d, d):
                pts = np.array([[x, y, z] for (y, z) in b])
                out.append((self.center + pts @ self.R.T, 0))
        leg_h = -self.center[2] - H  # from bottom of the frame to the ground
        if leg_h > 0.05:
            for y0 in (-H, H - 0.08):
                pts = np.array([[0, y0, H], [0, y0 + 0.08, H], [0, y0 + 0.08, H + leg_h], [0, y0, H + leg_h]])
                out.append((self.center + pts @ self.R.T, 1))
        return out


@dataclass
class Track:
    name: str
    gates: list
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_yaw: float = 0.0
    laps_order: list | None = None   # gate indices in race order (default: all, then gate 0 again)

    def order(self):
        if self.laps_order is not None:
            return list(self.laps_order)
        return list(range(len(self.gates))) + [0]


def _auto_yaw(points, start):
    """Gate yaw = heading of the chord from the previous to the next point."""
    yaws = []
    n = len(points)
    for i in range(n):
        prev = start if i == 0 else points[i - 1]
        nxt = points[(i + 1) % n]
        d = nxt - prev
        yaws.append(float(np.arctan2(d[1], d[0])))
    return yaws


def make_track(name: str = "loop7") -> Track:
    if name == "loop7":
        pts = np.array([
            [10.0, 0.0, -2.0],
            [22.0, 6.0, -3.0],
            [22.0, 18.0, -2.2],
            [10.0, 23.0, -3.5],
            [-2.0, 19.0, -2.0],
            [-6.0, 8.0, -2.6],
            [-2.0, -1.5, -2.0],
        ])
        start = np.array([0.0, 0.0, 0.0])
    elif name == "slalom5":
        pts = np.array([
            [8.0, 0.0, -2.0],
            [16.0, 4.0, -2.5],
            [24.0, 0.0, -2.0],
            [32.0, -4.0, -3.0],
            [40.0, 0.0, -2.0],
        ])
        start = np.array([0.0, 0.0, 0.0])
        yaws = _auto_yaw(pts, start)
        yaws[-1] = 0.0
        return Track(name, [Gate(p, y) for p, y in zip(pts, yaws)], start, 0.0,
                     laps_order=list(range(len(pts))))
    else:
        raise ValueError(name)
    yaws = _auto_yaw(pts, start)
    return Track(name, [Gate(p, y) for p, y in zip(pts, yaws)], start, 0.0)
