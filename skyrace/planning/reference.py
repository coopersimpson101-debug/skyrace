"""Differential flatness: polynomial (p, v, a, j) + yaw -> thrust, attitude, body rates.

With linear drag c and NED gravity, Newton gives
    f := m (a - g e3) + c v = -T R e3
so T = ||f||, body z axis z_b = -f / T. Yaw psi fixes x_b through the
intermediate heading x_c = (cos psi, sin psi, 0):  y_b = z_b x x_c / ||.||,
x_b = y_b x z_b (Mellinger & Kumar 2011, adapted to NED/FRD).

Yaw follows the velocity heading (the FPV camera looks where the quad is
going) and falls back to the acceleration heading at low speed. Body rates
are obtained from the sampled rotation sequence, w_k = log(R_k^T R_{k+1}) / dt,
which is exact up to the sampling interval and also yields the yaw rate.
"""
from __future__ import annotations

import numpy as np

from ..so3 import log_so3
from .minsnap import E3, G, PolyTraj


def attitude_from_force(f, yaw):
    T = np.linalg.norm(f)
    z = -f / max(T, 1e-9)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, xc)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    else:
        y /= ny
    x = np.cross(y, z)
    return T, np.c_[x, y, z]


class Reference:
    """Sampled reference along a PolyTraj (dt = 5 ms) with linear interpolation."""

    def __init__(self, traj: PolyTraj, mass=0.65, drag=0.15, yaw0=0.0, dt=0.005, t_offset=0.0,
                 hold_yaw=False):
        self.traj = traj
        self.mass, self.drag = mass, drag
        self.dt = dt
        self.t_offset = t_offset
        t = np.arange(0.0, traj.total + dt, dt)
        self.t = t
        self.p = traj.eval(t, 0)
        self.v = traj.eval(t, 1)
        self.a = traj.eval(t, 2)
        self.j = traj.eval(t, 3)
        yaw = np.empty(len(t))
        prev = yaw0
        for k in range(len(t)):
            h = self.v[k, :2]
            if not hold_yaw and np.linalg.norm(h) > 1.5:
                prev = np.arctan2(h[1], h[0])
            elif not hold_yaw and np.linalg.norm(h) > 0.2:
                w = (np.linalg.norm(h) - 0.2) / 1.3   # blend in the heading as speed builds up
                prev = prev + w * np.angle(np.exp(1j * (np.arctan2(h[1], h[0]) - prev)))
            yaw[k] = prev
        self.yaw = np.unwrap(yaw)
        f = mass * (self.a - G * E3) + drag * self.v
        self.T = np.zeros(len(t))
        self.R = np.zeros((len(t), 3, 3))
        for k in range(len(t)):
            self.T[k], self.R[k] = attitude_from_force(f[k], self.yaw[k])
        self.w = np.zeros((len(t), 3))
        for k in range(len(t) - 1):
            self.w[k] = log_so3(self.R[k].T @ self.R[k + 1]) / dt
        self.w[-1] = self.w[-2] if len(t) > 1 else 0.0

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, t: float) -> dict:
        s = (t - self.t_offset) / self.dt
        s = min(max(s, 0.0), len(self.t) - 1.000001)
        k = int(s)
        f = s - k
        lerp = lambda X: X[k] * (1 - f) + X[k + 1] * f  # noqa: E731
        R = self.R[k] if f < 0.5 else self.R[k + 1]
        return dict(p=lerp(self.p), v=lerp(self.v), a=lerp(self.a), j=lerp(self.j), T=lerp(self.T),
                    R=R, w=lerp(self.w), yaw=lerp(self.yaw))
