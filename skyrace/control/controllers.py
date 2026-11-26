"""Trajectory trackers that output (collective thrust [N], body rates [rad/s]).

* :class:`FlatnessTracker` - minimum-snap reference + differential-flatness
  feed-forward (thrust, attitude, body rates) + PD on position/velocity and a
  proportional attitude loop (Mellinger & Kumar 2011 / Faessler 2018 style).
* :class:`MpcTracker` - linear time-varying MPC around the same reference:
  error state (dp, dv, dtheta) with R = R_ref exp([dtheta]x), inputs
  (dT, dw), box constraints on thrust and rates, condensed and solved with the
  ADMM QP in :mod:`skyrace.control.qp` (warm-started).
* :class:`PurePursuit` - baseline: chase a look-ahead point on the polyline
  through the gate centres at a fixed cruise speed, no planning.

All three share :func:`accel_to_command`, the thrust/attitude mapping.
"""
from __future__ import annotations

import numpy as np

from ..planning.minsnap import E3, G
from ..planning.reference import Reference, attitude_from_force
from ..so3 import exp_so3, log_so3, skew
from .qp import solve_qp


def accel_to_command(a_cmd, v, R, yaw, mass, drag, w_ff=np.zeros(3), k_att=9.0,
                     t_min=1.0, t_max=36.0, w_max=np.radians(600)):
    f = mass * (a_cmd - G * E3) + drag * v
    _, R_des = attitude_from_force(f, yaw)
    T = float(np.clip(f @ (-R[:, 2]), t_min, t_max))
    e_R = log_so3(R.T @ R_des)
    w = R.T @ R_des @ w_ff + k_att * e_R
    return T, np.clip(w, -w_max, w_max)


class FlatnessTracker:
    name = "minsnap+flatness"

    def __init__(self, ref: Reference, mass=0.65, drag=0.15, kp=(9.0, 9.0, 12.0), kv=(4.5, 4.5, 6.0),
                 k_att=9.0, latency=0.03):
        self.ref, self.m, self.c = ref, mass, drag
        self.kp, self.kv = np.array(kp), np.array(kv)
        self.k_att = k_att
        self.latency = latency

    def command(self, t, est):
        r = self.ref.at(t + self.latency)
        p, v, R = predict(est, self.latency)
        a = r["a"] + self.kp * (r["p"] - p) + self.kv * (r["v"] - v)
        return accel_to_command(a, v, R, r["yaw"], self.m, self.c, r["w"], self.k_att)

    def reference(self, t):
        return self.ref.at(t)


def predict(est, dt):
    """Constant-acceleration / constant-rate forward prediction (latency compensation)."""
    a = est.get("a", np.zeros(3))
    p = est["p"] + est["v"] * dt + 0.5 * a * dt * dt
    v = est["v"] + a * dt
    R = est["R"] @ exp_so3(est["w"] * dt)
    return p, v, R


class MpcTracker:
    name = "minsnap+MPC"

    def __init__(self, ref: Reference, mass=0.65, drag=0.15, N=16, dt=0.04,
                 q_pos=80.0, q_vel=8.0, q_att=3.0, r_thrust=0.01, r_rate=0.6, q_term=3.0,
                 t_min=1.0, t_max=36.0, w_max=np.radians(600), latency=0.03):
        self.ref, self.m, self.c = ref, mass, drag
        self.N, self.dt = N, dt
        self.Q = np.diag([q_pos] * 3 + [q_vel] * 3 + [q_att] * 3)
        self.QN = q_term * self.Q
        self.Ru = np.diag([r_thrust] + [r_rate] * 3)
        self.t_min, self.t_max, self.w_max = t_min, t_max, w_max
        self.latency = latency
        self.U_prev = None
        self.y_prev = None
        self.last_iters = 0
        self.solve_stats = []

    def _lin(self, r):
        m, c, dt = self.m, self.c, self.dt
        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 3:6] = -(c / m) * np.eye(3)
        A[3:6, 6:9] = (r["T"] / m) * r["R"] @ skew(E3)
        A[6:9, 6:9] = -skew(r["w"])
        B = np.zeros((9, 4))
        B[3:6, 0] = -(1.0 / m) * r["R"][:, 2]
        B[6:9, 1:4] = np.eye(3)
        Ad = np.eye(9) + A * dt + 0.5 * (A @ A) * dt * dt
        Bd = (np.eye(9) + 0.5 * A * dt) @ B * dt
        return Ad, Bd

    def command(self, t, est):
        N, nx, nu = self.N, 9, 4
        t0 = t + self.latency
        p, v, R = predict(est, self.latency)
        refs = [self.ref.at(t0 + k * self.dt) for k in range(N + 1)]
        r0 = refs[0]
        e0 = np.r_[p - r0["p"], v - r0["v"], log_so3(r0["R"].T @ R)]
        # condensed prediction X = Phi e0 + Gam U  (X = e_1..e_N)
        Phi = np.zeros((N * nx, nx))
        Gam = np.zeros((N * nx, N * nu))
        lin = [self._lin(refs[k]) for k in range(N)]
        M = np.eye(nx)
        for k in range(N):
            Ad, Bd = lin[k]
            M = Ad @ M
            Phi[k * nx:(k + 1) * nx] = M
            Gam[k * nx:(k + 1) * nx, k * nu:(k + 1) * nu] = Bd
            if k > 0:
                Gam[k * nx:(k + 1) * nx, :k * nu] = Ad @ Gam[(k - 1) * nx:k * nx, :k * nu]
        Qb = np.kron(np.eye(N), self.Q)
        Qb[-nx:, -nx:] = self.QN
        Rb = np.kron(np.eye(N), self.Ru)
        H = Gam.T @ Qb @ Gam + Rb
        g = Gam.T @ Qb @ (Phi @ e0)
        lb = np.zeros(N * nu)
        ub = np.zeros(N * nu)
        for k in range(N):
            rk = refs[k]
            lb[k * nu] = self.t_min - rk["T"]
            ub[k * nu] = self.t_max - rk["T"]
            lb[k * nu + 1:(k + 1) * nu] = -self.w_max - rk["w"]
            ub[k * nu + 1:(k + 1) * nu] = self.w_max - rk["w"]
        x0 = None
        if self.U_prev is not None:
            x0 = np.r_[self.U_prev[nu:], self.U_prev[-nu:]]
        res = solve_qp(H, g, np.eye(N * nu), lb, ub, x0=x0, y0=None, rho=1.0,
                       eps_abs=1e-4, eps_rel=1e-4, max_iter=400)
        U = np.clip(res.x, lb, ub)
        self.U_prev = U
        self.last_iters = res.iters
        self.solve_stats.append((res.iters, res.converged))
        T = r0["T"] + U[0]
        w = r0["w"] + U[1:4]
        return float(T), w

    def reference(self, t):
        return self.ref.at(t)


class PurePursuit:
    name = "pure-pursuit"

    def __init__(self, waypoints, speed=8.0, lookahead=3.0, la_gain=0.3, kv=3.0, a_max=25.0,
                 mass=0.65, drag=0.15, yaw0=0.0):
        self.W = np.asarray(waypoints, float)
        self.speed, self.L0, self.kL, self.kv, self.a_max = speed, lookahead, la_gain, kv, a_max
        self.m, self.c = mass, drag
        self.seg = 0
        self.yaw = yaw0
        d = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
        self.s_knots = np.r_[0.0, np.cumsum(d)]

    def _point_at(self, s):
        s = min(max(s, 0.0), self.s_knots[-1])
        i = min(np.searchsorted(self.s_knots, s, side="right") - 1, len(self.W) - 2)
        f = (s - self.s_knots[i]) / max(self.s_knots[i + 1] - self.s_knots[i], 1e-9)
        return self.W[i] + f * (self.W[i + 1] - self.W[i])

    def _project(self, p):
        best, best_s, seg = np.inf, 0.0, self.seg
        for i in range(self.seg, min(self.seg + 2, len(self.W) - 1)):
            a, b = self.W[i], self.W[i + 1]
            ab = b - a
            f = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
            d = np.linalg.norm(a + f * ab - p)
            if d < best:
                best, best_s, seg = d, self.s_knots[i] + f * np.linalg.norm(ab), i
        if seg > self.seg and best_s > self.s_knots[seg] + 0.05:
            self.seg = seg
        return best_s

    def command(self, t, est):
        p, v, R = est["p"], est["v"], est["R"]
        s = self._project(p)
        L = self.L0 + self.kL * np.linalg.norm(v)
        tgt = self._point_at(s + L)
        d = tgt - p
        end_dist = self.s_knots[-1] - s
        spd = self.speed if end_dist > 1.0 else 0.0
        v_des = spd * d / max(np.linalg.norm(d), 1e-6)
        a = self.kv * (v_des - v)
        n = np.linalg.norm(a)
        if n > self.a_max:
            a *= self.a_max / n
        if np.linalg.norm(v_des[:2]) > 0.5:
            self.yaw = float(np.arctan2(v_des[1], v_des[0]))
        return accel_to_command(a, v, R, self.yaw, self.m, self.c)

    def reference(self, t):
        return None
