"""Minimum-snap piecewise polynomials through waypoints, with time allocation.

Closed-form unconstrained QP of Richter, Bry & Roy (2013/2016):

* each segment i is a degree-7 polynomial p_i(tau), tau in [0, T_i]
* snap cost J_i = int_0^Ti (p_i''''(tau))^2 dtau = c_i^T Q_i c_i
* endpoint derivatives d_i = A_i c_i (orders 0..3 at both ends), so
  J = sum d_i^T A_i^-T Q_i A_i^-1 d_i
* continuity is enforced by construction: a selection matrix S maps the
  vector D of *unique* knot derivatives (4 per knot) to the stacked d_i, so
  J = D^T R D with R = S^T blkdiag(A^-T Q A^-1) S
* D = [D_F; D_P] (fixed: knot positions, start at rest; free: interior
  velocity/acceleration/jerk and the final derivatives) and the optimum is
      D_P* = -R_PP^-1 R_PF D_F
  per axis (the same R for x, y, z).

Time allocation. Uniformly scaling all durations by alpha leaves the spatial
path unchanged (all fixed derivatives except positions are zero), so the
lap time for a given *ratio* vector r is alpha_min(r) * sum(r), where
alpha_min is the smallest scale at which the trajectory is dynamically
feasible: collective thrust ||m (a - g e3) + c v|| within [T_min, T_max]
and body-rate magnitude ||(I - z z^T)(m j + c a)|| / T below w_max.
We minimise that lap time over log r by numerical-gradient descent with
backtracking (a direct version of the Richter et al. time-allocation step,
with the time penalty replaced by the feasibility-scaled duration).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

G = 9.80665
E3 = np.array([0.0, 0.0, 1.0])
NCOEF = 8
NDER = 4


def _deriv_row(k: int, t: float) -> np.ndarray:
    """Row r with r @ c = d^k/dt^k sum c_n t^n at t."""
    r = np.zeros(NCOEF)
    for n in range(k, NCOEF):
        r[n] = factorial(n) / factorial(n - k) * t ** (n - k)
    return r


def _A(T: float) -> np.ndarray:
    return np.array([_deriv_row(k, 0.0) for k in range(NDER)] + [_deriv_row(k, T) for k in range(NDER)])


def _Q(T: float) -> np.ndarray:
    Q = np.zeros((NCOEF, NCOEF))
    for i in range(4, NCOEF):
        for j in range(4, NCOEF):
            ci = factorial(i) / factorial(i - 4)
            cj = factorial(j) / factorial(j - 4)
            e = i + j - 7
            Q[i, j] = ci * cj * T ** e / e
    return Q


@dataclass
class PolyTraj:
    coeffs: np.ndarray          # M x 3 x 8
    durations: np.ndarray       # M

    @property
    def total(self) -> float:
        return float(self.durations.sum())

    @property
    def knots(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.durations)]

    def eval(self, t, order: int = 0) -> np.ndarray:
        """Derivative ``order`` at time(s) t -> (N, 3)."""
        t = np.atleast_1d(np.asarray(t, float))
        kn = self.knots
        idx = np.clip(np.searchsorted(kn, t, side="right") - 1, 0, len(self.durations) - 1)
        tau = np.clip(t - kn[idx], 0.0, self.durations[idx])
        out = np.zeros((len(t), 3))
        for n in range(order, NCOEF):
            f = factorial(n) / factorial(n - order)
            out += f * self.coeffs[idx, :, n] * (tau ** (n - order))[:, None]
        return out

    def snap_cost(self) -> float:
        return float(sum(c @ _Q(T) @ c for i, T in enumerate(self.durations) for c in self.coeffs[i]))

    def scaled(self, alpha: float) -> "PolyTraj":
        """Same path, durations x alpha (coefficient n scales by alpha^-n)."""
        s = alpha ** -np.arange(NCOEF)
        return PolyTraj(self.coeffs * s, self.durations * alpha)


def min_snap(waypoints, durations, start_vel=None, end_fixed: bool = False) -> PolyTraj:
    """Waypoints (M+1) x 3, durations M. Starts at rest (or ``start_vel``)."""
    W = np.asarray(waypoints, float)
    T = np.asarray(durations, float)
    M = len(T)
    nk = M + 1
    nD = NDER * nk
    S = np.zeros((2 * NDER * M, nD))
    Hd = np.zeros((2 * NDER * M, 2 * NDER * M))
    Ainv = []
    for i in range(M):
        Ai = np.linalg.inv(_A(T[i]))
        Ainv.append(Ai)
        b = 2 * NDER * i
        Hd[b:b + 8, b:b + 8] = Ai.T @ _Q(T[i]) @ Ai
        for k in range(NDER):
            S[b + k, NDER * i + k] = 1.0
            S[b + NDER + k, NDER * (i + 1) + k] = 1.0
    R = S.T @ Hd @ S
    fixed = [0, 1, 2, 3] + [NDER * k for k in range(1, nk)]
    if end_fixed:
        fixed += [NDER * M + k for k in (1, 2, 3)]
    fixed = sorted(set(fixed))
    free = [j for j in range(nD) if j not in fixed]
    D = np.zeros((nD, 3))
    for k in range(nk):
        D[NDER * k] = W[k]
    if start_vel is not None:
        D[1] = start_vel
    Rpp = R[np.ix_(free, free)]
    Rpf = R[np.ix_(free, fixed)]
    D[free] = -np.linalg.solve(Rpp, Rpf @ D[fixed])
    coeffs = np.zeros((M, 3, NCOEF))
    for i in range(M):
        d = np.r_[D[NDER * i:NDER * i + NDER], D[NDER * (i + 1):NDER * (i + 1) + NDER]]  # 8 x 3
        coeffs[i] = (Ainv[i] @ d).T
    return PolyTraj(coeffs, T.copy())


@dataclass
class Limits:
    mass: float = 0.65
    drag: float = 0.15
    thrust_max: float = 28.0     # N (70 % of 4 x 10 N leaves margin for control)
    thrust_min: float = 1.5      # N
    rate_max: float = np.radians(400.0)
    speed_max: float = 25.0


def dynamics_profile(traj: PolyTraj, lim: Limits, n: int = 600):
    t = np.linspace(0, traj.total, n)
    v, a, j = traj.eval(t, 1), traj.eval(t, 2), traj.eval(t, 3)
    f = lim.mass * (a - G * E3) + lim.drag * v
    T = np.linalg.norm(f, axis=1)
    z = -f / T[:, None]
    fd = lim.mass * j + lim.drag * a
    perp = fd - (fd * z).sum(1, keepdims=True) * z
    w = np.linalg.norm(perp, axis=1) / T
    return t, T, w, np.linalg.norm(v, axis=1)


def feasible(traj: PolyTraj, lim: Limits) -> bool:
    _, T, w, s = dynamics_profile(traj, lim)
    return bool(T.max() <= lim.thrust_max and T.min() >= lim.thrust_min and w.max() <= lim.rate_max
                and s.max() <= lim.speed_max)


def _profile_arrays(v, a, j, lim: Limits, alpha: float):
    va, aa, ja = v / alpha, a / alpha ** 2, j / alpha ** 3
    f = lim.mass * (aa - G * E3) + lim.drag * va
    T = np.linalg.norm(f, axis=1)
    z = -f / T[:, None]
    fd = lim.mass * ja + lim.drag * aa
    perp = fd - (fd * z).sum(1, keepdims=True) * z
    return T, np.linalg.norm(perp, axis=1) / T, np.linalg.norm(va, axis=1)


def min_feasible_scale(traj: PolyTraj, lim: Limits, lo=0.05, hi=20.0, iters=24, n=400) -> float:
    """Smallest alpha such that ``traj.scaled(alpha)`` is feasible (bisection in log alpha)."""
    t = np.linspace(0, traj.total, n)
    v, a, j = traj.eval(t, 1), traj.eval(t, 2), traj.eval(t, 3)

    def ok(al):
        T, w, s = _profile_arrays(v, a, j, lim, al)
        return T.max() <= lim.thrust_max and T.min() >= lim.thrust_min and w.max() <= lim.rate_max             and s.max() <= lim.speed_max

    if not ok(hi):
        return np.inf
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def allocate_times(waypoints, lim: Limits, iters: int = 40, init=None, log=None):
    """Minimise feasibility-scaled lap time over segment-time ratios.

    Returns (trajectory, history of lap times)."""
    W = np.asarray(waypoints, float)
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    x = np.log(np.maximum(seg, 0.3) ** 0.7) if init is None else np.log(init)

    def lap(xv):
        tr = min_snap(W, np.exp(xv))
        a = min_feasible_scale(tr, lim)
        return a * tr.total, tr, a

    f, tr, a = lap(x)
    hist = [f]
    step = 0.3
    h = 1e-3
    for it in range(iters):
        g = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = h
            g[i] = (lap(x + e)[0] - lap(x - e)[0]) / (2 * h)
        g -= g.mean()   # uniform scaling is a null direction (handled by alpha)
        gn = np.linalg.norm(g)
        if gn < 1e-6:
            break
        improved = False
        while step > 1e-3:
            xn = x - step * g / gn
            fn, trn, an = lap(xn)
            if fn < f - 1e-4:
                x, f, tr, a = xn, fn, trn, an
                step *= 1.5
                improved = True
                break
            step *= 0.5
        hist.append(f)
        if log:
            log(f"time allocation iter {it}: lap {f:.3f} s")
        if not improved:
            break
    return tr.scaled(a), hist
