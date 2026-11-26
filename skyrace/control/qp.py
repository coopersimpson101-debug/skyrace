"""A small dense ADMM QP solver (the OSQP iteration, written from scratch).

    minimise   1/2 x^T P x + q^T x
    subject to l <= A x <= u

Iteration (Stellato et al., "OSQP: an operator splitting solver for quadratic
programs", Math. Prog. Comp. 2020), with relaxation alpha and step rho:

    x~   = (P + sigma I + rho A^T A)^-1 (sigma x - q + A^T (rho z - y))
    z~   = A x~
    x    = alpha x~ + (1 - alpha) x
    z+   = clip(alpha z~ + (1 - alpha) z + y / rho, l, u)
    y    = y + rho (alpha z~ + (1 - alpha) z - z+)

Stops when ||A x - z||_inf <= eps_abs + eps_rel max(||Ax||, ||z||) and
||P x + q + A^T y||_inf <= eps_abs + eps_rel max(||Px||, ||A^T y||, ||q||).
rho is adapted every ``adapt_every`` iterations by sqrt(primal / dual
residual ratio) and the (small, dense) linear system is re-factorised.
Supports warm starts, which the MPC uses every control step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QpResult:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    iters: int
    converged: bool
    prim_res: float
    dual_res: float
    obj: float


def solve_qp(P, q, A, l, u, x0=None, y0=None, rho=0.1, sigma=1e-6, alpha=1.6,
             eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, adapt_every=25) -> QpResult:
    P = np.asarray(P, float)
    q = np.asarray(q, float)
    A = np.asarray(A, float)
    n, m = P.shape[0], A.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    y = np.zeros(m) if y0 is None else np.array(y0, float)
    z = np.clip(A @ x, l, u)
    AtA = A.T @ A
    eye = np.eye(n)

    def factor(r):
        return np.linalg.inv(P + sigma * eye + r * AtA)

    Kinv = factor(rho)
    pr = dr = np.inf
    k = 0
    conv = False
    for k in range(1, max_iter + 1):
        xt = Kinv @ (sigma * x - q + A.T @ (rho * z - y))
        zt = A @ xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        zn = np.clip(zr + y / rho, l, u)
        y = y + rho * (zr - zn)
        z = zn
        if k % 5 == 0 or k == max_iter:
            Ax = A @ x
            Px = P @ x
            Aty = A.T @ y
            pr = np.max(np.abs(Ax - z)) if m else 0.0
            dr = np.max(np.abs(Px + q + Aty))
            ep = eps_abs + eps_rel * max(np.max(np.abs(Ax)) if m else 0.0, np.max(np.abs(z)) if m else 0.0)
            ed = eps_abs + eps_rel * max(np.max(np.abs(Px)), np.max(np.abs(Aty)) if m else 0.0,
                                         np.max(np.abs(q)))
            if pr <= ep and dr <= ed:
                conv = True
                break
            if adapt_every and k % adapt_every == 0 and dr > 0 and pr > 0:
                ratio = np.sqrt((pr / max(ep, 1e-12)) / (dr / max(ed, 1e-12)))
                if ratio > 5 or ratio < 0.2:
                    rho = float(np.clip(rho * ratio, 1e-6, 1e6))
                    Kinv = factor(rho)
    obj = float(0.5 * x @ P @ x + q @ x)
    return QpResult(x, y, z, k, conv, float(pr), float(dr), obj)
