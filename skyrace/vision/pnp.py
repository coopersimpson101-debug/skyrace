"""Perspective-n-Point for a planar gate, written from scratch.

1. Normalised DLT homography (Hartley) between gate-plane coordinates (y, z)
   and normalised image coordinates.
2. Decompose H ~ [r2 r3 t] (gate y and z axes and the translation), complete
   r1 = r2 x r3 and project onto SO(3) with an SVD (Zhang 2000).
3. Levenberg-Marquardt on the 6-DoF reprojection error with the analytic
   Jacobian, left-perturbing R <- exp(dtheta) R.
4. Covariance = sigma_px^2 (J^T J)^-1 of (dtheta, dt) at the optimum.

``camera_pose_to_body`` turns the result into the drone's world pose and maps
the covariance onto the estimator's error state (position, body-frame
rotation), see docs/estimation.md for the Jacobian.
"""
from __future__ import annotations

import numpy as np

from ..so3 import exp_so3, skew


def _normalise(pts):
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T


def homography_dlt(src, dst):
    """H with dst ~ H src (both Nx2, N >= 4)."""
    Ts, Td = _normalise(src), _normalise(dst)
    s = (np.c_[src, np.ones(len(src))]) @ Ts.T
    d = (np.c_[dst, np.ones(len(dst))]) @ Td.T
    A = []
    for (x, y, w), (u, v, q) in zip(s, d):
        A.append([0, 0, 0, -q * x, -q * y, -q * w, v * x, v * y, v * w])
        A.append([q * x, q * y, q * w, 0, 0, 0, -u * x, -u * y, -u * w])
    _, _, Vt = np.linalg.svd(np.array(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def project(K, R, t, X):
    Xc = X @ R.T + t
    uv = (Xc[:, :2] / Xc[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, Xc


def _jacobian(K, R, t, X):
    Xc = X @ R.T + t
    fx, fy = K[0, 0], K[1, 1]
    J = np.zeros((2 * len(X), 6))
    for i, (x, y, z) in enumerate(Xc):
        Jp = np.array([[fx / z, 0, -fx * x / z ** 2], [0, fy / z, -fy * y / z ** 2]])
        J[2 * i:2 * i + 2, :3] = Jp @ (-skew(Xc[i] - t))   # d(R X)/d(dtheta) = -[R X]x
        J[2 * i:2 * i + 2, 3:] = Jp
    return J


def solve_pnp_planar(X, uv, K, sigma_px: float = 1.5, iters: int = 15, init=None):
    """X: Nx3 gate-frame points on the x = 0 plane. Returns dict or None.

    Keys: R (gate->camera), t, rms (px), cov (6x6 over [dtheta, dt]), ok.
    """
    X = np.asarray(X, float)
    uv = np.asarray(uv, float)
    if len(X) < 4:
        return None
    Kinv = np.linalg.inv(K)
    xn = (np.c_[uv, np.ones(len(uv))] @ Kinv.T)[:, :2]
    if init is None:
        H = homography_dlt(X[:, 1:3], xn)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lam = 2.0 / (np.linalg.norm(h1) + np.linalg.norm(h2))
        if h3[2] * lam < 0:
            lam = -lam
        r2, r3, t = lam * h1, lam * h2, lam * h3
        r1 = np.cross(r2, r3)
        U, _, Vt = np.linalg.svd(np.c_[r1, r2, r3])
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    else:
        R, t = init
    mu = 1e-3
    uv_hat, _ = project(K, R, t, X)
    err = (uv_hat - uv).ravel()
    cost = err @ err
    for _ in range(iters):
        J = _jacobian(K, R, t, X)
        A = J.T @ J
        g = J.T @ err
        dx = -np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-9), g)
        Rn = exp_so3(dx[:3]) @ R
        tn = t + dx[3:]
        uv_n, Xc = project(K, Rn, tn, X)
        if np.any(Xc[:, 2] <= 0):
            mu *= 10
            continue
        en = (uv_n - uv).ravel()
        cn = en @ en
        if cn < cost:
            R, t, err, cost = Rn, tn, en, cn
            mu = max(mu * 0.3, 1e-7)
            if np.linalg.norm(dx) < 1e-9:
                break
        else:
            mu *= 10
    J = _jacobian(K, R, t, X)
    try:
        cov = sigma_px ** 2 * np.linalg.inv(J.T @ J + 1e-12 * np.eye(6))
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt(cost / len(X)))
    _, Xc = project(K, R, t, X)
    return dict(R=R, t=t, rms=rms, cov=cov, ok=bool(np.all(Xc[:, 2] > 0)))


def camera_pose_to_body(sol, gate, R_bc):
    """PnP solution -> drone pose in world (p_wb, R_wb) and 6x6 cov of [dp, dphi_body]."""
    R_cg, t = sol["R"], sol["t"]
    R_wg = gate.R
    R_wc = R_wg @ R_cg.T
    p = gate.center + R_wg @ (-R_cg.T @ t)
    R_wb = R_wc @ R_bc.T
    J = np.zeros((6, 6))
    J[:3, :3] = -R_wc @ skew(t)
    J[:3, 3:] = -R_wc
    J[3:, :3] = -R_bc
    cov = J @ sol["cov"] @ J.T
    return p, R_wb, cov
