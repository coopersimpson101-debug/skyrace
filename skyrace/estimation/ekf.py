"""Error-state EKF: IMU propagation + gate-relative PnP pose updates.

Nominal state  x = (p, v, R, b_g, b_a)    p, v in NED world, R body->world
Error state   dx = (dp, dv, dtheta, db_g, db_a)  in R^15, with R = R_hat exp([dtheta]x)

IMU model      w_m = w + b_g + n_g,    f_m = R^T (a - g e3) + b_a + n_a
Propagation    a = R (f_m - b_a) + g e3,  p += v dt + a dt^2 / 2,  v += a dt,
               R <- R exp((w_m - b_g) dt)
Error dynamics (continuous):
    dp'     = dv
    dv'     = -R [f]x dtheta - R db_a - R n_a
    dtheta' = -[w]x dtheta - db_g - n_g
    db'     = random walk
Discretised to first order: F = I + A dt, Q_d = G Q_c G^T dt.

Measurement (one PnP solve of a known gate) z = (p_m, R_m):
    r = [p_m - p ; log(R^T R_m)],  H = [[I 0 0 0 0], [0 0 I 0 0]]
with the 6x6 covariance propagated from the PnP Jacobian
(:func:`skyrace.vision.pnp.camera_pose_to_body`). A chi-square gate (6 dof)
rejects outliers (wrong association, planar flip ambiguity).

This is deliberately lighter than insight-vio's MSCKF: the gates are a known
map, so a single PnP gives an absolute 6-DoF fix and no feature states are
needed. See README "Roadmap" for fusing MSCKF odometry between gates.
"""
from __future__ import annotations

import numpy as np

from ..so3 import exp_so3, log_so3, skew

G = 9.80665
E3 = np.array([0.0, 0.0, 1.0])


class GateEkf:
    def __init__(self, p0, R0, gyro_nd=0.004 / np.sqrt(400) * 4, accel_nd=0.08 / np.sqrt(400) * 4,
                 gyro_rw=2e-4, accel_rw=2e-3, chi2_gate=22.46):
        self.p = np.array(p0, float)
        self.v = np.zeros(3)
        self.R = np.array(R0, float)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.diag([0.05 ** 2] * 3 + [0.05 ** 2] * 3 + [np.radians(2) ** 2] * 3 +
                         [0.01 ** 2] * 3 + [0.15 ** 2] * 3)
        self.qg, self.qa, self.qbg, self.qba = gyro_nd ** 2, accel_nd ** 2, gyro_rw ** 2, accel_rw ** 2
        self.chi2_gate = chi2_gate
        self.n_updates = 0
        self.n_rejected = 0
        self.last_w = np.zeros(3)
        self.last_a = np.zeros(3)
        self.t = 0.0

    # ------------------------------------------------------------------ IMU
    def propagate(self, gyro, accel, dt):
        w = np.asarray(gyro) - self.bg
        f = np.asarray(accel) - self.ba
        R = self.R
        a = R @ f + G * E3
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.R = R @ exp_so3(w * dt)
        self.last_w, self.last_a = w, a
        A = np.zeros((15, 15))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 6:9] = -R @ skew(f)
        A[3:6, 12:15] = -R
        A[6:9, 6:9] = -skew(w)
        A[6:9, 9:12] = -np.eye(3)
        F = np.eye(15) + A * dt
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = self.qa * dt * np.eye(3)
        Q[6:9, 6:9] = self.qg * dt * np.eye(3)
        Q[9:12, 9:12] = self.qbg * dt * np.eye(3)
        Q[12:15, 12:15] = self.qba * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q
        self.t += dt

    # -------------------------------------------------------------- updates
    def _update(self, r, H, Rm):
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        d2 = float(r @ Si @ r)
        if d2 > self.chi2_gate:
            self.n_rejected += 1
            return False, d2
        K = self.P @ H.T @ Si
        dx = K @ r
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ exp_so3(dx[6:9])
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        self.n_updates += 1
        return True, d2

    def update_pose(self, p_m, R_m, cov):
        r = np.r_[np.asarray(p_m) - self.p, log_so3(self.R.T @ R_m)]
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        return self._update(r, H, cov + 1e-6 * np.eye(6))

    def update_corners(self, uv, P_w, R_bc, K, sigma_px=2.0):
        """Tightly-coupled update with the 4 (or more) corner pixels of a known gate.

        h(x) = pi(K, R_bc^T R^T (P_w - p)); with R = R_hat exp([dtheta]x):
            dX_c/ddp     = -R_bc^T R_hat^T
            dX_c/ddtheta =  R_bc^T [X_b]x,   X_b = R_hat^T (P_w - p)
        This avoids the planar-PnP rotation/translation ambiguity at long
        range (the PnP solution is still used to validate the association).
        """
        uv = np.asarray(uv, float)
        n = len(uv)
        H = np.zeros((2 * n, 15))
        r = np.zeros(2 * n)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for i in range(n):
            Xb = self.R.T @ (P_w[i] - self.p)
            Xc = R_bc.T @ Xb
            x, y, z = Xc
            if z < 0.2:
                return False, np.inf
            r[2 * i:2 * i + 2] = uv[i] - np.array([fx * x / z + cx, fy * y / z + cy])
            Jp = np.array([[fx / z, 0, -fx * x / z ** 2], [0, fy / z, -fy * y / z ** 2]])
            H[2 * i:2 * i + 2, 0:3] = Jp @ (-R_bc.T @ self.R.T)
            H[2 * i:2 * i + 2, 6:9] = Jp @ (R_bc.T @ skew(Xb))
        gate = self.chi2_gate
        self.chi2_gate = {8: 26.12}.get(2 * n, 3.0 * 2 * n)  # chi2(8) 99.9 %
        try:
            return self._update(r, H, sigma_px ** 2 * np.eye(2 * n))
        finally:
            self.chi2_gate = gate

    def update_zero_velocity(self, sigma=0.02):
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return self._update(-self.v, H, sigma ** 2 * np.eye(3))

    # -------------------------------------------------------------- queries
    @property
    def pos_std(self) -> float:
        return float(np.sqrt(np.trace(self.P[0:3, 0:3])))

    def state(self):
        return dict(p=self.p.copy(), v=self.v.copy(), R=self.R.copy(), w=self.last_w.copy(),
                    bg=self.bg.copy(), ba=self.ba.copy())
