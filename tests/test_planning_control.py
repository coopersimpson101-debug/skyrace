import numpy as np

from skyrace.control.controllers import MpcTracker
from skyrace.control.qp import solve_qp
from skyrace.estimation.ekf import GateEkf
from skyrace.planning.minsnap import Limits, allocate_times, dynamics_profile, min_snap
from skyrace.planning.reference import Reference
from skyrace.so3 import euler_to_rot, exp_so3, log_so3

W = np.array([[0, 0, -1.5], [6, 1, -2.0], [10, 6, -2.5], [8, 12, -2.0], [2, 14, -1.5]], float)


def test_min_snap_interpolates_and_is_smooth():
    T = np.array([1.5, 1.2, 1.6, 1.3])
    tr = min_snap(W, T)
    kn = tr.knots
    assert np.allclose(tr.eval(kn, 0), W, atol=1e-8)
    assert np.allclose(tr.eval(0.0, 1), 0) and np.allclose(tr.eval(0.0, 2), 0)
    for k in kn[1:-1]:
        for order in range(1, 5):   # C^4 across interior knots (optimum is C^6)
            a, b = tr.eval(k - 1e-7, order), tr.eval(k + 1e-7, order)
            assert np.allclose(a, b, atol=1e-3 * (1 + np.abs(a).max())), order


def test_min_snap_is_optimal():
    """Perturbing the free derivatives can only increase the snap cost."""
    T = np.array([1.5, 1.2, 1.6, 1.3])
    tr = min_snap(W, T)
    J = tr.snap_cost()
    rng = np.random.default_rng(0)
    from skyrace.planning import minsnap as ms
    for _ in range(5):
        c = tr.coeffs.copy()
        # add a polynomial bump that keeps positions at the knots (zero at both ends of a segment)
        i = int(rng.integers(len(T)))
        tau = np.linspace(0, T[i], 9)
        bump = np.polyfit(tau, rng.normal(0, 0.05, 9) * np.sin(np.pi * tau / T[i]), 7)[::-1]
        c[i, 0] += bump
        J2 = ms.PolyTraj(c, T).snap_cost()
        assert J2 >= J - 1e-9


def test_time_allocation_reduces_lap_time_and_is_feasible():
    lim = Limits(thrust_max=20.0)
    tr, hist = allocate_times(W, lim, iters=10)
    assert hist[-1] <= hist[0] + 1e-9
    _, T, w, s = dynamics_profile(tr, lim, n=2000)
    assert T.max() <= lim.thrust_max * 1.01 and w.max() <= lim.rate_max * 1.01
    assert abs(tr.total - hist[-1]) < 1e-6


def test_admm_qp_matches_active_set_solution():
    rng = np.random.default_rng(3)
    n = 12
    M = rng.normal(size=(n, n))
    P = M @ M.T + 0.5 * np.eye(n)
    q = rng.normal(size=n) * 3
    A = np.eye(n)
    l, u = -0.4 * np.ones(n), 0.4 * np.ones(n)
    res = solve_qp(P, q, A, l, u, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000)
    assert res.converged
    x = res.x
    # KKT for a box QP: g = P x + q; g_i = 0 if strictly inside, g_i <= 0 at upper, >= 0 at lower
    g = P @ x + q
    for i in range(n):
        if x[i] > u[i] - 1e-6:
            assert g[i] <= 1e-5
        elif x[i] < l[i] + 1e-6:
            assert g[i] >= -1e-5
        else:
            assert abs(g[i]) < 1e-5
    # general constraints (rows of A not identity)
    A2 = rng.normal(size=(5, n))
    res2 = solve_qp(P, q, A2, -np.ones(5), np.ones(5), eps_abs=1e-8, eps_rel=1e-8, max_iter=20000)
    assert res2.converged and np.all(A2 @ res2.x <= 1 + 1e-5) and np.all(A2 @ res2.x >= -1 - 1e-5)


def test_mpc_corrects_offset_on_hover_reference():
    tr = min_snap(np.array([[0, 0, -1.5], [0.001, 0, -1.5]]), [3.0], end_fixed=True)
    ref = Reference(tr, hold_yaw=True)
    mpc = MpcTracker(ref, latency=0.0)
    # integrate the model the MPC itself uses (rigid body, rates tracked instantly)
    p = np.array([0.6, -0.4, -1.0])
    v = np.zeros(3)
    R = euler_to_rot(0.1, -0.1, 0.2)
    dt = 0.01
    for k in range(200):
        T, w = mpc.command(k * dt, dict(p=p, v=v, R=R, w=np.zeros(3), a=np.zeros(3)))
        a = -T / 0.65 * R[:, 2] + np.array([0, 0, 9.80665]) - 0.15 / 0.65 * v
        p, v, R = p + v * dt, v + a * dt, R @ exp_so3(w * dt)
    assert np.linalg.norm(p - np.array([0, 0, -1.5])) < 0.05
    assert np.linalg.norm(v) < 0.05
    assert all(c for _, c in mpc.solve_stats[-20:])


def test_ekf_converges_with_pose_fixes_and_rejects_outliers():
    rng = np.random.default_rng(0)
    p_true = np.array([2.0, 1.0, -2.0])
    R_true = euler_to_rot(0.0, 0.0, 0.5)
    e = GateEkf(p_true + np.array([0.5, -0.4, 0.3]), R_true @ exp_so3([0.05, -0.03, 0.08]))
    e.P[:3, :3] = np.eye(3) * 0.5 ** 2
    e.P[6:9, 6:9] = np.eye(3) * 0.1 ** 2
    f_body = R_true.T @ np.array([0, 0, -9.80665])
    cov = np.diag([0.03 ** 2] * 3 + [np.radians(1) ** 2] * 3)
    for k in range(400):
        e.propagate(rng.normal(0, 0.004, 3), f_body + rng.normal(0, 0.08, 3), 1 / 400)
        if k % 13 == 0:
            e.update_pose(p_true + rng.normal(0, 0.03, 3), R_true @ exp_so3(rng.normal(0, 0.017, 3)), cov)
    assert np.linalg.norm(e.p - p_true) < 0.06
    assert np.linalg.norm(log_so3(e.R.T @ R_true)) < np.radians(2)
    ok, d2 = e.update_pose(p_true + np.array([3.0, 0, 0]), R_true, cov)
    assert not ok and e.n_rejected == 1
