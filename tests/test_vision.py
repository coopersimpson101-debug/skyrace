import os

import numpy as np
import pytest

from skyrace.sim.camera import CameraModel, Renderer, Scene
from skyrace.sim.track import Gate
from skyrace.so3 import euler_to_rot, exp_so3, log_so3
from skyrace.vision import classical
from skyrace.vision.cnn import backward, focal_loss_grad, forward, init_params
from skyrace.vision.detector import DEFAULT_WEIGHTS, CornerDetector, measure_gate
from skyrace.vision.pnp import camera_pose_to_body, homography_dlt, project, solve_pnp_planar


def _scene(dist=8.0):
    gate = Gate(np.array([dist, 0.5, -2.0]), np.radians(10))
    R = euler_to_rot(np.radians(5), np.radians(-12), np.radians(4))
    p = np.array([0.0, 0.0, -1.7])
    return gate, R, p


def test_homography_exact():
    rng = np.random.default_rng(0)
    H = np.array([[1.2, 0.1, 3], [-0.2, 0.9, 1], [1e-3, 2e-3, 1]])
    src = rng.uniform(-1, 1, (6, 2))
    d = np.c_[src, np.ones(6)] @ H.T
    dst = d[:, :2] / d[:, 2:]
    Hh = homography_dlt(src, dst)
    assert np.allclose(Hh / Hh[2, 2], H / H[2, 2], atol=1e-8)


def test_pnp_recovers_pose_and_body_pose():
    cam = CameraModel()
    gate, R_wb, p_wb = _scene()
    mir = False
    X = gate.inner_corners_local(mir)
    uv, _ = cam.project(R_wb, p_wb, gate.inner_corners_world(mir))
    sol = solve_pnp_planar(X, uv, cam.K)
    assert sol["rms"] < 1e-6
    p, R, cov = camera_pose_to_body(sol, gate, cam.R_bc)
    assert np.linalg.norm(p - p_wb) < 1e-6
    assert np.linalg.norm(log_so3(R_wb.T @ R)) < 1e-6
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_pnp_noise_and_covariance_consistent():
    """Monte-Carlo spread of the body position matches the propagated covariance (within 35 %).

    At 3 m the problem is close to linear; at long range a planar 4-point PnP is dominated by the
    rotation/translation ambiguity and the linearised covariance becomes optimistic/pessimistic
    per axis - the EKF's chi-square gate and its tight IMU attitude prior handle that regime."""
    cam = CameraModel()
    gate, R_wb, p_wb = _scene(3.0)
    X = gate.inner_corners_local()
    uv0, _ = cam.project(R_wb, p_wb, gate.inner_corners_world())
    rng = np.random.default_rng(1)
    ps, covs = [], []
    for _ in range(300):
        sol = solve_pnp_planar(X, uv0 + rng.normal(0, 1.0, uv0.shape), cam.K, sigma_px=1.0)
        p, R, cov = camera_pose_to_body(sol, gate, cam.R_bc)
        ps.append(p)
        covs.append(cov)
    emp = np.cov(np.array(ps).T)
    pred = np.mean(covs, 0)[:3, :3]
    ratio = np.trace(emp) / np.trace(pred)
    assert 0.65 < ratio < 1.35, ratio


def test_render_labels_and_classical_baseline():
    cam = CameraModel()
    r = Renderer(cam, Scene(seed=3, distractors=0))
    gate, R, p = _scene()
    img = r.render(R, p, [gate])
    uv, vis, mir, z = r.corner_labels(R, p, [gate])[0]
    assert vis.all() and not mir
    gates = classical.detect_gates(img)
    assert len(gates) == 1
    assert np.abs(gates[0] - uv).max() < 1.5


def test_cnn_backprop_matches_finite_differences():
    P = {k: v.astype(np.float64) for k, v in init_params(1).items()}
    rng = np.random.default_rng(0)
    x = rng.random((3, 2, 16, 24))
    tgt = np.zeros((4, 2, 4, 6))
    tgt[1, 0, 2, 3] = 1.0
    tgt[0, 1, 1, 1] = 0.6

    def loss(P):
        return focal_loss_grad(forward(P, x), tgt)[0]

    lg, cache = forward(P, x, cache=True)
    _, dl = focal_loss_grad(lg, tgt)
    g = backward(P, cache, dl.astype(np.float64))
    for k in ["c1.w", "c2.w", "c4.w", "c5.b", "c6.w", "c7.w", "head.w", "head.b"]:
        idx = tuple(int(rng.integers(0, s)) for s in P[k].shape)
        e = 1e-5
        P[k][idx] += e
        lp = loss(P)
        P[k][idx] -= 2 * e
        lm = loss(P)
        P[k][idx] += e
        num = (lp - lm) / (2 * e)
        assert abs(num - g[k][idx]) <= 1e-6 + 1e-4 * abs(num), (k, num, g[k][idx])


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS), reason="trained weights not present")
def test_trained_cnn_measures_gate():
    cam = CameraModel()
    r = Renderer(cam, Scene(seed=7))          # a texture the network was not trained on
    gate, R, p = _scene(4.5)
    img = r.render(R, p, [gate])
    img = (img * 255 + 0.5).astype(np.uint8).astype(np.float32) / 255
    det = CornerDetector("cnn")
    cands = det(img)
    R_prior = R @ exp_so3(np.radians([1.0, -1.0, 1.5]))
    m = measure_gate(cands, cam, gate, R_prior, p + np.array([0.15, -0.1, 0.05]))
    assert m is not None
    assert np.linalg.norm(m["p"] - p) < 0.25
