import threading

import numpy as np
import pytest

from conftest import needs_sitl
from skyrace.companion import CompanionStack, plan_race
from skyrace.link.hil import HilServer, run_client
from skyrace.link.safety import SafetyConfig
from skyrace.planning.minsnap import Limits
from skyrace.race import RaceSession, run_race
from skyrace.sim.camera import CameraModel, Renderer
from skyrace.sim.quad import QuadSim
from skyrace.sim.track import make_track

LIM = Limits(thrust_max=18.0)


@pytest.fixture(scope="module")
def setup():
    tr = make_track("loop7")
    traj, _ = plan_race(tr, LIM)
    return tr, traj, CameraModel()


def _stack(setup, **kw):
    tr, traj, cam = setup
    kw.setdefault("vision", "truth")
    return CompanionStack(tr, cam, race_traj=traj, lim=LIM, **kw)


def test_full_lap_python_backend_truth_vision(setup):
    tr, traj, cam = setup
    res, log, _, _ = run_race(QuadSim(), _stack(setup, controller="minsnap"), Renderer(cam), tr, keep_frames=False)
    assert res["pass_rate"] == 1.0 and not res["crashed"]
    assert abs(res["lap_time"] - traj.total) < 0.2


@needs_sitl
def test_full_lap_real_skyforge_c_truth_vision(setup):
    from skyrace.sitl import SkyforgeSITL
    tr, traj, cam = setup
    res, *_ = run_race(SkyforgeSITL(), _stack(setup, controller="mpc"), Renderer(cam), tr, keep_frames=False)
    assert res["pass_rate"] == 1.0 and not res["crashed"]


def _fly_until(sess, stack, t_stop, hook=None):
    stack.truth_fn = sess.truth_pose
    while sess.t < t_stop:
        t, imu, img, rx = sess.sense()
        if hook:
            rx = hook(t, rx)
        sess.actuate(stack.tick(t, imu, img, rx))
        sess.advance()


def test_operator_kill_disarms_fc_within_two_frames(setup):
    tr, traj, cam = setup
    sim = QuadSim()
    sess = RaceSession(sim, Renderer(cam), tr)
    stack = _stack(setup, controller="minsnap")
    _fly_until(sess, stack, 4.0)
    assert sim.fc.armed and stack.phase == "TAKEOFF" and sim.pos[2] < -0.3
    stack.safety.operator_kill = True
    t_kill = sess.t
    while sim.fc.armed:
        _fly_until(sess, stack, sess.t + 0.005)
    assert sess.t - t_kill <= 0.025      # 10 ms command delay + one 5 ms frame + PID step
    _fly_until(sess, stack, sess.t + 1.0)
    assert not sim.fc.armed and stack.safety.state == "KILLED"
    assert stack.safety.events[-1][2] == "operator kill switch"


def test_geofence_kill(setup):
    tr, traj, cam = setup
    sim = QuadSim()
    sess = RaceSession(sim, Renderer(cam), tr)
    stack = _stack(setup, controller="minsnap", safety_cfg=SafetyConfig(geofence_max=(35.0, 35.0, 0.5),
                                                                        geofence_min=(-15.0, -12.0, -1.0)))
    _fly_until(sess, stack, 6.0)     # takeoff climbs to 1.5 m > 1.0 m ceiling
    assert stack.safety.state == "KILLED" and stack.safety.reason == "geofence"
    assert not sim.fc.armed


def test_telemetry_loss_kill(setup):
    tr, traj, cam = setup
    sim = QuadSim()
    sess = RaceSession(sim, Renderer(cam), tr)
    stack = _stack(setup, controller="minsnap")
    _fly_until(sess, stack, 3.4, hook=lambda t, rx: b"" if t > 2.5 else rx)
    assert stack.safety.state == "KILLED" and "telemetry" in stack.safety.reason
    assert not sim.fc.armed


def test_hil_over_tcp_matches_in_process(setup):
    """Same seed, same bytes: the HIL run (two sockets, lockstep) reproduces the in-process run."""
    tr, traj, cam = setup
    t_max = 4.5
    sim_a = QuadSim()
    sess_a = RaceSession(sim_a, Renderer(cam), tr)
    stack_a = _stack(setup, controller="minsnap", vision="classical")
    _fly_until(sess_a, stack_a, t_max)

    server = HilServer(QuadSim(), Renderer(cam), tr, port=0, t_max=t_max)
    out = {}
    th = threading.Thread(target=lambda: out.update(r=server.serve_one()))
    th.start()
    run_client(_stack(setup, controller="minsnap", vision="classical"), port=server.port, sens_port=server.sens_port)
    th.join(30)
    server.close()
    res_b, sess_b = out["r"]
    pa = np.array(sess_a.truth["p"])
    pb = np.array(sess_b.truth["p"])
    n = min(len(pa), len(pb))
    assert n > 800
    assert np.max(np.abs(pa[:n] - pb[:n])) < 1e-9
    assert res_b["uart_frames"] >= n - 2 and res_b["uart_bytes"] == 26 * res_b["uart_frames"]
