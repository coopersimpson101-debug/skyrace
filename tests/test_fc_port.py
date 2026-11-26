import ctypes

import numpy as np

from conftest import needs_sitl
from skyrace.fcport import fc as F
from skyrace.link.commands import CommandEncoder
from skyrace.sim.quad import QuadSim


def test_rate_curve_inverse():
    for r in np.linspace(-660, 660, 89):
        s = F.rc_rate_curve_inverse(r, 200, 670, 0.54)
        assert abs(F.rc_rate_curve(s, 200, 670, 0.54) - r) < 1e-6
    assert F.rc_rate_curve_inverse(5000, 200, 670, 0.54) == 1.0


def test_mixer_airmode_desaturates():
    out, sat = F.mixer_mix(0.0, 0.4, 0.0, 0.0)
    assert sat and min(out) >= 0.0 and max(out) - min(out) > 0.39   # authority kept at zero throttle
    out, sat = F.mixer_mix(0.5, 0.0, 0.0, 0.0)
    assert not sat and out == [0.5] * 4


def _frames(sim, enc, n, **kw):
    for _ in range(n):
        sim.write_uart(enc.frame((0, 0, 0), 0.0, **kw))
        sim.step(0.005)


def test_arming_needs_switch_edge_and_low_throttle():
    sim = QuadSim()
    enc = CommandEncoder(sim.cfg)
    _frames(sim, enc, 60, arm=True, throttle=0.0)          # switch already ON at boot
    assert not sim.fc.armed                                 # must be seen OFF first
    _frames(sim, enc, 300, arm=False, throttle=0.0)        # gyro calibration
    _frames(sim, enc, 10, arm=True, throttle=0.3)          # throttle too high
    assert not sim.fc.armed and sim.fc.arm_block & F.ARM_BLOCK_THROTTLE
    _frames(sim, enc, 10, arm=False, throttle=0.0)
    _frames(sim, enc, 10, arm=True, throttle=0.0)
    assert sim.fc.armed
    _frames(sim, enc, 2, arm=False, throttle=0.0)          # kill / pilot disarm wins at once
    assert not sim.fc.armed


def test_failsafe_lands_then_disarms():
    sim = QuadSim()
    enc = CommandEncoder(sim.cfg)
    _frames(sim, enc, 300, arm=False, throttle=0.0)
    _frames(sim, enc, 10, arm=True, throttle=0.0)
    for _ in range(300):                                    # climb a bit
        sim.write_uart(enc.frame((0, 0, 0), 0.65 * 9.81 * 1.2, arm=True))
        sim.step(0.005)
    assert sim.fc.armed and sim.pos[2] < -0.5
    sim.step(0.35)                                          # frames stop: > fs_timeout_ms = 300
    assert sim.fc.fs == F.FS_LANDING and sim.fc.armed
    sim.step(5.2)                                           # fs_land_time_s = 5
    assert sim.fc.fs == F.FS_DISARMED and not sim.fc.armed


@needs_sitl
def test_pid_rate_curve_mixer_match_c(sitl_lib):
    f = ctypes.c_float
    rng = np.random.default_rng(2)
    for _ in range(50):
        s, c, m, e = rng.uniform(-1, 1), rng.uniform(100, 300), rng.uniform(300, 900), rng.uniform(0, 1)
        assert abs(sitl_lib.sitl_rate_curve(s, c, m, e) - F.rc_rate_curve(s, c, m, e)) < 1e-2
        thr, r, p, y = rng.uniform(0, 1), *rng.uniform(-1, 1, 3)
        out = (f * 4)()
        sat = sitl_lib.sitl_mix(thr, r, p, y, 1, out)
        po, ps = F.mixer_mix(thr, r, p, y, True)
        assert np.allclose(list(out), po, atol=1e-5) and bool(sat) == ps
        mo, idle = rng.uniform(0, 1), 0.055
        assert abs(sitl_lib.sitl_dshot_value(mo, idle) - F.dshot_value_from_motor(mo, idle)) <= 1
    n = 400
    sp = rng.uniform(-300, 300, n).astype(np.float32)
    me = (sp + rng.normal(0, 30, n)).astype(np.float32)
    out = (f * n)()
    sitl_lib.sitl_pid_run(sp.ctypes.data_as(ctypes.POINTER(f)), me.ctypes.data_as(ctypes.POINTER(f)), n,
                          55, 50, 45, 100, 4000.0, out)
    a = F.PidAxis(55, 50, 45, 100, 4000.0, 100.0, 40.0)
    py = [a.update(float(sp[i]), float(me[i]), 1 / 4000.0, 1.0, False) for i in range(n)]
    assert np.allclose(list(out), py, atol=1e-4)
