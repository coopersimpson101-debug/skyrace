import math

import numpy as np

from skyrace.fcport import crsf
from skyrace.fcport.fc import FcConfig, rc_from_crsf
from skyrace.link.commands import AirframeModel, CommandEncoder


def test_rates_and_thrust_roundtrip_through_channels():
    enc = CommandEncoder(FcConfig())
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.uniform(-math.radians(600), math.radians(600), 3)
        T = rng.uniform(1.0, 35.0)
        d = enc.decode(enc.channels(w, T, arm=True))
        # 11-bit quantisation: < 1.5 deg/s near full deflection, far less near centre
        assert np.all(np.abs(np.degrees(np.array(d["rates"]) - w)) < 1.5)
        assert abs(d["thrust"] - T) < 0.06
        assert d["arm"] and not d["angle"]


def test_fc_sees_same_sticks():
    cfg = FcConfig()
    enc = CommandEncoder(cfg)
    f = enc.frame((0.0, 0.0, 0.0), 0.0, arm=True, angle_mode=True, throttle=0.25)
    p = crsf.Parser()
    assert p.feed_bytes(f, 0) == [crsf.TYPE_RC_CHANNELS]
    rc = rc_from_crsf(cfg, p.channels)
    assert rc.arm_switch and rc.angle_mode and abs(rc.throttle - 0.25) < 1e-3
    assert rc.roll == 0.0 and rc.pitch == 0.0 and rc.yaw == 0.0


def test_thrust_model_inverse():
    a = AirframeModel()
    for thr in np.linspace(0, 1, 11):
        assert abs(a.thrust_to_throttle(a.throttle_to_thrust(thr)) - thr) < 1e-9
