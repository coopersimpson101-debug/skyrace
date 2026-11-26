"""Body-rate + collective-thrust commands -> CRSF RC channels that skyforge-fc understands.

skyforge-fc in ACRO mode turns sticks into rate setpoints with
``rc_rate_curve`` (after a deadband) and passes the throttle stick straight to
the mixer. So the companion computer inverts that chain:

* rate (deg/s)  --inverse rate curve-->  stick  --undo deadband-->  channel
  (pitch is negated: forward stick = nose down = negative pitch rate)
* thrust (N)    --inverse motor/thrust map-->  throttle  -->  channel, using
  ``T_total = 4 T_max (idle + (1 - idle) thr)^2`` (motor law of the sim / the
  DShot mapping in skyforge-fc; on a real quad identify T_max on a thrust stand)
* AUX1 (ch 4) = arm switch (> 1700 us = armed), AUX2 (ch 5) = angle mode (> 1300 us)

Channel order is AETR: 0 roll, 1 pitch, 2 throttle, 3 yaw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..fcport import crsf
from ..fcport.fc import FcConfig, rc_rate_curve, rc_rate_curve_inverse

ARM_ON_US = 2000.0
ARM_OFF_US = 1000.0
MODE_ACRO_US = 1000.0
MODE_ANGLE_US = 2000.0


@dataclass
class AirframeModel:
    """What the companion believes about the airframe (used for thrust -> throttle)."""

    mass: float = 0.65
    thrust_max: float = 10.0      # N per motor at full command
    motor_idle: float = 0.055
    n_motors: int = 4

    def thrust_to_throttle(self, thrust_n: float) -> float:
        s = math.sqrt(max(thrust_n, 0.0) / (self.n_motors * self.thrust_max))
        thr = (s - self.motor_idle) / (1.0 - self.motor_idle)
        return min(1.0, max(0.0, thr))

    def throttle_to_thrust(self, thr: float) -> float:
        s = self.motor_idle + (1.0 - self.motor_idle) * min(1.0, max(0.0, thr))
        return self.n_motors * self.thrust_max * s * s


def _stick_with_deadband(eff: float, db: float) -> float:
    """Inverse of the FC's deadband rescaling ``(v - sign(v) db) / (1 - db)``."""
    if eff == 0.0:
        return 0.0
    return math.copysign(abs(eff) * (1.0 - db) + db, eff)


class CommandEncoder:
    def __init__(self, cfg: FcConfig | None = None, airframe: AirframeModel | None = None):
        self.cfg = cfg or FcConfig()
        self.air = airframe or AirframeModel(motor_idle=self.cfg.motor_idle)

    def rate_to_stick(self, axis: int, rate_dps: float) -> float:
        c = self.cfg
        if axis == 1:
            rate_dps = -rate_dps
        eff = rc_rate_curve_inverse(rate_dps, c.rate_center[axis], c.rate_max[axis], c.rate_expo[axis])
        return _stick_with_deadband(eff, c.rc_deadband)

    def stick_to_rate(self, axis: int, stick: float) -> float:
        c = self.cfg
        db = c.rc_deadband
        eff = 0.0 if abs(stick) <= db else (stick - math.copysign(db, stick)) / (1.0 - db)
        r = rc_rate_curve(eff, c.rate_center[axis], c.rate_max[axis], c.rate_expo[axis])
        return -r if axis == 1 else r

    def channels(self, rates_rad_s, thrust_n: float, arm: bool, angle_mode: bool = False,
                 throttle: float | None = None) -> list[int]:
        """Rates (p, q, r) in rad/s (FRD body) + collective thrust in N -> 16 channels."""
        ch = [crsf.CH_MID] * crsf.NUM_CHANNELS
        for axis, ch_idx in ((0, 0), (1, 1), (2, 3)):
            ch[ch_idx] = crsf.norm_to_ch(self.rate_to_stick(axis, math.degrees(rates_rad_s[axis])))
        thr = self.air.thrust_to_throttle(thrust_n) if throttle is None else throttle
        ch[2] = crsf.norm_to_ch(2.0 * thr - 1.0)
        ch[self.cfg.arm_channel] = crsf.us_to_ch(ARM_ON_US if arm else ARM_OFF_US)
        ch[self.cfg.mode_channel] = crsf.us_to_ch(MODE_ANGLE_US if angle_mode else MODE_ACRO_US)
        return ch

    def frame(self, rates_rad_s, thrust_n: float, arm: bool, angle_mode: bool = False,
              throttle: float | None = None) -> bytes:
        return crsf.build_rc_channels(self.channels(rates_rad_s, thrust_n, arm, angle_mode, throttle))

    def decode(self, ch) -> dict:
        """Channels -> what the FC will do (for logging and tests)."""
        rates = [math.radians(self.stick_to_rate(a, crsf.ch_to_norm(ch[i])))
                 for a, i in ((0, 0), (1, 1), (2, 3))]
        thr = 0.5 * (crsf.ch_to_norm(ch[2]) + 1.0)
        return dict(rates=rates, throttle=thr, thrust=self.air.throttle_to_thrust(thr),
                    arm=crsf.ch_to_us(ch[self.cfg.arm_channel]) > 1700.0,
                    angle=crsf.ch_to_us(ch[self.cfg.mode_channel]) > 1300.0)
