"""Python port of the skyforge-fc flight-controller core (``src/core``).

Ported pieces (same structure, same default gains, same scaling constants):

* ``config_defaults``  -> :class:`FcConfig`
* ``pt1`` / ``biquad`` filters, ``pid_axis_update`` (P, conditional I, D on
  measurement, feed-forward), ``pid_tpa_factor``, ``rc_rate_curve``
* ``angle_controller`` (used by the failsafe landing), Madgwick AHRS
* ``mixer_mix`` (quad-X with airmode desaturation), ``dshot_value_from_motor``
* ``fc_rc_from_crsf`` (AETR + AUX arm/mode), and ``fc_pid_step``: arming state
  machine (arm-switch edge, throttle / tilt / gyro-cal / RX-loss blocks),
  RX-loss failsafe (timeout -> level + fixed-throttle landing -> disarm),
  acro rate setpoints, inner rate loop, mixer.

What is *not* ported: the gyro-calibration motion check (calibration simply
averages the first second of samples), the EKF AHRS option, the notch filter
dynamics (disabled by default in skyforge-fc too), bench motor test and CLI.
The C backend (:mod:`skyrace.sitl`) runs the real C code when you want those.

All floats are Python floats; the functions are written scalar-by-scalar on
purpose so that one 1 kHz PID step costs ~20 us in CPython.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import crsf

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
GRAVITY = 9.80665

PID_P_SCALE = 3.2e-5
PID_I_SCALE = 2.44e-4
PID_D_SCALE = 5.29e-7
PID_F_SCALE = 1.4e-7

DSHOT_MIN_THROTTLE = 48
DSHOT_MAX_THROTTLE = 2047
DSHOT_RANGE = DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE
DSHOT_CMD_MOTOR_STOP = 0

AIRMODE_START_THROTTLE = 0.25
FS_PROC_DROP = 0
FS_PROC_LAND = 1

FS_IDLE, FS_LANDING, FS_DISARMED = 0, 1, 2

ARM_BLOCK_GYRO_CAL = 1 << 0
ARM_BLOCK_RX_LOSS = 1 << 1
ARM_BLOCK_FAILSAFE = 1 << 2
ARM_BLOCK_THROTTLE = 1 << 3
ARM_BLOCK_ANGLE = 1 << 4
ARM_BLOCK_ARM_SWITCH = 1 << 5

AXIS_ROLL, AXIS_PITCH, AXIS_YAW = 0, 1, 2


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class FcConfig:
    """``config_defaults()`` of skyforge-fc (only the fields the core uses)."""

    pid_p: list = field(default_factory=lambda: [55.0, 58.0, 50.0])
    pid_i: list = field(default_factory=lambda: [50.0, 52.0, 60.0])
    pid_d: list = field(default_factory=lambda: [45.0, 48.0, 0.0])
    pid_f: list = field(default_factory=lambda: [100.0, 105.0, 90.0])
    rate_center: list = field(default_factory=lambda: [200.0, 200.0, 200.0])
    rate_max: list = field(default_factory=lambda: [670.0, 670.0, 670.0])
    rate_expo: list = field(default_factory=lambda: [0.54, 0.54, 0.54])
    i_limit: float = 0.25
    dterm_lpf_hz: float = 100.0
    ff_lpf_hz: float = 40.0
    tpa_rate: float = 0.65
    tpa_breakpoint: float = 0.65
    angle_p: float = 6.0
    angle_max_deg: float = 55.0
    angle_max_rate: float = 500.0
    gyro_lpf_hz: float = 200.0
    accel_lpf_hz: float = 20.0
    madgwick_beta: float = 0.02
    airmode: int = 1
    yaw_reversed: int = 0
    motor_idle: float = 0.055
    gyro_rate_hz: int = 8000
    pid_loop_hz: int = 4000
    arm_channel: int = 4
    mode_channel: int = 5
    rc_deadband: float = 0.01
    fs_timeout_ms: int = 300
    fs_procedure: int = FS_PROC_LAND
    fs_throttle: float = 0.30
    fs_land_time_s: float = 5.0
    arm_max_angle_deg: float = 25.0
    arm_max_throttle: float = 0.05


# --------------------------------------------------------------------- filters
class Pt1:
    __slots__ = ("k", "y")

    def __init__(self, cutoff_hz: float, sample_hz: float):
        if cutoff_hz <= 0.0 or sample_hz <= 0.0:
            self.k = 1.0
        else:
            rc = 1.0 / (2.0 * math.pi * cutoff_hz)
            dt = 1.0 / sample_hz
            self.k = dt / (rc + dt)
        self.y = 0.0

    def apply(self, x: float) -> float:
        self.y += self.k * (x - self.y)
        return self.y


class BiquadLpf:
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "z1", "z2")

    def __init__(self, cutoff_hz: float, sample_hz: float, q: float = 0.70710678):
        self.z1 = self.z2 = 0.0
        if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * sample_hz or q <= 0.0:
            self.b0, self.b1, self.b2, self.a1, self.a2 = 1.0, 0.0, 0.0, 0.0, 0.0
            return
        w0 = 2.0 * math.pi * cutoff_hz / sample_hz
        sn, cs = math.sin(w0), math.cos(w0)
        alpha = sn / (2.0 * q)
        a0 = 1.0 + alpha
        self.b0 = ((1.0 - cs) * 0.5) / a0
        self.b1 = (1.0 - cs) / a0
        self.b2 = self.b0
        self.a1 = (-2.0 * cs) / a0
        self.a2 = (1.0 - alpha) / a0

    def apply(self, x: float) -> float:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


# ------------------------------------------------------------------------- PID
class PidAxis:
    def __init__(self, kp, ki, kd, kf, loop_hz, dterm_lpf_hz, ff_lpf_hz, i_limit=0.25):
        self.kp, self.ki, self.kd, self.kf = kp, ki, kd, kf
        self.i_limit = i_limit
        self.out_limit = 1.0
        self.dterm_lpf = Pt1(dterm_lpf_hz, loop_hz)
        self.ff_lpf = Pt1(ff_lpf_hz, loop_hz)
        self.reset()

    def reset(self):
        self.integ = 0.0
        self.prev_meas = 0.0
        self.prev_sp = 0.0
        self.dterm_lpf.y = 0.0
        self.ff_lpf.y = 0.0
        self.initialised = False
        self.out = 0.0

    def update(self, sp: float, meas: float, dt: float, tpa: float, saturated: bool) -> float:
        if dt <= 0.0:
            return self.out
        if not self.initialised:
            self.prev_meas = meas
            self.prev_sp = sp
            self.initialised = True
        err = sp - meas
        p = self.kp * PID_P_SCALE * err * tpa
        di = self.ki * PID_I_SCALE * err * dt
        if not saturated or self.integ * di < 0.0:
            self.integ += di
        self.integ = clamp(self.integ, -self.i_limit, self.i_limit)
        d = -self.kd * PID_D_SCALE * self.dterm_lpf.apply((meas - self.prev_meas) / dt) * tpa
        f = self.kf * PID_F_SCALE * self.ff_lpf.apply((sp - self.prev_sp) / dt)
        self.prev_meas = meas
        self.prev_sp = sp
        self.out = clamp(p + self.integ + d + f, -self.out_limit, self.out_limit)
        return self.out


def pid_tpa_factor(throttle: float, breakpoint: float, rate: float) -> float:
    if throttle <= breakpoint or breakpoint >= 1.0:
        return 1.0
    t = (throttle - breakpoint) / (1.0 - breakpoint)
    return 1.0 - clamp(rate, 0.0, 1.0) * clamp(t, 0.0, 1.0)


def rc_rate_curve(stick: float, center_dps: float, max_dps: float, expo: float) -> float:
    """Stick [-1, 1] -> rate (deg/s); ``rc_rate_curve`` of skyforge-fc."""
    stick = clamp(stick, -1.0, 1.0)
    expo = clamp(expo, 0.0, 1.0)
    ax = abs(stick)
    x4 = ax * ax * ax * ax
    shape = stick * ax * ((1.0 - expo) + expo * x4)
    extra = max(0.0, max_dps - center_dps)
    return stick * center_dps + extra * shape


def rc_rate_curve_inverse(rate_dps: float, center_dps: float, max_dps: float, expo: float) -> float:
    """Exact inverse of :func:`rc_rate_curve` (it is odd and strictly increasing).

    Newton iterations on |stick| in [0, 1], safeguarded by bisection.
    Rates beyond ``max_dps`` saturate at stick = +-1.
    """
    s = 1.0 if rate_dps >= 0.0 else -1.0
    r = abs(rate_dps)
    if r >= rc_rate_curve(1.0, center_dps, max_dps, expo):
        return s
    lo, hi = 0.0, 1.0
    x = min(1.0, r / max(center_dps, 1e-6))
    extra = max(0.0, max_dps - center_dps)
    for _ in range(40):
        f = rc_rate_curve(x, center_dps, max_dps, expo) - r
        if f > 0.0:
            hi = x
        else:
            lo = x
        if abs(f) < 1e-9:
            break
        df = center_dps + extra * (2.0 * x * (1.0 - expo) + 6.0 * expo * x ** 5)
        xn = x - f / df if df > 0.0 else 0.5 * (lo + hi)
        x = xn if lo < xn < hi else 0.5 * (lo + hi)
    return s * x


# ------------------------------------------------------------ quaternion (w,x,y,z)
def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def q_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def q_normalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def q_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def q_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def q_to_rotvec(q):
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-7:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    k = 2.0 * math.atan2(s, w) / s
    return (x * k, y * k, z * k)


def q_tilt_deg(q):
    w, x, y, z = q
    return math.degrees(math.acos(clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def angle_controller(q, roll_sp, pitch_sp, kp, max_rate_dps):
    _, _, yaw = q_to_euler(q)
    q_err = q_mul(q_conj(q), q_from_euler(roll_sp, pitch_sp, yaw))
    rv = q_to_rotvec(q_err)
    return (clamp(rv[0] * kp * RAD2DEG, -max_rate_dps, max_rate_dps),
            clamp(rv[1] * kp * RAD2DEG, -max_rate_dps, max_rate_dps), 0.0)


class Madgwick:
    """``madgwick.c`` (NED/FRD), with the 2 s boot gain used by ``ahrs.c``."""

    def __init__(self, beta: float):
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.beta = beta
        self.beta_boot = 2.5
        self.boot_time_s = 2.0
        self.t = 0.0
        self.acc_reject = 0.25
        self.aligned = False

    def align(self, a):
        n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        if n < 0.1:
            return
        d = (-a[0] / n, -a[1] / n, -a[2] / n)
        self.q = q_from_euler(math.atan2(d[1], d[2]), math.asin(clamp(-d[0], -1, 1)), 0.0)

    def update(self, g, a, dt):
        if not self.aligned:
            an = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
            if 0.8 < an < 1.2:
                self.align(a)
                self.aligned = True
        if math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) > 1.0:
            a = (0.0, 0.0, 0.0)  # acc_rate_gate: gyro-only above ~57 deg/s
        qw, qx, qy, qz = self.q
        gx, gy, gz = g
        dw = 0.5 * (-qx * gx - qy * gy - qz * gz)
        dx = 0.5 * (qw * gx + qy * gz - qz * gy)
        dy = 0.5 * (qw * gy - qx * gz + qz * gx)
        dz = 0.5 * (qw * gz + qx * gy - qy * gx)
        an = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        if an > 1e-3 and abs(an - 1.0) < self.acc_reject:
            vx, vy, vz = -a[0] / an, -a[1] / an, -a[2] / an
            f1 = 2.0 * (qx * qz - qw * qy) - vx
            f2 = 2.0 * (qy * qz + qw * qx) - vy
            f3 = 1.0 - 2.0 * (qx * qx + qy * qy) - vz
            sw = -2.0 * qy * f1 + 2.0 * qx * f2
            sx = 2.0 * qz * f1 + 2.0 * qw * f2 - 4.0 * qx * f3
            sy = -2.0 * qw * f1 + 2.0 * qz * f2 - 4.0 * qy * f3
            sz = 2.0 * qx * f1 + 2.0 * qy * f2
            gn = math.sqrt(sw * sw + sx * sx + sy * sy + sz * sz)
            if gn > 1e-9:
                s = (self.beta_boot if self.t < self.boot_time_s else self.beta) / gn
                dw -= s * sw
                dx -= s * sx
                dy -= s * sy
                dz -= s * sz
        self.q = q_normalize((qw + dw * dt, qx + dx * dt, qy + dy * dt, qz + dz * dt))
        self.t += dt


# ----------------------------------------------------------------------- mixer
# roll, pitch, yaw factors (already x0.5) for motors 1 RR CW, 2 FR CCW, 3 RL CCW, 4 FL CW
QUADX_RULES = ((-0.5, -0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))


def mixer_mix(throttle, roll, pitch, yaw, airmode=True, yaw_reversed=False):
    """Returns (motor outputs [4], saturated)."""
    ys = -1.0 if yaw_reversed else 1.0
    throttle = clamp(throttle, 0.0, 1.0)
    rpy = [roll * r + pitch * p + ys * yaw * y for (r, p, y) in QUADX_RULES]
    mn, mx = min(rpy), max(rpy)
    sat = False
    if mx - mn > 1.0:
        sc = 1.0 / (mx - mn)
        rpy = [v * sc for v in rpy]
        mn *= sc
        mx *= sc
        sat = True
    if airmode:
        t = clamp(throttle, -mn, 1.0 - mx)
        if t != throttle:
            sat = True
        return [clamp(t + v, 0.0, 1.0) for v in rpy], sat
    out = []
    for v in rpy:
        u = throttle + v
        if u < 0.0 or u > 1.0:
            sat = True
        out.append(clamp(u, 0.0, 1.0))
    return out, sat


def dshot_value_from_motor(motor: float, idle: float) -> int:
    motor = clamp(motor, 0.0, 1.0)
    idle = clamp(idle, 0.0, 0.5)
    u = idle + (1.0 - idle) * motor
    return min(DSHOT_MAX_THROTTLE, DSHOT_MIN_THROTTLE + int(u * DSHOT_RANGE + 0.5))


# -------------------------------------------------------------------- RC input
@dataclass
class RcCommand:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    arm_switch: bool = False
    angle_mode: bool = False


def rc_from_crsf(cfg: FcConfig, ch) -> RcCommand:
    """``fc_rc_from_crsf``: AETR order + AUX arm (>1700 us) / angle mode (>1300 us)."""
    return RcCommand(
        roll=crsf.ch_to_norm(ch[0]),
        pitch=crsf.ch_to_norm(ch[1]),
        throttle=0.5 * (crsf.ch_to_norm(ch[2]) + 1.0),
        yaw=crsf.ch_to_norm(ch[3]),
        arm_switch=crsf.ch_to_us(ch[cfg.arm_channel & 15]) > 1700.0,
        angle_mode=crsf.ch_to_us(ch[cfg.mode_channel & 15]) > 1300.0,
    )


def _deadband(v, db):
    if abs(v) <= db:
        return 0.0
    return (v - math.copysign(db, v)) / (1.0 - db)


# ------------------------------------------------------------------- FC core
class FlightController:
    """``fc_t`` + ``fc_gyro_sample`` + ``fc_rc_update`` + ``fc_pid_step``."""

    def __init__(self, cfg: FcConfig | None = None):
        self.cfg = cfg or FcConfig()
        c = self.cfg
        self.gyro_hz = float(c.gyro_rate_hz)
        self.pid_denom = max(1, min(8, int(self.gyro_hz / c.pid_loop_hz + 0.5)))
        self.pid_hz = self.gyro_hz / self.pid_denom
        self.gyro_count = 0
        self.lpf = [BiquadLpf(c.gyro_lpf_hz, self.gyro_hz) for _ in range(3)]
        self.acc_lpf = [Pt1(c.accel_lpf_hz, self.gyro_hz) for _ in range(3)]
        self.acc_lpf[2].y = -1.0
        self.ahrs = Madgwick(c.madgwick_beta)
        self.pid = [PidAxis(c.pid_p[a], c.pid_i[a], c.pid_d[a], c.pid_f[a], self.pid_hz,
                            c.dterm_lpf_hz, c.ff_lpf_hz, c.i_limit) for a in range(3)]
        self.cal_n = 0
        self.cal_target = int(self.gyro_hz)
        self.cal_sum = [0.0, 0.0, 0.0]
        self.cal_done = False
        self.bias = (0.0, 0.0, 0.0)
        self.gyro_filt = (0.0, 0.0, 0.0)
        self.accel = (0.0, 0.0, -1.0)
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.rc = RcCommand()
        self.rc_last_us = 0
        self.rc_ever = False
        self.rate_sp = (0.0, 0.0, 0.0)
        self.throttle = 0.0
        self.armed = False
        self.airmode_active = False
        self.arm_block = 0
        self.arm_sw_prev = True  # switch must be seen OFF before first arm
        self.fs = FS_IDLE
        self.fs_start_us = 0
        self.motor = [0.0, 0.0, 0.0, 0.0]
        self.saturated = False
        self.now_us = 0
        self.pid_steps = 0

    # sensors ------------------------------------------------------------
    def gyro_sample(self, gyro, accel) -> bool:
        if not self.cal_done:
            for i in range(3):
                self.cal_sum[i] += gyro[i]
            self.cal_n += 1
            if self.cal_n >= self.cal_target:
                self.bias = tuple(s / self.cal_n for s in self.cal_sum)
                self.cal_done = True
        b = self.bias
        self.gyro_filt = (self.lpf[0].apply(gyro[0] - b[0]), self.lpf[1].apply(gyro[1] - b[1]),
                          self.lpf[2].apply(gyro[2] - b[2]))
        self.accel = (self.acc_lpf[0].apply(accel[0]), self.acc_lpf[1].apply(accel[1]),
                      self.acc_lpf[2].apply(accel[2]))
        self.gyro_count += 1
        if self.gyro_count >= self.pid_denom:
            self.gyro_count = 0
            return True
        return False

    def rc_update(self, rc: RcCommand, now_us: int) -> None:
        self.rc = rc
        self.rc_last_us = now_us
        self.rc_ever = True

    def _disarm(self):
        self.armed = False
        for p in self.pid:
            p.reset()

    # control ------------------------------------------------------------
    def pid_step(self, now_us: int) -> None:
        c = self.cfg
        dt = 1.0 / self.pid_hz
        self.now_us = now_us
        self.pid_steps += 1
        self.ahrs.update(self.gyro_filt, self.accel, dt)
        self.q = self.ahrs.q

        rx_ok = self.rc_ever and (now_us - self.rc_last_us) <= c.fs_timeout_ms * 1000
        if self.armed and not rx_ok and self.fs == FS_IDLE:
            if c.fs_procedure == FS_PROC_DROP:
                self._disarm()
                self.fs = FS_DISARMED
            else:
                self.fs = FS_LANDING
                self.fs_start_us = now_us
        if self.fs == FS_LANDING:
            if not self.armed or (now_us - self.fs_start_us) >= c.fs_land_time_s * 1e6:
                self._disarm()
                self.fs = FS_DISARMED
        if self.fs == FS_DISARMED and rx_ok and not self.rc.arm_switch:
            self.fs = FS_IDLE

        blk = 0
        if not self.cal_done:
            blk |= ARM_BLOCK_GYRO_CAL
        if not rx_ok:
            blk |= ARM_BLOCK_RX_LOSS
        if self.fs != FS_IDLE:
            blk |= ARM_BLOCK_FAILSAFE
        if self.rc.throttle > c.arm_max_throttle:
            blk |= ARM_BLOCK_THROTTLE
        if q_tilt_deg(self.q) > c.arm_max_angle_deg:
            blk |= ARM_BLOCK_ANGLE
        if rx_ok:
            sw = self.rc.arm_switch
            if not self.armed:
                if sw and not self.arm_sw_prev and blk == 0:
                    self.armed = True
                    self.airmode_active = False
                    for p in self.pid:
                        p.reset()
                elif sw:
                    blk |= ARM_BLOCK_ARM_SWITCH
            elif not sw:
                self._disarm()
                if self.fs == FS_LANDING:
                    self.fs = FS_IDLE
            self.arm_sw_prev = sw
        self.arm_block = blk

        rc = self.rc
        roll, pitch, yaw, thr, angle = rc.roll, rc.pitch, rc.yaw, rc.throttle, rc.angle_mode
        if self.fs == FS_LANDING:
            roll = pitch = yaw = 0.0
            thr = c.fs_throttle
            angle = True
        sr = _deadband(roll, c.rc_deadband)
        sp = _deadband(pitch, c.rc_deadband)
        sy = _deadband(yaw, c.rc_deadband)
        self.throttle = clamp(thr, 0.0, 1.0)
        rz = rc_rate_curve(sy, c.rate_center[2], c.rate_max[2], c.rate_expo[2])
        if angle:
            amax = c.angle_max_deg * DEG2RAD
            rx, ry, _ = angle_controller(self.q, sr * amax, -sp * amax, c.angle_p,
                                         c.angle_max_rate)
        else:
            rx = rc_rate_curve(sr, c.rate_center[0], c.rate_max[0], c.rate_expo[0])
            ry = -rc_rate_curve(sp, c.rate_center[1], c.rate_max[1], c.rate_expo[1])
        self.rate_sp = (rx, ry, rz)

        if self.armed:
            if self.throttle > AIRMODE_START_THROTTLE:
                self.airmode_active = True
            hold_i = (not self.airmode_active) if c.airmode else (self.throttle < 0.02)
            tpa = pid_tpa_factor(self.throttle, c.tpa_breakpoint, c.tpa_rate)
            g = self.gyro_filt
            out = []
            for a in range(3):
                o = self.pid[a].update(self.rate_sp[a], g[a] * RAD2DEG, dt,
                                       1.0 if a == AXIS_YAW else tpa, self.saturated)
                if hold_i:
                    self.pid[a].integ = 0.0
                out.append(o)
            self.motor, self.saturated = mixer_mix(self.throttle, out[0], out[1], out[2],
                                                   bool(c.airmode), bool(c.yaw_reversed))
        else:
            self.saturated = False
            self.motor = [0.0, 0.0, 0.0, 0.0]

    def motor_dshot(self):
        if not self.armed:
            return [DSHOT_CMD_MOTOR_STOP] * 4
        return [dshot_value_from_motor(m, self.cfg.motor_idle) for m in self.motor]


class CrsfReceiverPort:
    """The FC's receiver UART: raw bytes in -> parser -> ``fc_rc_update``.

    Mirrors ``task_rx`` in skyforge-fc ``src/app/main.c``.
    """

    def __init__(self, fc: FlightController):
        self.fc = fc
        self.parser = crsf.Parser()

    def feed(self, data: bytes, now_us: int) -> int:
        n = 0
        for b in data:
            if self.parser.feed(b, now_us) == crsf.TYPE_RC_CHANNELS:
                self.fc.rc_update(rc_from_crsf(self.fc.cfg, self.parser.channels), now_us)
                self.parser.rc_fresh = False
                n += 1
        return n
