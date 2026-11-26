"""Rigid-body quadrotor + motors + drag, with the (ported) skyforge-fc core in the loop.

The physics is the model of skyforge-fc ``sim/quad_sim.c`` (NED world, FRD body,
explicit Euler at the gyro rate)::

    motor speed      s_i' = (u_i - s_i) / tau,     u_i from the DShot value
    thrust           T_i  = T_max s_i^2           along body -z
    reaction torque  Q_i  = -/+ k_q T_i
    rotation         I w' = sum r_i x F_i + Q - c_w w - w x I w
    translation      m a  = R F_b + m g e3 - c_v v

with the same default airframe (650 g 5" quad, 10 N per motor, 25 ms motor
time constant). The only interface to the flight controller is the receiver
UART: :meth:`QuadSim.write_uart` takes raw CRSF bytes exactly as a companion
computer would send them, and :meth:`QuadSim.read_uart` returns the CRSF
telemetry frames the FC transmits (attitude + flight mode at 10 Hz).

A second, independent IMU (the companion computer's) is sampled at
``imu_rate_hz`` with its own bias and noise; that is what the estimator sees.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ..fcport import crsf
from ..fcport.fc import (DSHOT_MIN_THROTTLE, DSHOT_RANGE, GRAVITY, CrsfReceiverPort, FcConfig,
                         FlightController, q_mul, q_normalize, q_to_euler)


@dataclass
class QuadParams:
    """Defaults = ``sim_reset()`` in skyforge-fc ``sim/quad_sim.c``."""

    mass: float = 0.65
    ixx: float = 0.0022
    iyy: float = 0.0024
    izz: float = 0.0040
    arm_xy: float = 0.0795
    thrust_max: float = 10.0
    kq: float = 0.015
    motor_tau: float = 0.025
    drag_lin: float = 0.15
    drag_rot: float = 0.0004
    gyro_noise: float = 0.01
    accel_noise: float = 0.03
    gyro_bias: tuple = (0.010, -0.008, 0.005)
    # companion-computer IMU (what the estimator uses), SI units
    imu_rate_hz: float = 500.0
    imu_gyro_noise: float = 0.004      # rad/s per sample
    imu_accel_noise: float = 0.08      # m/s^2 per sample
    imu_gyro_bias: tuple = (0.004, -0.003, 0.002)
    imu_accel_bias: tuple = (0.05, -0.04, 0.06)


def _qrot(q, v):
    """Rotate body vector v into world with quaternion q (w,x,y,z)."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx))


def _qrot_inv(q, v):
    return _qrot((q[0], -q[1], -q[2], -q[3]), v)


def _q_from_rotvec(rx, ry, rz):
    a = math.sqrt(rx * rx + ry * ry + rz * rz)
    if a < 1e-6:
        return (1.0 - a * a * 0.125, 0.5 * rx, 0.5 * ry, 0.5 * rz)
    s = math.sin(0.5 * a) / a
    return (math.cos(0.5 * a), rx * s, ry * s, rz * s)


class QuadSim:
    """Pure-Python backend. ``gyro_rate_hz``/``pid_loop_hz`` default to 2 kHz / 1 kHz
    (skyforge-fc runs 8 kHz / 4 kHz on hardware; the C SITL backend does too)."""

    name = "python-port"

    def __init__(self, params: QuadParams | None = None, cfg: FcConfig | None = None,
                 seed: int = 1, gyro_rate_hz: int = 2000, pid_loop_hz: int = 1000):
        self.p = params or QuadParams()
        if cfg is None:
            cfg = FcConfig(gyro_rate_hz=gyro_rate_hz, pid_loop_hz=pid_loop_hz)
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, pos=(0.0, 0.0, 0.0), yaw: float = 0.0):
        self.fc = FlightController(self.cfg)
        self.rx = CrsfReceiverPort(self.fc)
        self.pos = list(map(float, pos))
        self.vel = [0.0, 0.0, 0.0]
        self.q = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
        self.w = [0.0, 0.0, 0.0]
        self.s = [0.0, 0.0, 0.0, 0.0]
        self.on_ground = pos[2] >= 0.0
        self.t = 0.0
        self.now_us = 1000
        self.dt = 1.0 / self.cfg.gyro_rate_hz
        self.dt_us = int(1e6 / self.cfg.gyro_rate_hz + 0.5)
        self.tx = bytearray()
        self.next_telem = 0.1
        self.imu_buf = []
        self.imu_dt = 1.0 / self.p.imu_rate_hz
        self.next_imu = self.imu_dt
        self.acc_world = (0.0, 0.0, 0.0)
        self.thrust = 0.0
        self.max_ground_speed = 0.0
        self.touchdowns = 0

    # ---------------------------------------------------------------- UART
    def write_uart(self, data: bytes) -> None:
        """Companion -> FC receiver UART (raw CRSF bytes)."""
        self.rx.feed(data, self.now_us)

    def read_uart(self) -> bytes:
        """FC -> companion (CRSF telemetry frames)."""
        out = bytes(self.tx)
        self.tx.clear()
        return out

    def _telemetry(self):
        r, p, y = q_to_euler(self.fc.q)
        self.tx += crsf.build_attitude(p, r, y)
        fc = self.fc
        if fc.fs != 0:
            mode = "!FS!"
        else:
            mode = ("ANGL" if fc.rc.angle_mode else "ACRO") + ("" if fc.armed else "*")
        self.tx += crsf.build_flight_mode(mode)

    # ---------------------------------------------------------------- physics
    def _physics(self, u, dt):
        p = self.p
        a = p.arm_xy
        rx = (-a, a, -a, a)
        ry = (a, a, -a, -a)
        spin = (-1.0, 1.0, 1.0, -1.0)
        thrust = 0.0
        tx = ty = tz = 0.0
        k = dt / p.motor_tau
        for i in range(4):
            self.s[i] += (u[i] - self.s[i]) * k
            T = p.thrust_max * self.s[i] * self.s[i]
            thrust += T
            tx += -ry[i] * T
            ty += rx[i] * T
            tz += spin[i] * p.kq * T
        w = self.w
        tx -= w[0] * p.drag_rot
        ty -= w[1] * p.drag_rot
        tz -= w[2] * p.drag_rot
        Iw = (p.ixx * w[0], p.iyy * w[1], p.izz * w[2])
        wd = ((tx - (w[1] * Iw[2] - w[2] * Iw[1])) / p.ixx,
              (ty - (w[2] * Iw[0] - w[0] * Iw[2])) / p.iyy,
              (tz - (w[0] * Iw[1] - w[1] * Iw[0])) / p.izz)
        f = _qrot(self.q, (0.0, 0.0, -thrust))
        v = self.vel
        acc = (f[0] / p.mass - v[0] * p.drag_lin / p.mass,
               f[1] / p.mass - v[1] * p.drag_lin / p.mass,
               f[2] / p.mass + GRAVITY - v[2] * p.drag_lin / p.mass)
        self.thrust = thrust
        if self.on_ground:
            if acc[2] < 0.0:
                self.on_ground = False
            else:
                self.vel = [0.0, 0.0, 0.0]
                self.w = [0.0, 0.0, 0.0]
                self.acc_world = (0.0, 0.0, 0.0)
                return
        self.acc_world = acc
        self.w = [w[0] + wd[0] * dt, w[1] + wd[1] * dt, w[2] + wd[2] * dt]
        w = self.w
        self.q = q_normalize(q_mul(self.q, _q_from_rotvec(w[0] * dt, w[1] * dt, w[2] * dt)))
        self.vel = [v[0] + acc[0] * dt, v[1] + acc[1] * dt, v[2] + acc[2] * dt]
        self.pos = [self.pos[i] + self.vel[i] * dt for i in range(3)]
        if self.pos[2] > 0.0:
            self.max_ground_speed = max(self.max_ground_speed, math.sqrt(sum(x * x for x in self.vel)))
            self.touchdowns += 1
            self.pos[2] = 0.0
            self.vel = [0.0, 0.0, 0.0]
            self.w = [0.0, 0.0, 0.0]
            r, pch, y = q_to_euler(self.q)
            self.q = (math.cos(0.5 * y), 0.0, 0.0, math.sin(0.5 * y))
            self.on_ground = True

    def _specific_force_body(self):
        """Specific force in body frame (m/s^2): R^T (a - g e3)."""
        if self.on_ground and self.thrust < self.p.mass * GRAVITY:
            return _qrot_inv(self.q, (0.0, 0.0, -GRAVITY))
        a = self.acc_world
        return _qrot_inv(self.q, (a[0], a[1], a[2] - GRAVITY))

    def _sense_fc(self):
        p, n = self.p, self.rng.gauss
        g = (self.w[0] + p.gyro_bias[0] + p.gyro_noise * n(0, 1),
             self.w[1] + p.gyro_bias[1] + p.gyro_noise * n(0, 1),
             self.w[2] + p.gyro_bias[2] + p.gyro_noise * n(0, 1))
        f = self._specific_force_body()
        a = (f[0] / GRAVITY + p.accel_noise * n(0, 1), f[1] / GRAVITY + p.accel_noise * n(0, 1),
             f[2] / GRAVITY + p.accel_noise * n(0, 1))
        return g, a

    def _sense_imu(self):
        p, n = self.p, self.rng.gauss
        f = self._specific_force_body()
        g = tuple(self.w[i] + p.imu_gyro_bias[i] + p.imu_gyro_noise * n(0, 1) for i in range(3))
        a = tuple(f[i] + p.imu_accel_bias[i] + p.imu_accel_noise * n(0, 1) for i in range(3))
        self.imu_buf.append((self.t, np.array(g), np.array(a)))

    # ---------------------------------------------------------------- stepping
    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.dt)))
        fc = self.fc
        for _ in range(n):
            self.now_us += self.dt_us
            self.t += self.dt
            g, a = self._sense_fc()
            if fc.gyro_sample(g, a):
                fc.pid_step(self.now_us)
            d = fc.motor_dshot()
            u = [(x - DSHOT_MIN_THROTTLE) / DSHOT_RANGE if x >= DSHOT_MIN_THROTTLE else 0.0 for x in d]
            self._physics(u, self.dt)
            if self.t >= self.next_imu - 1e-9:
                self.next_imu += self.imu_dt
                self._sense_imu()
            if self.t >= self.next_telem - 1e-9:
                self.next_telem += 0.1
                self._telemetry()

    def imu_samples(self):
        out, self.imu_buf = self.imu_buf, []
        return out

    # --------------------------------------------------------------- queries
    def state(self) -> dict:
        return dict(t=self.t, pos=np.array(self.pos), vel=np.array(self.vel), q=np.array(self.q),
                    w=np.array(self.w), armed=self.fc.armed, fs=self.fc.fs,
                    motors=list(self.fc.motor), on_ground=self.on_ground,
                    rate_sp=np.array(self.fc.rate_sp), throttle=self.fc.throttle)

    def hover_throttle(self) -> float:
        sh = math.sqrt(self.p.mass * GRAVITY / (4.0 * self.p.thrust_max))
        idle = self.cfg.motor_idle
        return (sh - idle) / (1.0 - idle)
