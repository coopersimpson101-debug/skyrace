"""ctypes backend that flies the REAL skyforge-fc C code (8 kHz gyro / 4 kHz PID).

Build the library first (needs a skyforge-fc checkout and a C compiler)::

    python scripts/build_sitl.py --skyforge-fc ../skyforge-fc

The backend has the same interface as :class:`skyrace.sim.quad.QuadSim`, so
the race runner and the HIL server can use either. The companion IMU is
synthesised here from the C model's state (finite-differenced velocity),
because quad_sim.c only exposes the FC's own IMU.
"""
from __future__ import annotations

import ctypes
import math
import os
import random
import sys

import numpy as np

from .sim.quad import QuadParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXT = ".dll" if sys.platform.startswith("win") else (".dylib" if sys.platform == "darwin" else ".so")
DEFAULT_LIB = os.environ.get("SKYRACE_SITL_LIB", os.path.join(ROOT, "build", "sitl", "skyrace_sitl" + _EXT))
G = 9.80665


def available(path: str = DEFAULT_LIB) -> bool:
    return os.path.exists(path)


def load(path: str = DEFAULT_LIB):
    L = ctypes.CDLL(path)
    vp, f, i, u32 = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_uint32
    L.sitl_create.restype = vp
    L.sitl_destroy.argtypes = [vp]
    L.sitl_reset.argtypes = [vp, u32, f, f, f, f]
    L.sitl_set_param.argtypes = [vp, ctypes.c_char_p, f]
    L.sitl_set_param.restype = i
    L.sitl_set_airframe.argtypes = [vp, f, f, f]
    L.sitl_uart_write.argtypes = [vp, ctypes.c_char_p, i]
    L.sitl_uart_read.argtypes = [vp, ctypes.c_char_p, i]
    L.sitl_uart_read.restype = i
    L.sitl_step.argtypes = [vp, i]
    L.sitl_state_len.restype = i
    L.sitl_get.argtypes = [vp, ctypes.POINTER(f)]
    L.sitl_crc8.argtypes = [ctypes.c_char_p, i]
    L.sitl_crc8.restype = ctypes.c_uint8
    L.sitl_pack.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_char_p]
    L.sitl_unpack.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16)]
    L.sitl_rate_curve.argtypes = [f, f, f, f]
    L.sitl_rate_curve.restype = f
    L.sitl_ch_to_norm.argtypes = [ctypes.c_uint16]
    L.sitl_ch_to_norm.restype = f
    L.sitl_mix.argtypes = [f, f, f, f, i, ctypes.POINTER(f)]
    L.sitl_mix.restype = i
    L.sitl_dshot_value.argtypes = [f, f]
    L.sitl_dshot_value.restype = i
    L.sitl_pid_run.argtypes = [ctypes.POINTER(f), ctypes.POINTER(f), i, f, f, f, f, f, ctypes.POINTER(f)]
    L.sitl_pid_run.restype = f
    return L


class SkyforgeSITL:
    name = "skyforge-fc C (SITL)"
    GYRO_HZ = 8000

    def __init__(self, params: QuadParams | None = None, seed: int = 1, lib_path: str = DEFAULT_LIB,
                 airframe=None):
        self.L = load(lib_path)
        self.h = self.L.sitl_create()
        self.p = params or QuadParams()
        self.seed = seed
        self.airframe = airframe   # (mass, thrust_max, drag) override for model-mismatch runs
        self.n = self.L.sitl_state_len()
        self.buf = (ctypes.c_float * self.n)()
        self.rbuf = ctypes.create_string_buffer(4096)
        self.reset()

    def __del__(self):
        if getattr(self, "h", None):
            self.L.sitl_destroy(self.h)
            self.h = None

    def reset(self, pos=(0.0, 0.0, 0.0), yaw: float = 0.0):
        self.L.sitl_reset(self.h, self.seed, float(pos[0]), float(pos[1]), float(pos[2]), float(yaw))
        if self.airframe is not None:
            self.L.sitl_set_airframe(self.h, *map(float, self.airframe))
        self.rng = random.Random(self.seed + 7)
        self.imu_buf = []
        self.imu_steps = int(round(self.GYRO_HZ / self.p.imu_rate_hz))
        self._read()

    def set_param(self, name: str, value: float) -> bool:
        return bool(self.L.sitl_set_param(self.h, name.encode(), float(value)))

    def write_uart(self, data: bytes) -> None:
        if data:
            self.L.sitl_uart_write(self.h, bytes(data), len(data))

    def read_uart(self) -> bytes:
        n = self.L.sitl_uart_read(self.h, self.rbuf, 4096)
        return self.rbuf.raw[:n]

    def _read(self):
        self.L.sitl_get(self.h, self.buf)
        self.s = np.frombuffer(self.buf, dtype=np.float32, count=self.n).astype(float)
        return self.s

    def step(self, seconds: float) -> None:
        total = max(1, int(round(seconds * self.GYRO_HZ)))
        done = 0
        while done < total:
            k = min(self.imu_steps, total - done)
            v0 = self.s[4:7].copy()
            self.L.sitl_step(self.h, k)
            done += k
            s = self._read()
            if k == self.imu_steps:
                self._imu(v0, s, k / self.GYRO_HZ)

    def _imu(self, v0, s, dt):
        q = s[7:11]
        R = _rot(q)
        a = (s[4:7] - v0) / dt
        f_b = R.T @ (a - np.array([0.0, 0.0, G]))
        p, n = self.p, self.rng.gauss
        g = s[11:14] + np.array(p.imu_gyro_bias) + np.array([p.imu_gyro_noise * n(0, 1) for _ in range(3)])
        acc = f_b + np.array(p.imu_accel_bias) + np.array([p.imu_accel_noise * n(0, 1) for _ in range(3)])
        self.imu_buf.append((float(s[0]), g, acc))

    def imu_samples(self):
        out, self.imu_buf = self.imu_buf, []
        return out

    def state(self) -> dict:
        s = self.s
        return dict(t=s[0], pos=s[1:4].copy(), vel=s[4:7].copy(), q=s[7:11].copy(), w=s[11:14].copy(),
                    armed=bool(s[14] > 0.5), fs=int(s[15]), motors=list(s[16:20]),
                    rate_sp=s[20:23].copy(), on_ground=bool(s[23] > 0.5), throttle=s[24],
                    rc_frames=int(s[25]), arm_block=int(s[26]))


def _rot(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
