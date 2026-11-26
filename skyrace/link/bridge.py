"""Serial bridge from the companion computer to the flight controller's receiver UART.

The companion replaces the ELRS receiver: wire its UART TX -> FC RX (and FC
TX -> companion RX for telemetry) on the UART that skyforge-fc uses for CRSF,
420000 baud 8N1, non-inverted, 3.3 V logic, common ground.

    Pi 5        : /dev/ttyAMA0  (GPIO14 TX / GPIO15 RX, enable_uart=1, disable the serial console)
    Jetson Orin : /dev/ttyTHS1  (40-pin header pins 8 / 10)
    Orange Pi 5 : /dev/ttyS4    (overlay uart4-m0)  - check your board's pinout

pyserial is the only extra dependency (``pip install skyrace[serial]``).
Nothing in this module has been run against real hardware here (see the
README "Hardware validation" section).
"""
from __future__ import annotations

import time

from ..fcport import crsf

PORTS = {"pi5": "/dev/ttyAMA0", "jetson": "/dev/ttyTHS1", "orangepi5": "/dev/ttyS4"}


class SerialUart:
    def __init__(self, port: str, baud: int = crsf.BAUDRATE, timeout: float = 0.0):
        try:
            import serial  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("pyserial is required for the serial bridge: pip install pyserial") from e
        self.ser = serial.Serial(port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=timeout,
                                 write_timeout=0.05)

    def write(self, data: bytes):
        self.ser.write(data)

    def read(self) -> bytes:
        n = self.ser.in_waiting
        return self.ser.read(n) if n else b""

    def close(self):
        self.ser.close()


class LoopbackUart:
    """In-memory UART pair for tests: what one side writes, the other reads."""

    def __init__(self):
        self.a2b = bytearray()
        self.b2a = bytearray()

    class _End:
        def __init__(self, tx, rx):
            self.tx, self.rx = tx, rx

        def write(self, data: bytes):
            self.tx += data

        def read(self) -> bytes:
            out = bytes(self.rx)
            self.rx.clear()
            return out

        def close(self):
            pass

    def ends(self):
        return self._End(self.a2b, self.b2a), self._End(self.b2a, self.a2b)


def frame_airtime_s(nbytes: int, baud: int = crsf.BAUDRATE) -> float:
    """8N1 = 10 bits per byte. One 26-byte RC frame at 420 kbaud takes 0.62 ms."""
    return nbytes * 10.0 / baud


def run_bridge(stack, uart, camera=None, imu=None, rate_hz: float = 200.0, duration: float = 60.0):
    """Real-time companion loop on hardware (camera/imu are user-supplied callables).

    ``camera()`` -> latest HxWx3 float image or None; ``imu()`` -> list of
    (t, gyro, accel) since the last call. This is the loop a Pi 5 would run;
    it is exercised in tests only with :class:`LoopbackUart` and fake sensors.
    """
    period = 1.0 / rate_hz
    t0 = time.monotonic()
    next_t = t0
    while True:
        now = time.monotonic()
        t = now - t0
        if t > duration:
            break
        rx = uart.read()
        img = camera() if camera else None
        samples = imu() if imu else []
        uart.write(stack.tick(t, samples, img, rx))
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))
    # leave the FC disarmed
    uart.write(stack.enc.frame((0, 0, 0), 0.0, arm=False, throttle=0.0))
