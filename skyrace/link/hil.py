"""Hardware-in-the-loop over TCP: the companion talks to the simulator in another
process using *exactly the CRSF bytes* it would write to the flight controller.

Two TCP connections (both opened by the companion/client):

* **UART** (``port``): a raw, unframed byte stream - companion -> FC: CRSF
  RC_CHANNELS frames; FC -> companion: CRSF ATTITUDE / FLIGHT_MODE telemetry.
  On the real quad this socket is replaced by ``/dev/ttyAMA0`` (Pi 5) or
  ``/dev/ttyTHS1`` (Jetson) at 420000 baud, see :mod:`skyrace.link.bridge`.
* **Sensors** (``port + 1``, HIL only): length-prefixed messages carrying what
  real hardware would get from its own IMU and camera.

Lockstep: per 5 ms tick the server sends one SENSOR message, then reads the
UART socket until it has seen exactly one complete CRSF frame from the
client (found with a separate sniffer parser, the FC's parser gets the raw
bytes), then advances physics. That makes HIL runs deterministic and equal
to the in-process run with the same seed (``tests/test_hil.py``).

SENSOR message: ``b"SKRH" | u32 seq | f64 t | u32 n_imu | u32 n_uart | u8 has_img | u8 end``
followed by n_imu x (f64 t, 3 x f32 gyro, 3 x f32 accel) and, if has_img,
``u16 H | u16 W | H*W*3 bytes RGB``. The client answers the END message with
a JSON summary (u32 length + UTF-8). ``n_uart`` is the number of telemetry
bytes the server wrote to the UART socket for this tick; the client reads
exactly that many before ticking, so both streams stay in lockstep.
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

from ..fcport import crsf
from ..race import RaceSession

HDR = struct.Struct("<4sIdIIBB")
IMU = struct.Struct("<d3f3f")
MAGIC = b"SKRH"


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def encode_sensor(seq, t, imu, img, end=False, n_uart=0) -> bytes:
    parts = [HDR.pack(MAGIC, seq, t, len(imu), n_uart, 1 if img is not None else 0, 1 if end else 0)]
    for (ti, g, a) in imu:
        parts.append(IMU.pack(ti, *map(float, g), *map(float, a)))
    if img is not None:
        u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        parts.append(struct.pack("<HH", u8.shape[0], u8.shape[1]))
        parts.append(u8.tobytes())
    return b"".join(parts)


def decode_sensor(sock):
    magic, seq, t, n, n_uart, has_img, end = HDR.unpack(_recv_exact(sock, HDR.size))
    if magic != MAGIC:
        raise ValueError("bad HIL header")
    imu = []
    for _ in range(n):
        v = IMU.unpack(_recv_exact(sock, IMU.size))
        imu.append((v[0], np.array(v[1:4]), np.array(v[4:7])))
    img = None
    if has_img:
        H, W = struct.unpack("<HH", _recv_exact(sock, 4))
        img = np.frombuffer(_recv_exact(sock, H * W * 3), np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
    return seq, t, imu, img, bool(end), n_uart


class HilServer:
    """Simulator side. ``serve_one()`` accepts one companion and flies until it finishes."""

    def __init__(self, backend, renderer, track, host="localhost", port=5760, t_max=40.0, cmd_delay=0.01):
        self.backend, self.renderer, self.track = backend, renderer, track
        self.t_max, self.cmd_delay = t_max, cmd_delay
        self.ls_uart = socket.create_server((host, port))
        self.ls_sens = socket.create_server((host, port + 1 if port else 0))
        self.port = self.ls_uart.getsockname()[1]
        self.sens_port = self.ls_sens.getsockname()[1]
        self.bytes_rx = 0
        self.frames_rx = 0

    def serve_one(self):
        uart, _ = self.ls_uart.accept()
        sens, _ = self.ls_sens.accept()
        for s in (uart, sens):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sess = RaceSession(self.backend, self.renderer, self.track, cmd_delay=self.cmd_delay)
        sniffer = crsf.Parser()
        pending = bytearray()
        seq = 0
        try:
            while sess.t < self.t_max:
                t, imu, img, rx = sess.sense()
                if rx:
                    uart.sendall(rx)
                sens.sendall(encode_sensor(seq, t, imu, img, n_uart=len(rx)))
                seq += 1
                # read until one complete CRSF frame arrived from the companion
                got = 0
                tick_bytes = bytearray(pending)
                pending.clear()
                for b in bytes(tick_bytes):
                    got += 1 if sniffer.feed(b, 0) else 0
                while got == 0:
                    chunk = uart.recv(256)
                    if not chunk:
                        raise ConnectionError("companion closed the UART")
                    for i, b in enumerate(chunk):
                        if got:
                            pending += chunk[i:]
                            break
                        tick_bytes.append(b)
                        got += 1 if sniffer.feed(b, 0) else 0
                self.bytes_rx += len(tick_bytes)
                self.frames_rx += got
                sess.actuate(bytes(tick_bytes))
                sess.advance()
                if sess.judge.done and sess.t > (sess.judge.t_finish or sess.t) + 0.5:
                    break
            sens.sendall(encode_sensor(seq, sess.t, [], None, end=True))
            n = struct.unpack("<I", _recv_exact(sens, 4))[0]
            client = json.loads(_recv_exact(sens, n).decode())
        finally:
            uart.close()
            sens.close()
        res = sess.summary(client.get("t_race0"), client.get("planned_lap"))
        res.update(client=client, uart_bytes=self.bytes_rx, uart_frames=self.frames_rx)
        return res, sess

    def close(self):
        self.ls_uart.close()
        self.ls_sens.close()


class TcpUart:
    """Client-side UART transport over TCP (same API as :class:`skyrace.link.bridge.SerialUart`)."""

    def __init__(self, host="localhost", port=5760):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)

    def write(self, data: bytes):
        self.sock.setblocking(True)
        self.sock.sendall(data)
        self.sock.setblocking(False)

    def read_exact(self, n: int) -> bytes:
        self.sock.setblocking(True)
        try:
            return _recv_exact(self.sock, n) if n else b""
        finally:
            self.sock.setblocking(False)

    def read(self) -> bytes:
        out = bytearray()
        while True:
            try:
                chunk = self.sock.recv(4096)
            except (BlockingIOError, InterruptedError):
                break
            if not chunk:
                break
            out += chunk
        return bytes(out)

    def close(self):
        self.sock.close()


def run_client(stack, host="localhost", port=5760, sens_port=None):
    """Companion side of HIL: run ``stack`` until the server sends END."""
    uart = TcpUart(host, port)
    sens = socket.create_connection((host, sens_port or port + 1))
    sens.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            seq, t, imu, img, end, n_uart = decode_sensor(sens)
            if end:
                break
            rx = uart.read_exact(n_uart)
            tx = stack.tick(t, imu, img, rx)
            uart.write(tx)
        summary = dict(t_race0=stack.t_race0, phase=stack.phase, gate_fixes=stack.n_fixes,
                       frames=stack.n_frames, safety_events=[list(e) for e in stack.safety.events],
                       planned_lap=float(stack.ref_race.duration) if stack.ref_race is not None else None)
        b = json.dumps(summary).encode()
        sens.sendall(struct.pack("<I", len(b)) + b)
    finally:
        uart.close()
        sens.close()
    return summary
