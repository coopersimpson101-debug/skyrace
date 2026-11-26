"""CRSF (TBS Crossfire / ExpressLRS) wire protocol, byte-compatible with skyforge-fc.

This is a line-by-line Python port of ``src/core/crsf.c`` in skyforge-fc:

* frame  ``[sync][len][type][payload...][crc8]`` with ``len = 1 + payload + 1``
* CRC-8/DVB-S2 (polynomial 0xD5, init 0) over ``type + payload``
* RC_CHANNELS_PACKED (type 0x16): 16 channels x 11 bit, LSB first, 22 bytes
* channel value mapping 172 -> 988 us, 992 -> 1500 us, 1811 -> 2012 us
* parser accepts sync bytes 0xC8 / 0xEE / 0xEA / 0xEC, rejects ``len`` outside
  [2, 62], and drops a partial frame after 1500 us of silence.

``tests/test_crsf.py`` checks this port against the real C implementation
(compiled from a skyforge-fc checkout) when one is available.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

BAUDRATE = 420000
SYNC_FC = 0xC8
ADDR_RADIO = 0xEA
ADDR_TX_MODULE = 0xEE
ADDR_RECEIVER = 0xEC
MAX_FRAME = 64
MAX_PAYLOAD = MAX_FRAME - 4
NUM_CHANNELS = 16
FRAME_TIMEOUT_US = 1500

CH_MIN = 172
CH_MID = 992
CH_MAX = 1811

TYPE_GPS = 0x02
TYPE_BATTERY = 0x08
TYPE_LINK_STATS = 0x14
TYPE_RC_CHANNELS = 0x16
TYPE_ATTITUDE = 0x1E
TYPE_FLIGHT_MODE = 0x21

_SYNC_BYTES = (SYNC_FC, ADDR_TX_MODULE, ADDR_RADIO, ADDR_RECEIVER)


def _make_table() -> list[int]:
    table = []
    for b in range(256):
        crc = b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC_TABLE = _make_table()


def crc8_update(crc: int, byte: int) -> int:
    """Bitwise CRC-8/DVB-S2 step, identical to ``crsf_crc8_update``."""
    crc ^= byte & 0xFF
    for _ in range(8):
        crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def crc8(data: bytes | bytearray) -> int:
    """Table-driven CRC-8/DVB-S2 (equivalent to repeated :func:`crc8_update`)."""
    crc = 0
    for b in data:
        crc = _CRC_TABLE[crc ^ b]
    return crc


def pack_channels(ch) -> bytes:
    """16 x 11-bit channels -> 22 bytes, LSB first (``crsf_pack_channels``)."""
    if len(ch) != NUM_CHANNELS:
        raise ValueError("need 16 channels")
    acc = 0
    bits = 0
    out = bytearray()
    for v in ch:
        acc |= (int(v) & 0x7FF) << bits
        bits += 11
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    return bytes(out)


def unpack_channels(payload: bytes) -> list[int]:
    """22 bytes -> 16 channels (``crsf_unpack_channels``)."""
    acc = 0
    bits = 0
    ch: list[int] = []
    for b in payload[:22]:
        acc |= b << bits
        bits += 8
        while bits >= 11 and len(ch) < NUM_CHANNELS:
            ch.append(acc & 0x7FF)
            acc >>= 11
            bits -= 11
    return ch


def ch_to_us(v: float) -> float:
    return (v - CH_MID) * 0.625 + 1500.0


def us_to_ch(us: float) -> int:
    v = int(round((us - 1500.0) / 0.625 + CH_MID))
    return max(0, min(0x7FF, v))


def ch_to_norm(v: float) -> float:
    n = (v - CH_MID) / (CH_MAX - CH_MID)
    return -1.0 if n < -1.0 else (1.0 if n > 1.0 else n)


def norm_to_ch(n: float) -> int:
    """Inverse of :func:`ch_to_norm` (rounded to the nearest 11-bit code)."""
    n = -1.0 if n < -1.0 else (1.0 if n > 1.0 else n)
    return int(round(CH_MID + n * (CH_MAX - CH_MID)))


def build_frame(ftype: int, payload: bytes, sync: int = SYNC_FC) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload too long")
    body = bytes([ftype]) + bytes(payload)
    return bytes([sync, len(payload) + 2]) + body + bytes([crc8(body)])


def build_rc_channels(ch, sync: int = SYNC_FC) -> bytes:
    return build_frame(TYPE_RC_CHANNELS, pack_channels(ch), sync)


def _rad_to_crsf(r: float) -> int:
    v = r * 10000.0
    v = max(-32768.0, min(32767.0, v))
    return int(v + 0.5) if v >= 0 else int(v - 0.5)


def build_attitude(pitch: float, roll: float, yaw: float) -> bytes:
    return build_frame(TYPE_ATTITUDE, struct.pack(">hhh", _rad_to_crsf(pitch),
                                                  _rad_to_crsf(roll), _rad_to_crsf(yaw)))


def build_battery(voltage: float, current: float, mah: int, pct: int) -> bytes:
    dv = int(voltage * 10.0 + 0.5) & 0xFFFF
    da = int(current * 10.0 + 0.5) & 0xFFFF
    pl = struct.pack(">HH", dv, da) + bytes([(mah >> 16) & 0xFF, (mah >> 8) & 0xFF, mah & 0xFF,
                                             pct & 0xFF])
    return build_frame(TYPE_BATTERY, pl)


def build_flight_mode(mode: str) -> bytes:
    return build_frame(TYPE_FLIGHT_MODE, mode.encode("ascii")[:14] + b"\x00")


def build_link_stats(lq: int = 100, rssi_dbm: int = 40, snr: int = 10) -> bytes:
    pl = bytes([rssi_dbm & 0xFF, rssi_dbm & 0xFF, lq & 0xFF, snr & 0xFF, 0, 4, 2,
                rssi_dbm & 0xFF, lq & 0xFF, snr & 0xFF])
    return build_frame(TYPE_LINK_STATS, pl)


def decode_attitude(payload: bytes) -> tuple[float, float, float]:
    """Telemetry ATTITUDE payload -> (pitch, roll, yaw) in rad."""
    p, r, y = struct.unpack(">hhh", payload[:6])
    return p / 10000.0, r / 10000.0, y / 10000.0


def decode_flight_mode(payload: bytes) -> str:
    return payload.split(b"\x00", 1)[0].decode("ascii", "replace")


@dataclass
class Parser:
    """Byte-at-a-time CRSF parser, a port of ``crsf_parse_byte``."""

    buf: bytearray = field(default_factory=lambda: bytearray(MAX_FRAME))
    pos: int = 0
    last_byte_us: int = 0
    channels: list = field(default_factory=lambda: [CH_MID] * NUM_CHANNELS)
    last_rc_us: int = 0
    rc_fresh: bool = False
    frames_ok: int = 0
    crc_errors: int = 0
    framing_errors: int = 0
    last_type: int = 0
    last_payload: bytes = b""

    def _handle(self, now_us: int) -> None:
        ftype = self.buf[2]
        plen = self.buf[1] - 2
        pl = bytes(self.buf[3:3 + plen])
        self.last_type, self.last_payload = ftype, pl
        if ftype == TYPE_RC_CHANNELS and plen == 22:
            self.channels = unpack_channels(pl)
            self.last_rc_us = now_us
            self.rc_fresh = True

    def feed(self, byte: int, now_us: int) -> int:
        """Feed one byte; returns the frame type of a completed valid frame, else 0."""
        if self.pos > 0 and ((now_us - self.last_byte_us) & 0xFFFFFFFF) > FRAME_TIMEOUT_US:
            self.pos = 0
            self.framing_errors += 1
        self.last_byte_us = now_us
        if self.pos == 0:
            if byte in _SYNC_BYTES:
                self.buf[0] = byte
                self.pos = 1
            return 0
        if self.pos == 1:
            if byte < 2 or byte > MAX_FRAME - 2:
                self.pos = 0
                self.framing_errors += 1
                return 0
            self.buf[1] = byte
            self.pos = 2
            return 0
        self.buf[self.pos] = byte
        self.pos += 1
        total = self.buf[1] + 2
        if self.pos < total:
            return 0
        self.pos = 0
        if crc8(self.buf[2:total - 1]) != self.buf[total - 1]:
            self.crc_errors += 1
            return 0
        self.frames_ok += 1
        self._handle(now_us)
        return self.buf[2]

    def feed_bytes(self, data: bytes, now_us: int) -> list[int]:
        return [t for t in (self.feed(b, now_us) for b in data) if t]
