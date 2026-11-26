import ctypes
import random

import numpy as np

from conftest import needs_sitl
from skyrace.fcport import crsf


def test_crc8_dvb_s2_known_vector():
    # CRC-8/DVB-S2 check value for "123456789" is 0xBC
    assert crsf.crc8(b"123456789") == 0xBC
    c = 0
    for b in b"123456789":
        c = crsf.crc8_update(c, b)
    assert c == 0xBC


def test_pack_unpack_roundtrip():
    rng = random.Random(0)
    for _ in range(200):
        ch = [rng.randrange(0, 2048) for _ in range(16)]
        pl = crsf.pack_channels(ch)
        assert len(pl) == 22
        assert crsf.unpack_channels(pl) == ch


def test_rc_frame_layout():
    f = crsf.build_rc_channels([992] * 16)
    assert f[0] == 0xC8 and f[1] == 24 and f[2] == 0x16 and len(f) == 26
    assert crsf.crc8(f[2:-1]) == f[-1]


def test_parser_split_garbage_crc_and_timeout():
    ch = list(range(172, 172 + 16 * 50, 50))
    f = crsf.build_rc_channels(ch)
    p = crsf.Parser()
    assert p.feed_bytes(b"\x00\x13\x37", 0) == []
    assert p.feed_bytes(f[:10], 10) == []
    assert p.feed_bytes(f[10:], 20) == [crsf.TYPE_RC_CHANNELS]
    assert p.channels == ch and p.frames_ok == 1
    bad = bytearray(f)
    bad[5] ^= 0xFF
    assert p.feed_bytes(bytes(bad), 100) == []
    assert p.crc_errors == 1
    # partial frame, then > 1.5 ms of silence: dropped, and the next frame parses
    p.feed_bytes(f[:7], 1000)
    assert p.feed_bytes(f, 5000) == [crsf.TYPE_RC_CHANNELS]
    assert p.framing_errors >= 1


def test_telemetry_roundtrip():
    f = crsf.build_attitude(0.1, -0.2, 3.0)
    p = crsf.Parser()
    assert p.feed_bytes(f, 0) == [crsf.TYPE_ATTITUDE]
    pi, ro, ya = crsf.decode_attitude(p.last_payload)
    assert abs(pi - 0.1) < 1e-4 and abs(ro + 0.2) < 1e-4 and abs(ya - 3.0) < 1e-4
    p.feed_bytes(crsf.build_flight_mode("ACRO*"), 0)
    assert crsf.decode_flight_mode(p.last_payload) == "ACRO*"


@needs_sitl
def test_matches_skyforge_c(sitl_lib):
    rng = np.random.default_rng(1)
    for _ in range(100):
        data = bytes(rng.integers(0, 256, int(rng.integers(1, 60))).astype(np.uint8))
        assert sitl_lib.sitl_crc8(data, len(data)) == crsf.crc8(data)
        ch = [int(x) for x in rng.integers(0, 2048, 16)]
        out = ctypes.create_string_buffer(22)
        sitl_lib.sitl_pack((ctypes.c_uint16 * 16)(*ch), out)
        assert out.raw == crsf.pack_channels(ch)
        back = (ctypes.c_uint16 * 16)()
        sitl_lib.sitl_unpack(out.raw, back)
        assert list(back) == ch
    for v in range(0, 2048, 37):
        assert abs(sitl_lib.sitl_ch_to_norm(v) - crsf.ch_to_norm(v)) < 1e-6
