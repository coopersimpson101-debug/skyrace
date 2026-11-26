#!/usr/bin/env python3
"""Companion-computer bring-up tool for a real skyforge-fc (PROPS OFF).

    python scripts/companion_main.py --port /dev/ttyAMA0 --bench          # centred sticks, disarmed
    python scripts/companion_main.py --port /dev/ttyAMA0 --bench --arm    # arm at zero throttle, Enter = kill

It writes CRSF RC frames at 200 Hz through skyrace.link.bridge.SerialUart and
prints the FC telemetry (flight mode / attitude). Full autonomous flight needs
camera + IMU drivers for your board, which are not part of v0.1.0 (see the
README roadmap); the stack itself is the same CompanionStack used in the HIL
tests. Untested on hardware so far.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyrace.fcport import crsf  # noqa: E402
from skyrace.link.bridge import PORTS, SerialUart  # noqa: E402
from skyrace.link.commands import CommandEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORTS["pi5"])
    ap.add_argument("--bench", action="store_true", help="required: props-off bench mode")
    ap.add_argument("--arm", action="store_true", help="run the arm sequence (zero throttle)")
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    if not a.bench:
        sys.exit("only --bench mode exists in v0.1.0 (props off!)")
    uart = SerialUart(a.port)
    enc = CommandEncoder()
    parser = crsf.Parser()
    kill = threading.Event()
    threading.Thread(target=lambda: (input("press Enter to KILL\n"), kill.set()), daemon=True).start()
    t0 = time.monotonic()
    nxt = t0
    mode = ""
    while time.monotonic() - t0 < a.seconds and not kill.is_set():
        t = time.monotonic() - t0
        arm = a.arm and t > 1.5          # arm switch must be seen low first
        uart.write(enc.frame((0, 0, 0), 0.0, arm=arm, throttle=0.0))
        for b in uart.read():
            if parser.feed(b, int(t * 1e6)) == crsf.TYPE_FLIGHT_MODE:
                m = crsf.decode_flight_mode(parser.last_payload)
                if m != mode:
                    print(f"{t:6.2f} s  FC mode {m}")
                    mode = m
        nxt += 0.005
        time.sleep(max(0.0, nxt - time.monotonic()))
    for _ in range(20):
        uart.write(enc.frame((0, 0, 0), 0.0, arm=False, throttle=0.0))
        time.sleep(0.005)
    print("disarm frames sent; CRSF frames ok/crc errors from FC:", parser.frames_ok, parser.crc_errors)
    uart.close()


if __name__ == "__main__":
    main()
