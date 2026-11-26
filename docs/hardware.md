# Companion computer, wiring and safety

> Status: ⏳ nothing in this document has been run on real hardware yet. The byte
> protocol it relies on is tested against the real skyforge-fc C parser in
> simulation (`tests/test_crsf.py`, `tests/test_closed_loop.py`).

## Wiring

The companion replaces the ELRS receiver on the FC's CRSF UART.

```
 companion (3.3 V UART)              skyforge-fc (CRSF UART, 420000 8N1)
   TX  ───────────────────────────►  RX
   RX  ◄───────────────────────────  TX   (telemetry: ATTITUDE, FLIGHT_MODE @ 10 Hz)
   GND ────────────────────────────  GND
```

| Board | Port | Pins | Notes |
|---|---|---|---|
| Raspberry Pi 5 | `/dev/ttyAMA0` | GPIO14 TX, GPIO15 RX | `dtparam=uart0=on`, disable the serial console |
| Jetson Orin Nano | `/dev/ttyTHS1` | header pins 8 / 10 | disable `nvgetty` |
| Orange Pi 5 | `/dev/ttyS4` | overlay `uart4-m0` | check your board revision's pinout |

420 kbaud is not a standard Linux rate on every UART driver; check that
`stty -F <port> 420000` is accepted (Pi 5 PL011 and Tegra HSUART accept it).

## Recommended safety architecture

1. **Independent kill.** Keep a second radio link that the companion cannot
   override: e.g. a small ELRS receiver on the companion's USB that feeds
   `Safety.operator_kill`, *and* a physical arm/kill switch or a relay on the
   ESC power if the field rules allow. Software kill on the companion alone is
   not enough for a first flight.
2. **FC-side failsafe is the last line.** If the companion crashes or the cable
   falls out, skyforge-fc stops receiving frames, after 300 ms levels itself in
   angle mode at `fs_throttle` (0.30) and disarms after 5 s
   (`test_failsafe_lands_then_disarms`). Lower `fs_throttle` below hover for a
   descending failsafe on your airframe.
3. **Companion supervisor** (`skyrace/link/safety.py`): geofence, estimator
   divergence, vision outage, telemetry loss, non-finite commands → LEVEL
   (angle mode, hover throttle) or KILL (arm channel low, latched).
4. **Props off** for every first test of a new build, and `motor` bench tests
   through the skyforge-fc CLI before any flight.

## Bring-up checklist (for the author)

1. Bench, props off: `python scripts/companion_main.py --port /dev/ttyAMA0 --bench`
   streams centred sticks with the arm switch low; check in the skyforge-fc CLI
   (`status`) that RX is OK and no RX_LOSS flag is set.
2. Arm-sequence check, props off: `--bench --arm` must show the FC arming (CLI
   `status`, flight mode telemetry `ACRO`) and disarming within one frame of
   the kill key.
3. Thrust identification on a stand: measure thrust vs throttle and set
   `AirframeModel.thrust_max` / `motor_idle` so `thrust_to_throttle` matches.
4. Camera calibration (`fx`, uptilt) and gate size: edit `CameraModel` and
   `skyrace/sim/track.py` (`INNER`, `OUTER`).
5. Tethered hover with the flatness tracker at 30 % of the planner limits.
6. Single gate, then the full track, increasing `Limits.thrust_max` stepwise.
