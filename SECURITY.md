# Security and safety policy

skyrace sends motor commands to a flight controller. Treat bugs that can cause
an unintended arm, a missed kill or a fly-away as **security issues**.

## Reporting

Please open a private security advisory on GitHub
(`Security` tab → `Report a vulnerability`) rather than a public issue. Include
the commit, the exact command, and a log or the CRSF byte stream if you have it.

## Scope

* `skyrace/link/` (CRSF encoding, serial bridge, HIL sockets, safety supervisor)
* the arming sequence and kill path in `skyrace/companion.py`
* anything that could make the FC arm with the arm switch low, or keep motors
  running after `Safety.kill()`

The HIL server binds to `localhost` by default and has no authentication; do
not expose it on a network you do not control.

## Operating safely

This is research software, untested on real hardware at v0.1.0. Fly only with
props off until the bring-up checklist in [docs/hardware.md](docs/hardware.md)
is complete, keep an independent kill switch, and follow local regulations.
