# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-26

### Added
- Python port of the skyforge-fc core (CRSF parser, rate curve, PID, mixer, DShot mapping,
  Madgwick AHRS, arming state machine, RX-loss failsafe) and a rigid-body quadrotor simulator
  with the same physics as skyforge-fc `sim/quad_sim.c`.
- SITL backend: a C shim that compiles the real skyforge-fc code and accepts raw CRSF bytes,
  plus conformance tests (CRC, channel packing, rate curve, mixer, DShot, PID) against it.
- Pinhole FPV camera renderer (textured ground, sky, anti-aliased gates).
- GateNet corner-heatmap CNN with hand-written numpy forward/backward pass and training,
  trained weights, a colour/hole-corner classical baseline, and held-out evaluation.
- Planar PnP (DLT homography + Levenberg-Marquardt + covariance) and an error-state EKF
  fusing the companion IMU with gate PnP fixes.
- Closed-form minimum-snap planner with feasibility-scaled time allocation, differential-flatness
  reference, flatness tracker, LTV-MPC with a hand-written ADMM QP solver, and a pure-pursuit baseline.
- CRSF command encoder (rates + thrust -> 16 channels), safety supervisor (arming, kill, geofence,
  estimator/vision/telemetry watchdogs), serial bridge, and lockstep TCP hardware-in-the-loop mode.
- Lap-time experiments, figures (top-down planned-vs-flown, lap GIF with camera inset), CI, CodeQL,
  release workflow.
