"""Arming sequence and kill-switch / safety supervision on the companion computer.

The flight controller already enforces its own rules (skyforge-fc
``fc_pid_step``): it only arms on an OFF->ON edge of the arm channel with
throttle below ``arm_max_throttle``, tilt below ``arm_max_angle_deg`` and a
finished gyro calibration; it disarms immediately when the arm channel goes
low; and if RC frames stop for ``fs_timeout_ms`` (300 ms) it levels itself
at ``fs_throttle`` and disarms after ``fs_land_time_s``.

On top of that the companion runs this supervisor. Actions, in increasing
severity:

* ``LEVEL`` - stop autonomous commands, put the FC in ANGLE mode with sticks
  centred and hover throttle (the FC self-levels on its own AHRS)
* ``KILL``  - arm channel low on every frame from now on (latched; motors stop
  on the next frame the FC parses). Re-arming needs :meth:`reset` *and* the
  FC's own arm-switch cycle.

Triggers: operator kill (keyboard / GPIO / spare RC channel), geofence,
estimator divergence (position std), vision outage during the race, FC
telemetry loss, a non-finite controller output, or a persistent LEVEL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DISARMED, ARMING, ARMED, KILLED = "DISARMED", "ARMING", "ARMED", "KILLED"
OK, LEVEL, KILL = "OK", "LEVEL", "KILL"


@dataclass
class SafetyConfig:
    geofence_min: tuple = (-15.0, -12.0, -9.0)
    geofence_max: tuple = (35.0, 35.0, 0.5)
    max_pos_std: float = 1.5          # m, estimator health
    vision_timeout: float = 2.0       # s without an accepted gate fix while racing
    telemetry_timeout: float = 0.6    # s without CRSF telemetry from the FC
    level_to_kill: float = 3.0        # s in LEVEL before escalating to KILL
    arm_low_time: float = 0.2         # s of arm-switch-low before the arm edge
    arm_timeout: float = 1.0          # s to see the FC report armed


@dataclass
class Safety:
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    state: str = DISARMED
    action: str = OK
    reason: str = ""
    t_arm_start: float = 0.0
    t_level: float = -1.0
    operator_kill: bool = False
    events: list = field(default_factory=list)

    def reset(self):
        self.state, self.action, self.reason = DISARMED, OK, ""
        self.operator_kill = False
        self.t_level = -1.0

    def kill(self, t: float, reason: str):
        if self.state != KILLED:
            self.events.append((t, "KILL", reason))
        self.state, self.action, self.reason = KILLED, KILL, reason

    def request_arm(self, t: float):
        if self.state == DISARMED:
            self.state = ARMING
            self.t_arm_start = t
            self.events.append((t, "ARMING", ""))

    def arm_switch(self, t: float, fc_armed: bool) -> bool:
        """Arm-channel value to send this frame."""
        if self.state == KILLED:
            return False
        if self.state == ARMING:
            if fc_armed:
                self.state = ARMED
                self.events.append((t, "ARMED", ""))
                return True
            dt = t - self.t_arm_start
            if dt > self.cfg.arm_low_time + self.cfg.arm_timeout:
                self.kill(t, "FC did not arm (check arming flags)")
                return False
            return dt >= self.cfg.arm_low_time
        return self.state == ARMED

    def check(self, t, est_p, pos_std, racing: bool, t_last_fix: float, t_last_telem: float,
              cmd=None) -> str:
        """Run every tick; returns OK / LEVEL / KILL."""
        c = self.cfg
        if self.state == KILLED:
            return KILL
        if self.operator_kill:
            self.kill(t, "operator kill switch")
        elif self.state == ARMED and (np.any(np.asarray(est_p) < c.geofence_min) or
                                      np.any(np.asarray(est_p) > c.geofence_max)):
            self.kill(t, "geofence")
        elif self.state in (ARMING, ARMED) and t - t_last_telem > c.telemetry_timeout:
            self.kill(t, "FC telemetry lost")
        elif cmd is not None and not np.all(np.isfinite(np.r_[cmd[0], cmd[1]])):
            self.kill(t, "non-finite controller output")
        if self.state == KILLED:
            return KILL
        level_reason = self.reason if self.action == LEVEL else ""   # LEVEL is latched
        if level_reason:
            pass
        elif pos_std > c.max_pos_std:
            level_reason = "estimator diverged"
        elif racing and t - t_last_fix > c.vision_timeout:
            level_reason = "vision outage"
        if level_reason:
            if self.t_level < 0:
                self.t_level = t
                self.events.append((t, "LEVEL", level_reason))
            if t - self.t_level > c.level_to_kill:
                self.kill(t, level_reason + " (persistent)")
                return KILL
            self.action, self.reason = LEVEL, level_reason
            return LEVEL
        self.action = OK
        return OK
