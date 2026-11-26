"""The companion-computer stack: vision -> estimation -> planning/control -> CRSF bytes.

:class:`CompanionStack` is a pure function of its inputs - time, IMU samples,
an optional camera frame and the bytes received from the FC UART - and
returns the bytes to write to the FC UART. The same object runs

* in-process against a simulator backend (:func:`skyrace.race.run_race`),
* in a separate process over TCP in hardware-in-the-loop mode
  (:mod:`skyrace.link.hil`), and
* on a real Pi 5 / Jetson / Orange Pi with a serial port and a camera
  (``scripts/companion_main.py``, untested on hardware here).

Mission phases: PRE (disarmed frames while the FC calibrates its gyro) ->
ARMING -> TAKEOFF (min-snap climb to 1.5 m) -> HOLD -> RACE -> FINISH
(FC angle mode, hover throttle, then disarm) -> DONE.
"""
from __future__ import annotations

import numpy as np

from .control.controllers import FlatnessTracker, MpcTracker, PurePursuit, accel_to_command
from .estimation.ekf import GateEkf
from .fcport import crsf
from .fcport.fc import FcConfig
from .link.commands import AirframeModel, CommandEncoder
from .link.safety import ARMED, KILL, LEVEL, Safety, SafetyConfig
from .planning.minsnap import Limits, allocate_times, min_snap
from .planning.reference import Reference
from .so3 import euler_to_rot
from .vision.detector import CornerDetector, measure_gate, predicted_corners

HOVER_ALT = 1.5


def race_waypoints(track, hover_alt=HOVER_ALT):
    start = track.start + np.array([0.0, 0.0, -hover_alt])
    return np.array([start] + [track.gates[i].center for i in track.order()])


def pursuit_path(track, approach=2.0, hover_alt=HOVER_ALT):
    """Polyline for the pure-pursuit baseline: straight through every gate along its normal."""
    pts = [track.start + np.array([0.0, 0.0, -hover_alt])]
    for i in track.order():
        g = track.gates[i]
        pts += [g.center - approach * g.normal, g.center, g.center + approach * g.normal]
    return np.array(pts)


def plan_race(track, lim: Limits, log=None):
    W = race_waypoints(track)
    traj, hist = allocate_times(W, lim, log=log)
    return traj, hist


class CompanionStack:
    def __init__(self, track, cam, controller: str = "mpc", race_traj=None, lim: Limits | None = None,
                 vision: str = "cnn", fc_cfg: FcConfig | None = None, airframe: AirframeModel | None = None,
                 pursuit_speed: float = 8.0, safety_cfg: SafetyConfig | None = None,
                 t_takeoff: float = 1.6, mass: float = 0.65, drag: float = 0.15,
                 truth_fn=None, mpc_hz: float = 50.0, ekf_update: str = "corners"):
        self.track, self.cam = track, cam
        self.lim = lim or Limits()
        self.enc = CommandEncoder(fc_cfg or FcConfig(), airframe)
        self.safety = Safety(safety_cfg or SafetyConfig())
        self.m, self.c = mass, drag
        self.ekf = GateEkf(track.start.copy(), euler_to_rot(0, 0, track.start_yaw))
        self.vision = vision
        self.ekf_update = ekf_update
        self.detector = CornerDetector(vision) if vision in ("cnn", "classical") else None
        self.truth_fn = truth_fn  # only for the "truth" ablation (no vision)
        self.parser = crsf.Parser()
        self.fc_armed = False
        self.fc_mode = ""
        self.t_last_telem = 0.0
        self.t_last_fix = 0.0
        self.phase = "PRE"
        self.t_phase = 0.0
        self.t_takeoff = t_takeoff
        # takeoff: pad -> hover point in 2.5 s (min-snap, single segment, end at rest)
        W = race_waypoints(track)
        to = min_snap(np.array([track.start, W[0]]), [2.5], end_fixed=True)
        self.ref_takeoff = Reference(to, mass, drag, yaw0=track.start_yaw, hold_yaw=True)
        self.takeoff_ctl = FlatnessTracker(self.ref_takeoff, mass, drag)
        self.race_traj = race_traj
        self.controller_name = controller
        if controller in ("mpc", "minsnap"):
            if race_traj is None:
                race_traj, _ = plan_race(track, self.lim)
                self.race_traj = race_traj
            self.ref_race = Reference(race_traj, mass, drag, yaw0=track.start_yaw)
            self.race_ctl = MpcTracker(self.ref_race, mass, drag) if controller == "mpc" \
                else FlatnessTracker(self.ref_race, mass, drag)
        elif controller == "pursuit":
            self.ref_race = None
            self.race_ctl = PurePursuit(pursuit_path(track), speed=pursuit_speed, mass=mass, drag=drag,
                                        yaw0=track.start_yaw)
        else:
            raise ValueError(controller)
        self.t_race0 = None
        self.ctl_dt = 1.0 / mpc_hz if controller == "mpc" else 0.0
        self.next_ctl = 0.0
        self.held_cmd = (0.0, np.zeros(3))
        self.race_timeout = (self.ref_race.duration + 4.0) if self.ref_race is not None else 40.0
        self.order = track.order()
        self.progress = 0          # gates crossed according to the *estimate*
        self.p_est_prev = self.ekf.p.copy()
        self.last_cmd = (0.0, np.zeros(3))
        self.last_meas = []
        self.n_frames = 0
        self.n_fixes = 0

    # ----------------------------------------------------------- telemetry
    def _rx(self, data: bytes, t: float):
        for b in data:
            ft = self.parser.feed(b, int(t * 1e6))
            if ft == crsf.TYPE_FLIGHT_MODE:
                self.fc_mode = crsf.decode_flight_mode(self.parser.last_payload)
                self.fc_armed = not self.fc_mode.endswith("*") and self.fc_mode != "!FS!"
                self.t_last_telem = t
            elif ft == crsf.TYPE_ATTITUDE:
                self.t_last_telem = t

    # ---------------------------------------------------------------- vision
    def _vision(self, t, img):
        self.n_frames += 1
        self.last_meas = []
        e = self.ekf
        if self.vision == "truth":
            p, R = self.truth_fn()
            cov = np.diag([0.05 ** 2] * 3 + [np.radians(1) ** 2] * 3)
            ok, _ = e.update_pose(p, R, cov)
            if ok:
                self.t_last_fix = t
            return
        cands = self.detector(img)
        self.last_cands = cands
        for gate in self.track.gates:
            pred, mirrored, z = predicted_corners(self.cam, e.R, e.p, gate)
            if not np.all(np.isfinite(pred)) or z.max() > 25.0:
                continue
            inside = np.all((pred[:, 0] > -2) & (pred[:, 0] < self.cam.width + 2) &
                            (pred[:, 1] > -2) & (pred[:, 1] < self.cam.height + 2))
            if not inside:
                continue
            radius = 8.0 + 3.0 * e.pos_std * self.cam.fx / max(z.mean(), 1.0)
            m = measure_gate(cands, self.cam, gate, e.R, e.p, radius=min(radius, 30.0), typed=self.detector.typed)
            if m is None:
                continue
            if self.ekf_update == "pnp":
                ok, d2 = e.update_pose(m["p"], m["R"], m["cov"])
            else:   # default: tightly coupled corner reprojection (PnP only validates the association)
                ok, d2 = e.update_corners(m["uv"], m["X_w"], self.cam.R_bc, self.cam.K)
            m["accepted"] = ok
            self.last_meas.append(m)
            if ok:
                self.t_last_fix = t
                self.n_fixes += 1

    # ---------------------------------------------------------------- main tick
    def tick(self, t: float, imu, img=None, rx: bytes = b"") -> bytes:
        self._rx(rx, t)
        for (ti, g, a) in imu:
            dt = ti - self.ekf.t if self.ekf.t > 0 else 1.0 / 400.0
            self.ekf.propagate(g, a, max(min(dt, 0.02), 1e-4))
            self.ekf.t = ti
        if self.phase in ("PRE", "ARMING"):
            self.ekf.update_zero_velocity()
        if img is not None and self.vision != "none":
            self._vision(t, img)
        est = self.estimate()
        self._track_progress(t, est["p"])
        racing = self.phase == "RACE"
        if self.phase in ("PRE", "ARMING") and self.vision != "none":
            self.t_last_fix = t
        act = self.safety.check(t, est["p"], self.ekf.pos_std, racing, self.t_last_fix, self.t_last_telem,
                                self.last_cmd)
        arm = self.safety.arm_switch(t, self.fc_armed)
        if act == KILL:
            self.phase = "KILLED" if self.phase not in ("DONE",) else self.phase
            return self.enc.frame((0, 0, 0), 0.0, arm=False, throttle=0.0)
        if act == LEVEL or self.phase == "FINISH":
            hover = self.enc.air.thrust_to_throttle(self.m * 9.80665)
            if self.phase == "FINISH" and t - self.t_phase > 1.5:
                self.phase = "DONE"
                self.safety.kill(t, "race finished")
            return self.enc.frame((0, 0, 0), 0.0, arm=arm, angle_mode=True, throttle=hover * 0.97)
        if self.phase == "PRE":
            if t >= self.t_takeoff - 0.4 and t - self.t_last_telem < 0.3:
                self.safety.request_arm(t)
                self._set_phase("ARMING", t)
            return self.enc.frame((0, 0, 0), 0.0, arm=False, throttle=0.0)
        if self.phase == "ARMING":
            if self.safety.state == ARMED:
                self._set_phase("TAKEOFF", t)
                self.ref_takeoff.t_offset = t
            return self.enc.frame((0, 0, 0), 0.0, arm=arm, throttle=0.0)
        if self.phase == "TAKEOFF":
            T, w = self.takeoff_ctl.command(t, est)
            if t - self.t_phase > self.ref_takeoff.duration + 0.6:
                self._set_phase("RACE", t)
                self.t_race0 = t
                self.t_last_fix = max(self.t_last_fix, t - 0.5)
                if self.ref_race is not None:
                    self.ref_race.t_offset = t
        elif self.phase == "RACE":
            if t >= self.next_ctl - 1e-9:     # MPC at ctl_hz, the others every tick
                self.next_ctl = t + self.ctl_dt
                self.held_cmd = self.race_ctl.command(t, est)
            T, w = self.held_cmd
        else:
            T, w = 0.0, np.zeros(3)
        self.last_cmd = (T, w)
        return self.enc.frame(w, T, arm=arm)

    def _track_progress(self, t, p):
        """Own lap logic (no ground truth on a real drone): count estimated gate-plane crossings."""
        if self.phase == "RACE" and self.progress < len(self.order):
            g = self.track.gates[self.order[self.progress]]
            a, b = g.to_local(self.p_est_prev), g.to_local(p)
            if a[0] < 0.0 <= b[0] and abs(b[1]) < 3.0 and abs(b[2]) < 3.0:
                self.progress += 1
                if self.progress == len(self.order):
                    self.finish(t)
        self.p_est_prev = p.copy()

    def finish(self, t):
        """Called by the race judge (or the stack's own lap logic) after the last gate."""
        if self.phase == "RACE":
            self._set_phase("FINISH", t)

    def _set_phase(self, ph, t):
        self.phase = ph
        self.t_phase = t

    def estimate(self):
        e = self.ekf
        return dict(p=e.p.copy(), v=e.v.copy(), R=e.R.copy(), w=e.last_w.copy(), a=e.last_a.copy())

    def reference_pos(self, t):
        if self.phase == "RACE" and self.ref_race is not None:
            return self.ref_race.at(t)["p"]
        if self.phase == "TAKEOFF":
            return self.ref_takeoff.at(t)["p"]
        return None


__all__ = ["CompanionStack", "plan_race", "race_waypoints", "accel_to_command", "MpcTracker",
           "PurePursuit"]
