"""Closed-loop race session (the "world" side) and gate judge.

:class:`RaceSession` owns the simulator backend, the camera renderer and the
judge. Each 5 ms tick it produces what the companion would receive - IMU
samples, a camera frame when one is due (30 Hz), and the bytes the FC sent on
its UART - and it accepts the bytes the companion writes back, delivered to the
FC after ``cmd_delay`` (UART + scheduling latency). Ground truth is used for
scoring only. The same session runs in-process (:func:`run_race`) and as the
HIL server (:mod:`skyrace.link.hil`).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .so3 import quat_to_rot

DRONE_RADIUS = 0.12   # m, prop-tip radius used for gate clearance
TICK = 0.005


@dataclass
class GateJudge:
    track: object
    order: list = field(default_factory=list)
    idx: int = 0
    passes: list = field(default_factory=list)    # (gate index, t, local y, local z)
    misses: list = field(default_factory=list)
    crashed: str = ""
    t_finish: float | None = None

    def __post_init__(self):
        self.order = self.track.order()

    @property
    def done(self) -> bool:
        return self.idx >= len(self.order) or bool(self.crashed)

    def update(self, p0, p1, t0, t1):
        if self.done:
            return None
        g = self.track.gates[self.order[self.idx]]
        a, b = g.to_local(p0), g.to_local(p1)
        if a[0] < 0.0 <= b[0]:
            f = -a[0] / (b[0] - a[0])
            c = a + f * (b - a)
            tc = t0 + f * (t1 - t0)
            lim_in = 0.5 * g.inner - DRONE_RADIUS
            if abs(c[1]) <= lim_in and abs(c[2]) <= lim_in:
                self.passes.append((self.order[self.idx], float(tc), float(c[1]), float(c[2])))
                ev = "pass"
            elif abs(c[1]) <= 0.5 * g.outer + DRONE_RADIUS and abs(c[2]) <= 0.5 * g.outer + DRONE_RADIUS:
                self.crashed = f"hit gate {self.order[self.idx]} frame"
                return "crash"
            elif abs(c[1]) < 4.0 and abs(c[2]) < 4.0:
                self.misses.append((self.order[self.idx], float(tc)))
                ev = "miss"
            else:
                return None
            self.idx += 1
            if self.idx >= len(self.order):
                self.t_finish = float(tc)
            return ev
        return None


class RaceSession:
    def __init__(self, backend, renderer, track, camera_hz=30.0, cmd_delay=0.01, dt=TICK):
        self.backend, self.renderer, self.track = backend, renderer, track
        self.dt = dt
        self.camera_hz = camera_hz
        self.n_delay = int(round(cmd_delay / dt))
        backend.reset(pos=tuple(track.start), yaw=track.start_yaw)
        self.judge = GateJudge(track)
        self.q = deque()
        self.t = 0.0
        self.k = 0
        self.next_frame = 0.0
        self.airborne = False
        self.p_prev = backend.state()["pos"].copy()
        self.truth = {k: [] for k in ("t", "p", "v", "q", "w")}

    def truth_pose(self):
        st = self.backend.state()
        return st["pos"].copy(), quat_to_rot(st["q"])

    def sense(self):
        """(t, imu samples, image or None, uart bytes from the FC)."""
        st = self.backend.state()
        img = None
        if self.t >= self.next_frame - 1e-9:
            self.next_frame += 1.0 / self.camera_hz
            img = self.renderer.render(quat_to_rot(st["q"]), st["pos"], self.track.gates)
            img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float32) / 255.0  # 8-bit camera
        return self.t, self.backend.imu_samples(), img, self.backend.read_uart()

    def actuate(self, tx: bytes):
        self.q.append(tx)
        while len(self.q) > self.n_delay:
            self.backend.write_uart(self.q.popleft())

    def advance(self):
        self.backend.step(self.dt)
        self.t += self.dt
        self.k += 1
        st = self.backend.state()
        p = st["pos"].copy()
        if not st["on_ground"] and p[2] < -0.3:
            self.airborne = True
        if not self.judge.done:
            self.judge.update(self.p_prev, p, self.t - self.dt, self.t)
            if self.airborne and st["on_ground"] and not self.judge.crashed:
                self.judge.crashed = "hit the ground"
        self.p_prev = p
        tr = self.truth
        tr["t"].append(self.t)
        tr["p"].append(p)
        tr["v"].append(st["vel"].copy())
        tr["q"].append(st["q"].copy())
        tr["w"].append(st["w"].copy())
        return st

    def summary(self, t_race0, planned=None):
        j = self.judge
        n = len(j.order)
        lap = (j.t_finish - t_race0) if (j.t_finish is not None and t_race0 is not None) else None
        return dict(lap_time=lap, gates_passed=len(j.passes), gates_total=n, pass_rate=len(j.passes) / n,
                    misses=len(j.misses), crashed=j.crashed, passes=j.passes,
                    backend=getattr(self.backend, "name", "?"), planned_lap=planned, sim_s=self.t)


def run_race(backend, stack, renderer, track, t_max=40.0, cmd_delay=0.01, frame_every=4,
             keep_frames=True, verbose=False, camera_hz=30.0):
    """Fly one lap in-process. Returns (result dict, log, frames, judge)."""
    sess = RaceSession(backend, renderer, track, camera_hz, cmd_delay)
    stack.truth_fn = sess.truth_pose
    log = {k: [] for k in ("t", "p", "v", "p_est", "v_est", "p_ref", "T", "w_cmd", "w", "phase", "std")}
    frames = []
    wall0 = time.time()
    t_end = None
    while sess.t < t_max:
        t, imu, img, rx = sess.sense()
        tx = stack.tick(t, imu, img, rx)
        sess.actuate(tx)
        if img is not None and keep_frames and stack.phase in ("TAKEOFF", "RACE", "FINISH") and \
                stack.n_frames % frame_every == 0:
            frames.append(dict(t=t, img=(img * 255).astype(np.uint8),
                               meas=[(m["uv"], m["accepted"]) for m in stack.last_meas],
                               p=sess.backend.state()["pos"].copy(), p_est=stack.ekf.p.copy(),
                               phase=stack.phase))
        st = sess.advance()
        t = sess.t
        if sess.judge.done and t_end is None:
            t_end = t + 0.5     # the stack finishes on its *own* lap logic; we just stop logging
        if stack.phase in ("TAKEOFF", "RACE", "FINISH"):
            ref = stack.reference_pos(t)
            log["t"].append(t)
            log["p"].append(st["pos"].copy())
            log["v"].append(st["vel"].copy())
            log["p_est"].append(stack.ekf.p.copy())
            log["v_est"].append(stack.ekf.v.copy())
            log["p_ref"].append(ref if ref is not None else np.full(3, np.nan))
            log["T"].append(stack.last_cmd[0])
            log["w_cmd"].append(np.array(stack.last_cmd[1]))
            log["w"].append(st["w"].copy())
            log["phase"].append(stack.phase)
            log["std"].append(stack.ekf.pos_std)
        if stack.phase in ("KILLED", "DONE") or (t_end is not None and t >= t_end):
            break
        if stack.phase == "RACE" and t - stack.t_race0 > stack.race_timeout:
            sess.judge.crashed = sess.judge.crashed or "timeout"
            break
    log = {k: np.array(v) for k, v in log.items()}
    res = sess.summary(stack.t_race0, float(stack.ref_race.duration) if stack.ref_race is not None else None)
    race = log["phase"] == "RACE" if len(log["phase"]) else np.zeros(0, bool)
    if race.any():
        est_err = np.linalg.norm(log["p_est"][race] - log["p"][race], axis=1)
        trk = np.linalg.norm(log["p"][race] - log["p_ref"][race], axis=1)
        trk = trk[np.isfinite(trk)]
        res.update(est_err_rms=float(np.sqrt(np.mean(est_err ** 2))), est_err_max=float(est_err.max()),
                   track_err_rms=float(np.sqrt(np.mean(trk ** 2))) if len(trk) else None,
                   track_err_max=float(trk.max()) if len(trk) else None,
                   max_speed=float(np.linalg.norm(log["v"][race], axis=1).max()))
    res.update(controller=stack.controller_name, vision=stack.vision, frames_processed=stack.n_frames,
               gate_fixes=stack.n_fixes, ekf_rejected=stack.ekf.n_rejected,
               safety_events=list(stack.safety.events), wall_s=time.time() - wall0)
    if verbose:
        print({k: v for k, v in res.items() if k not in ("passes", "safety_events")})
    return res, log, frames, sess.judge
