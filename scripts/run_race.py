#!/usr/bin/env python3
"""Closed-loop lap-time experiments (vision -> EKF -> planning/control -> CRSF -> FC -> physics).

    python scripts/run_race.py                 # the full sweep used for the README (~1 h here)
    python scripts/run_race.py --quick         # one MPC lap with CNN vision

Backend: the real skyforge-fc C code (SITL library) if it has been built,
else the Python port. The simulated airframe deliberately differs from what
the companion believes (mass +8 %, thrust -5 %, drag +33 %).
"""
import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUE_AIRFRAME = dict(mass=0.70, thrust_max=9.5, drag=0.20)   # simulated quad (companion assumes 0.65 / 10 / 0.15)


def make_backend(kind, seed):
    from skyrace import sitl
    from skyrace.sim.quad import QuadParams, QuadSim
    a = TRUE_AIRFRAME
    if kind == "sitl" or (kind == "auto" and sitl.available()):
        return sitl.SkyforgeSITL(seed=seed, airframe=(a["mass"], a["thrust_max"], a["drag"]))
    return QuadSim(QuadParams(mass=a["mass"], thrust_max=a["thrust_max"], drag_lin=a["drag"]), seed=seed)


_PLANS = {}


def plan(frac):
    from skyrace.companion import plan_race
    from skyrace.planning.minsnap import Limits
    from skyrace.sim.track import make_track
    if frac not in _PLANS:
        lim = Limits(thrust_max=frac * 40.0)
        traj, hist = plan_race(make_track("loop7"), lim)
        _PLANS[frac] = (traj, lim, hist)
    return _PLANS[frac]


def job_key(job):
    ctl, level, vision, seed, backend, _ = job
    return f"{ctl}_{level}_{vision}_s{seed}_{backend}"


def run_cached(job):
    """Run a job unless results/runs/<key>.json exists (so partial sweeps can be resumed)."""
    path = os.path.join(ROOT, "results", "runs", job_key(job) + ".json")
    if os.path.exists(path) and not job[5]:
        with open(path) as f:
            return json.load(f), None
    res, extra = run_one(job)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res, extra


def run_one(job):
    from skyrace.companion import CompanionStack
    from skyrace.planning.minsnap import Limits
    from skyrace.race import run_race
    from skyrace.sim.camera import CameraModel, Renderer
    from skyrace.sim.track import make_track
    ctl, level, vision, seed, backend, keep = job
    tr = make_track("loop7")
    cam = CameraModel()
    if ctl == "pursuit":
        traj, lim = None, Limits()
        stack = CompanionStack(tr, cam, controller="pursuit", vision=vision, pursuit_speed=level)
    else:
        traj, lim, _ = plan(level)
        stack = CompanionStack(tr, cam, controller=ctl, race_traj=traj, lim=lim, vision=vision)
    be = make_backend(backend, seed)
    res, log, frames, judge = run_race(be, stack, Renderer(cam), tr, keep_frames=keep, frame_every=2)
    res.update(level=level, seed=seed, job=list(job))
    res["passes"] = [list(map(float, p)) for p in res["passes"]]
    res["safety_events"] = [list(e) for e in res["safety_events"]]
    extra = None
    if keep:
        extra = dict(log=log, frames=frames, ref=None if traj is None else dict(
            t=stack.ref_race.t, p=stack.ref_race.p, T=stack.ref_race.T, w=stack.ref_race.w))
    print(f"{ctl:8s} {level:5} {vision:9s} seed {seed}: lap {res['lap_time']}, "
          f"gates {res['gates_passed']}/{res['gates_total']} {res['crashed']} ({res['wall_s']:.0f} s)", flush=True)
    return res, extra


def save_extra(path, extra):
    log, frames, ref = extra["log"], extra["frames"], extra["ref"]
    d = {f"log_{k}": v for k, v in log.items() if k != "phase"}
    d["log_phase"] = np.array([str(x) for x in log["phase"]])
    if ref is not None:
        d.update({f"ref_{k}": v for k, v in ref.items()})
    d["frame_t"] = np.array([f["t"] for f in frames])
    d["frame_img"] = np.array([f["img"] for f in frames])
    d["frame_p"] = np.array([f["p"] for f in frames])
    d["frame_p_est"] = np.array([f["p_est"] for f in frames])
    uv = np.full((len(frames), 8, 4, 2), np.nan)
    acc = np.zeros((len(frames), 8), bool)
    for i, f in enumerate(frames):
        for j, (u, a) in enumerate(f["meas"][:8]):
            uv[i, j] = u
            acc[i, j] = a
    d["frame_uv"] = uv
    d["frame_acc"] = acc
    np.savez_compressed(path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "sitl", "python"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--only-vision", default=None, help="run only jobs with this vision mode (cache them)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "race_results.json"))
    ap.add_argument("--hero", default=os.path.join(ROOT, "results", "hero_run.npz"))
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.time()
    if a.quick:
        b = a.backend
        if b == "auto":
            from skyrace import sitl
            b = "sitl" if sitl.available() else "python"
        res, extra = run_one(("mpc", 0.6, "cnn", 1, b, False))
        print(json.dumps({k: v for k, v in res.items() if k not in ("passes",)}, indent=1, default=str))
        return
    levels = {"minsnap": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "mpc": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "pursuit": [6.0, 8.0, 10.0, 12.0, 14.0]}
    jobs = [(c, lv, "cnn", s, a.backend, False) for c, lvs in levels.items() for lv in lvs for s in a.seeds]
    jobs += [("mpc", lv, v, 1, a.backend, False) for v in ("truth", "classical") for lv in levels["mpc"]]
    jobs += [("minsnap", 0.6, v, 1, a.backend, False) for v in ("truth", "classical")]
    jobs += [("mpc", 0.6, "cnn", 1, "python", False)]
    if a.backend == "auto":
        from skyrace import sitl
        a.backend = "sitl" if sitl.available() else "python"
        jobs = [(c, lv, v, s, a.backend if b == "auto" else b, k) for (c, lv, v, s, b, k) in jobs]
    if a.only_vision:
        jobs = [j for j in jobs if j[2] == a.only_vision]
        with Pool(a.workers) as pool:
            pool.map(run_cached, jobs, chunksize=1)
        return
    for lv in levels["mpc"]:
        plan(lv)            # same deterministic plans the workers compute; recorded in the summary
    with Pool(a.workers) as pool:
        out = pool.map(run_cached, jobs, chunksize=1)
    results = [r for r, _ in out]
    # hero: the fastest clean MPC lap with CNN vision, re-run keeping logs and camera frames
    clean = [r for r in results if r["job"][0] == "mpc" and r["job"][2] == "cnn" and r["pass_rate"] == 1.0
             and not r["crashed"] and r["job"][4] == a.backend]
    hero = min(clean, key=lambda r: r["lap_time"]) if clean else None
    if hero is not None:
        j = list(hero["job"])
        j[5] = True
        _, extra = run_one(tuple(j))
        save_extra(a.hero, extra)
    summary = dict(machine=f"{platform.processor()}, {platform.system()}", date=time.strftime("%Y-%m-%d"),
                   true_airframe=TRUE_AIRFRAME, planned_laps={str(k): v[0].total for k, v in _PLANS.items()},
                   hero_job=hero["job"] if hero else None, wall_s=round(time.time() - t0, 1), runs=results)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
