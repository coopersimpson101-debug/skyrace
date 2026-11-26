#!/usr/bin/env python3
"""Held-out evaluation of GateNet vs the classical colour baseline.

Held-out renders: scene textures 6 and 7 (training used 0-5), new random gate
layouts and poses (seeds disjoint from training). Metrics:

* corner level, for labelled corners closer than 15 m: recall (a detection of
  the same corner type within 4 px), median / 90th-percentile pixel error of
  the matched detections, false positives per image (detections > 4 px from
  every labelled corner of their type, ignoring corners further than 22 m);
* gate level, for gates with all 4 corners in view and closer than 12 m:
  PnP success using the association step of the closed loop (prior = truth
  perturbed by 0.15 m / 2 deg), and the body-position error of that fix.

    python scripts/eval_detector.py --n 600
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from skyrace.sim.camera import CameraModel  # noqa: E402
from skyrace.so3 import exp_so3  # noqa: E402
from skyrace.vision.dataset import random_layout, random_view  # noqa: E402
from skyrace.vision.detector import CornerDetector, measure_gate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_heldout(seed):
    from skyrace.vision.dataset import render_sample
    rng = np.random.default_rng(seed)
    gates = random_layout(rng, int(rng.integers(3, 7)))
    img, lab, R, p = render_sample(seed, scene_seeds=(6, 7), gates=gates)
    return img.astype(np.float32) / 255.0, lab, R, p, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "detector_eval.json"))
    a = ap.parse_args()
    cam = CameraModel()
    dets = {"cnn": CornerDetector("cnn"), "classical": CornerDetector("classical")}
    stats = {k: dict(err=[], tp=0, n_lab=0, fp=0, t=[], g_n=0, g_ok=0, g_perr=[], g_dist=[], g_rerr=[]) for k in dets}
    rng = np.random.default_rng(12345)
    for i in range(a.n):
        img, lab, R, p, gates = render_heldout(9_000_000 + i)
        for name, det in dets.items():
            s = stats[name]
            t0 = time.perf_counter()
            cands = det(img)
            s["t"].append(time.perf_counter() - t0)
            for c in range(4):
                cc = cands[c]
                used = np.zeros(len(cc), bool)
                for g in range(lab.shape[0]):
                    u, v, vis, z = lab[g, c]
                    if vis < 0.5 or z > 15.0:
                        continue
                    s["n_lab"] += 1
                    if len(cc):
                        d = np.hypot(cc[:, 0] - u, cc[:, 1] - v)
                        j = int(np.argmin(d))
                        if d[j] <= 4.0:
                            s["tp"] += 1
                            s["err"].append(float(d[j]))
                            used[j] = True
                for j in range(len(cc)):
                    if used[j]:
                        continue
                    near = [np.hypot(cc[j, 0] - lab[g, c, 0], cc[j, 1] - lab[g, c, 1]) <= 4.0
                            for g in range(lab.shape[0]) if lab[g, c, 2] > 0.5 and lab[g, c, 3] <= 22.0]
                    far = [np.hypot(cc[j, 0] - lab[g, c, 0], cc[j, 1] - lab[g, c, 1]) <= 4.0
                           for g in range(lab.shape[0]) if lab[g, c, 2] > 0.5 and lab[g, c, 3] > 22.0]
                    if not any(near) and not any(far):
                        s["fp"] += 1
            for g, gate in enumerate(gates):
                if lab[g, :, 2].min() < 0.5 or lab[g, :, 3].max() > 12.0:
                    continue
                s["g_n"] += 1
                R_prior = R @ exp_so3(rng.normal(0, np.radians(2), 3))
                p_prior = p + rng.normal(0, 0.15, 3)
                m = measure_gate(cands, cam, gate, R_prior, p_prior, typed=det.typed)
                if m is not None:
                    s["g_ok"] += 1
                    s["g_perr"].append(float(np.linalg.norm(m["p"] - p)))
                    dtrue = float(np.linalg.norm(p - gate.center))
                    s["g_dist"].append(dtrue)
                    s["g_rerr"].append(abs(m["dist"] - dtrue))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}", {k: round(v["tp"] / max(v["n_lab"], 1), 3) for k, v in stats.items()})
    out = dict(n_images=a.n, heldout_textures=[6, 7], machine=f"{platform.processor()}, {platform.system()}",
               date=time.strftime("%Y-%m-%d"), methods={})
    for name, s in stats.items():
        e = np.array(s["err"])
        pe = np.array(s["g_perr"])
        out["methods"][name] = dict(
            corner_recall_4px=round(s["tp"] / max(s["n_lab"], 1), 4), corners_labelled=s["n_lab"],
            median_px_err=round(float(np.median(e)), 3) if len(e) else None,
            p90_px_err=round(float(np.percentile(e, 90)), 3) if len(e) else None,
            false_pos_per_image=round(s["fp"] / a.n, 3),
            gates_evaluated=s["g_n"], pnp_success=round(s["g_ok"] / max(s["g_n"], 1), 4),
            pnp_pos_err_median_m=round(float(np.median(pe)), 4) if len(pe) else None,
            pnp_pos_err_p90_m=round(float(np.percentile(pe, 90)), 4) if len(pe) else None,
            ms_per_frame=round(1000 * float(np.median(s["t"])), 1),
            pnp_by_distance={},
        )
        dd, re_ = np.array(s["g_dist"]), np.array(s["g_rerr"])
        for lo, hi in ((0, 4), (4, 8), (8, 12)):
            k = (dd >= lo) & (dd < hi)
            if k.any():
                out["methods"][name]["pnp_by_distance"][f"{lo}-{hi} m"] = dict(
                    n=int(k.sum()), pos_err_median_m=round(float(np.median(pe[k])), 3),
                    range_err_median_m=round(float(np.median(re_[k])), 3))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["methods"], indent=1))


if __name__ == "__main__":
    main()
