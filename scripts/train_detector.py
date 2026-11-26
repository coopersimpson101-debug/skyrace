#!/usr/bin/env python3
"""Render a synthetic dataset, train GateNet (hand-written numpy backprop), save weights.

    python scripts/train_detector.py --n-train 8000 --epochs 8

Train renders use scene textures 0-5; the held-out set used by
scripts/eval_detector.py uses textures 6-7 and different random layouts.
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from skyrace.vision.dataset import make_dataset  # noqa: E402
from skyrace.vision.train import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", default=None, help="npz cache for the rendered training set")
    ap.add_argument("--out", default=os.path.join(ROOT, "skyrace", "vision", "weights", "gatenet.npz"))
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "train_log.json"))
    a = ap.parse_args()
    t0 = time.time()
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache)
        imgs, labs = z["imgs"], z["labs"]
    else:
        imgs, labs = make_dataset(a.n_train, seed0=1_000_000, scene_seeds=range(6), workers=a.workers)
        if a.cache:
            np.savez_compressed(a.cache, imgs=imgs, labs=labs)
    t_render = time.time() - t0
    print(f"dataset {imgs.shape} rendered in {t_render:.0f} s")
    w, hist = train(imgs, labs, epochs=a.epochs, batch=a.batch)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **w)
    n_params = int(sum(v.size for v in w.values()))
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "w") as f:
        json.dump(dict(n_train=int(len(imgs)), epochs=a.epochs, loss=hist, params=n_params,
                       render_s=round(t_render, 1), total_s=round(time.time() - t0, 1),
                       machine=f"{platform.processor() or platform.machine()}, {platform.system()}",
                       date=time.strftime("%Y-%m-%d")), f, indent=1)
    print(f"saved {a.out} ({n_params} params) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
