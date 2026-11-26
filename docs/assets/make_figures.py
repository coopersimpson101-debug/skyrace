#!/usr/bin/env python3
"""Regenerate every figure in the README from results/ (no numbers are typed in by hand).

    python docs/assets/make_figures.py

Inputs : results/race_results.json, results/hero_run.npz (written by scripts/run_race.py, 3 MB,
         not committed), results/detector_eval.json,
         results/train_log.json
Outputs: docs/assets/{hero_topdown.png, hero_lap.gif, lap_times.png, detections.png,
         banner.svg}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from skyrace.sim.track import make_track  # noqa: E402

ACCENT = "#F43F5E"
BG = "#0d1117"
PANEL = "#161b22"
INK = "#e6edf3"
INK2 = "#8b949e"
GRID = "#30363d"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
CTL_COL = {"minsnap": BLUE, "mpc": ACCENT, "pursuit": AQUA}
CTL_NAME = {"minsnap": "min-snap + flatness tracker", "mpc": "min-snap + MPC", "pursuit": "pure pursuit (baseline)"}
RES = os.path.join(ROOT, "results")


def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)


def gate_segments(track):
    segs = []
    for g in track.gates:
        h = 0.5 * g.outer
        a = g.center + g.R @ np.array([0, -h, 0])
        b = g.center + g.R @ np.array([0, h, 0])
        segs.append((a, b, g))
    return segs


def draw_gates(ax, track, label=True):
    for i, (a, b, g) in enumerate(gate_segments(track)):
        ax.plot([a[1], b[1]], [a[0], b[0]], color=INK, lw=3.2, solid_capstyle="butt", zorder=5)
        if label:
            n = g.normal
            ax.annotate(f"G{i + 1}", (g.center[1] - 1.6 * n[0] * 0 + 1.2 * n[1] * 0, g.center[0]),
                        xytext=(g.center[1] - 1.9 * g.R[1, 1], g.center[0] - 1.9 * g.R[0, 1]),
                        color=INK2, fontsize=8, ha="center", va="center", zorder=6)


def hero_topdown(h, res, track):
    lg = {k[4:]: h[k] for k in h.files if k.startswith("log_")}
    race = lg["phase"] == "RACE"
    p, pe, pr, v = lg["p"][race], lg["p_est"][race], lg["p_ref"][race], lg["v"][race]
    t = lg["t"][race]
    fig = plt.figure(figsize=(9, 9.2), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    style(ax)
    ax.plot(h["ref_p"][:, 1], h["ref_p"][:, 0], color=INK2, lw=1.6, ls=(0, (5, 3)), label="planned (min-snap, time-allocated)",
            zorder=3)
    spd = np.linalg.norm(v, axis=1)
    pts = np.c_[p[:, 1], p[:, 0]][:, None, :]
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap="magma", linewidths=2.4, zorder=4)
    lc.set_array(spd[:-1])
    lc.set_clim(0, max(spd.max(), 1))
    ax.add_collection(lc)
    ax.plot(pe[:, 1], pe[:, 0], color=BLUE, lw=1.0, alpha=0.9, label="onboard estimate (EKF: IMU + CNN gate corners)", zorder=4)
    ax.plot([], [], color=ACCENT, lw=2.4, label="flown (simulator truth, colour = speed)")
    draw_gates(ax, track)
    ax.plot(track.start[1], track.start[0], marker="s", ms=8, color=AQUA, zorder=6, ls="none", label="start pad")
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("speed (m/s)", color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, framealpha=0.95)
    leg.set_zorder(10)
    ax.set_title(f"Closed loop in simulation: lap {res['lap_time']:.2f} s, {res['gates_passed']}/{res['gates_total']} gates, "
                 f"top speed {res['max_speed']:.1f} m/s", fontsize=11, loc="left", color=INK)
    ax2 = fig.add_subplot(gs[1])
    style(ax2)
    t0 = t[0]
    ax2.plot(t - t0, np.linalg.norm(p - pr, axis=1), color=ACCENT, lw=1.6, label="tracking error |flown - planned|")
    ax2.plot(t - t0, np.linalg.norm(pe - p, axis=1), color=BLUE, lw=1.6, label="estimation error |estimate - truth|")
    for gp in res["passes"]:
        ax2.axvline(gp[1] - t0, color=GRID, lw=1.0, zorder=0)
    ax2.set_xlabel("race time (s)   (vertical lines: gate passes)")
    ax2.set_ylabel("m")
    ax2.legend(loc="upper right", fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, ncol=2)
    ax2.set_xlim(0, t[-1] - t0)
    ax2.set_ylim(0, 1.45 * max(np.linalg.norm(p - pr, axis=1).max(), np.linalg.norm(pe - p, axis=1).max()))
    fig.text(0.01, 0.005, f"Simulated ({res['backend']} backend, CNN vision). Machine: {RJ['machine']}, {RJ['date']}.",
             color=INK2, fontsize=7.5)
    fig.savefig(os.path.join(HERE, "hero_topdown.png"), dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def lap_times(rj):
    runs = [r for r in rj["runs"] if r["job"][2] == "cnn" and r["job"][4] != "python"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), facecolor=BG, gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    for ax, key, title, ylab in ((axs[0], "lap_time", "Lap time vs planner aggressiveness", "lap time (s)"),
                                 (axs[1], "track_err_rms", "Tracking error (flown vs planned)", "RMS position error (m)")):
        style(ax)
        for ctl in ("minsnap", "mpc"):
            rr = [r for r in runs if r["job"][0] == ctl]
            for r in rr:
                x = r["level"] * 100 + (-1.2 if ctl == "minsnap" else 1.2)
                ok = r["pass_rate"] == 1.0 and not r["crashed"]
                ax.plot(x, r[key] if r[key] is not None else np.nan, marker="o" if ok else "x", ms=8,
                        color=CTL_COL[ctl], ls="none", mew=2)
            ax.plot([], [], marker="o", color=CTL_COL[ctl], ls="none", label=CTL_NAME[ctl])
        ax.set_xlabel("planner thrust limit (% of 40 N)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        if key == "track_err_rms":
            ax.set_ylim(0, None)
    axs[0].legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax = axs[2]
    style(ax)
    rr = [r for r in runs if r["job"][0] == "pursuit"]
    for r in rr:
        ok = r["pass_rate"] == 1.0 and not r["crashed"]
        if ok:
            ax.plot(r["level"], r["lap_time"], marker="o", ms=8, color=AQUA, ls="none")
    bad = sorted(set(r["level"] for r in rr if not (r["pass_rate"] == 1.0 and not r["crashed"])))
    for L in bad:
        ax.annotate("crash", (L, 1.2), color=INK2, fontsize=8, ha="center")
        ax.plot(L, 0.5, marker="x", ms=9, color=AQUA, mew=2, ls="none")
    ax.set_xlabel("cruise speed (m/s)")
    ax.set_ylabel("lap time (s)")
    ax.set_title("Baseline: pure pursuit", fontsize=10.5, loc="left", color=INK)
    ax.set_ylim(0, None)
    fig.text(0.01, -0.03, "3 seeds per setting (points overlap), CNN vision, skyforge-fc C code in the loop (SITL), "
             f"airframe mismatch on. Simulated, {rj['date']}.", color=INK2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lap_times.png"), dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def hero_gif(h, track, res, max_frames=78):
    imgs, ts, ps, uvs, acc = h["frame_img"], h["frame_t"], h["frame_p"], h["frame_uv"], h["frame_acc"]
    lg_p = h["log_p"]
    W, Hh = 600, 300
    # map panel (left, 300x300): east -> x, north -> -y
    allp = np.r_[h["ref_p"], lg_p]
    e0, e1 = allp[:, 1].min() - 3, allp[:, 1].max() + 3
    n0, n1 = allp[:, 0].min() - 3, allp[:, 0].max() + 3
    s = min(290 / (e1 - e0), 290 / (n1 - n0))

    def m(pt):
        return (5 + (pt[1] - e0) * s, 295 - (pt[0] - n0) * s)

    base = Image.new("RGB", (W, Hh), BG)
    d = ImageDraw.Draw(base)
    ref = [m(q) for q in h["ref_p"][::4]]
    d.line(ref, fill=(110, 118, 129), width=1)
    for a, b, g in gate_segments(track):
        d.line([m(a), m(b)], fill=(230, 237, 243), width=3)
    t_race0 = h["log_t"][h["log_phase"] == "RACE"][0]
    sel = np.nonzero(ts >= t_race0 - 0.4)[0]
    idx = sel[np.linspace(0, len(sel) - 1, min(max_frames, len(sel))).astype(int)]
    out = []
    t_log = h["log_t"]
    for i in idx:
        fr = base.copy()
        dd = ImageDraw.Draw(fr)
        k = int(np.searchsorted(t_log, ts[i]))
        trail = [m(q) for q in lg_p[max(0, k - 400):k + 1:3]]
        if len(trail) > 1:
            dd.line(trail, fill=(244, 63, 94), width=2)
        x, y = m(ps[i])
        dd.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(244, 63, 94))
        cam = Image.fromarray(imgs[i]).resize((288, 216), Image.BILINEAR)
        cd = ImageDraw.Draw(cam)
        sc = 288 / imgs.shape[2]
        for j in range(uvs.shape[1]):
            if not np.all(np.isfinite(uvs[i, j])):
                continue
            col = (25, 200, 120) if acc[i, j] else (255, 200, 0)
            q = uvs[i, j] * sc
            cd.line([tuple(q[0]), tuple(q[1]), tuple(q[2]), tuple(q[3]), tuple(q[0])], fill=col, width=2)
        fr.paste(cam, (306, 42))
        dd.rectangle([305, 41, 306 + 288, 42 + 216], outline=(48, 54, 61))
        dd.text((306, 8), "onboard camera + CNN gate fixes", fill=(230, 237, 243))
        dd.text((306, 22), f"race time {ts[i] - t_race0:5.2f} s", fill=(139, 148, 158))
        dd.text((306, 264), "green: gate corners fused by the EKF", fill=(139, 148, 158))
        dd.text((306, 278), f"simulated lap {res['lap_time']:.2f} s", fill=(139, 148, 158))
        out.append(fr.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    path = os.path.join(HERE, "hero_lap.gif")
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 * np.median(np.diff(ts[idx]))),
                loop=0, optimize=True)
    return path


def detections_figure():
    from skyrace.vision.classical import detect_corners
    from skyrace.vision.detector import CornerDetector
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_detector import render_heldout
    det = CornerDetector("cnn")
    seeds = [9_000_003, 9_000_011, 9_000_017, 9_000_024]
    fig, axs = plt.subplots(2, 4, figsize=(12, 4.6), facecolor=BG)
    cols = ["#ff5c7a", "#3987e5", "#19c37d", "#f5b000"]
    for c, sd in enumerate(seeds):
        img, lab, *_ = render_heldout(sd)
        for r, (name, cands) in enumerate((("GateNet CNN", det(img)), ("colour baseline", detect_corners(img)))):
            ax = axs[r, c]
            ax.imshow(img)
            for k in range(4):
                if len(cands[k]):
                    ax.plot(cands[k][:, 0] - 0.5, cands[k][:, 1] - 0.5, "o", mfc="none", mec=cols[k], ms=6, mew=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if c == 0:
                ax.set_ylabel(name, color=INK, fontsize=10)
    fig.text(0.01, 0.0, "Held-out renders (textures not seen in training). Circles: detected corners, colour = type "
             "(TL red, TR blue, BR green, BL yellow).", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "detections.png"), dpi=100, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def banner(h, track):
    """1280x400 SVG; motif = the real planned (dashed) and flown (accent) hero lap, top-down."""
    ref = h["ref_p"][::6]
    lg = h["log_p"][h["log_phase"] == "RACE"][::6]
    allp = np.r_[ref, lg]
    e0, e1, n0, n1 = allp[:, 1].min(), allp[:, 1].max(), allp[:, 0].min(), allp[:, 0].max()
    s = min(520 / (e1 - e0), 330 / (n1 - n0))
    ox, oy = 760, 35

    def m(q):
        return ox + (q[1] - e0) * s, oy + 330 - (q[0] - n0) * s

    def path(pts):
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in map(m, pts))

    gates = []
    for a, b, g in gate_segments(track):
        (x1, y1), (x2, y2) = m(a), m(b)
        gates.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e6edf3" '
                     f'stroke-width="5" stroke-linecap="butt" opacity="0.9"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400">
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0 L0 0 0 32" fill="none" stroke="#161b22" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="1280" height="400" fill="#0d1117"/>
  <rect width="1280" height="400" fill="url(#grid)"/>
  <text x="64" y="190" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="92" font-weight="700" fill="#e6edf3">skyrace</text>
  <text x="68" y="240" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="22" fill="#8b949e">autonomous FPV racing on your own flight controller</text>
  <text x="68" y="274" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="17" fill="#8b949e">CNN gates · PnP + EKF · min-snap · MPC · CRSF to skyforge-fc</text>
  <rect x="68" y="300" width="56" height="4" fill="{ACCENT}"/>
  <path d="{path(ref)}" fill="none" stroke="#8b949e" stroke-width="2" stroke-dasharray="7 5" opacity="0.8"/>
  <path d="{path(lg)}" fill="none" stroke="{ACCENT}" stroke-width="3.2" stroke-linejoin="round"/>
  {"".join(gates)}
  <text x="1216" y="384" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13" fill="#8b949e">simulated lap: planned (dashed) vs flown</text>
</svg>
'''
    with open(os.path.join(HERE, "banner.svg"), "w", encoding="utf-8") as f:
        f.write(svg)


if __name__ == "__main__":
    RJ = json.load(open(os.path.join(RES, "race_results.json")))
    h = np.load(os.path.join(RES, "hero_run.npz"))
    track = make_track("loop7")
    hero = [r for r in RJ["runs"] if r["job"] == RJ["hero_job"]][0]
    hero_topdown(h, hero, track)
    lap_times(RJ)
    print("gif:", hero_gif(h, track, hero))
    banner(h, track)
    detections_figure()
    print("figures written to", HERE)
