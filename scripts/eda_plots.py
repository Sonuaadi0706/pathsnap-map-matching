"""EDA figures (partition stated in each title). Output: figures/eda_*.png"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap import plotstyle as ps  # noqa: E402
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.dataset import load_partition  # noqa: E402

ps.apply()
plt = ps.plt
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
eda = pd.read_csv(ROOT / "results/eda_trips.csv", dtype={"trip": str})
tr = eda[eda.partition == "train"]
te = eda[eda.partition == "student_test"]

# 1. observations per trip
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bins = np.logspace(np.log10(200), np.log10(20000), 25)
ax.hist(tr.n_obs, bins=bins, color=ps.INK, edgecolor="white", label="train (80)")
ax.hist(te.n_obs, bins=bins, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="student_test (10)")
ax.set_xscale("log")
ax.set_xlabel("GPS observations per trip (log scale)")
ax.set_ylabel("trips")
ax.set_title("Trip length distribution (train + student_test)")
ax.legend()
fig.savefig(FIG / "eda_obs_per_trip.png")
plt.close(fig)

# 2. sampling intervals (all trips)
dts, steps, speeds = [], [], []
for _, r in eda.iterrows():
    t = load_trip(r.trip, r.partition, with_route=False)
    dt = np.diff(t.track[:, 2])
    dts.append(dt)
    st = np.linalg.norm(np.diff(t.xy, axis=0), axis=1)
    steps.append(st)
    speeds.append(st[dt <= 1.5] / dt[dt <= 1.5])
dts = np.concatenate(dts)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
vals, cnts = np.unique(np.clip(dts, 0, 31), return_counts=True)
axes[0].bar(vals, cnts, color=ps.INK, width=0.8)
axes[0].set_yscale("log")
axes[0].set_xlabel("time between consecutive fixes (s, 31 = 31+)")
axes[0].set_ylabel("count (log)")
axes[0].set_title("Sampling interval (all 90 trips)")
sp = np.concatenate(speeds)
axes[1].hist(np.clip(sp, 0, 60), bins=60, color=ps.MID, edgecolor="white")
axes[1].set_xlabel("apparent speed between 1 s fixes (m/s, clipped at 60)")
axes[1].set_ylabel("count")
axes[1].set_title("Apparent speed (all 90 trips)")
fig.tight_layout()
fig.savefig(FIG / "eda_sampling.png")
plt.close(fig)

# 3. GPS trajectories + road network for 6 training trips
sel = ["00000002", "00000086", "00000063", "00000075", "00000092", "00000085"]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, tid in zip(axes.ravel(), sel):
    t = load_trip(tid, "train", with_route=True)
    a, b = t.arc_geometry()
    lo, hi = t.xy.min(0) - 150, t.xy.max(0) + 150
    m = ((np.minimum(a, b) >= lo) & (np.maximum(a, b) <= hi)).all(1)
    segs = np.stack([a[m], b[m]], 1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=ps.LIGHT, linewidths=0.4))
    r = t.route
    ax.add_collection(LineCollection(np.stack([a[r], b[r]], 1), colors=ps.INK, linewidths=1.6, zorder=4))
    ax.scatter(t.xy[:, 0], t.xy[:, 1], s=4, color="#9a9a9a", alpha=0.5, lw=0, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"train {tid}  ({t.n_obs} fixes, {len(t.arcs)} arcs)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(False)
fig.suptitle("GPS fixes (grey dots), road network (light lines), reference route (black) - train partition, metres",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "eda_trajectories.png")
plt.close(fig)

# 4. road network size / connectivity
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for g, c, mk, lab in [(tr, ps.INK, "o", "train"), (te, ps.MID, "s", "student_test")]:
    axes[0].scatter(g.n_nodes, g.n_arcs, s=18, color=c, marker=mk, label=lab)
    axes[1].scatter(g.n_arcs, g.largest_scc_frac, s=18, color=c, marker=mk, label=lab)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs (directed)")
axes[0].set_title("Road-network size per trip"); axes[0].legend()
axes[1].set_xscale("log")
axes[1].set_xlabel("arcs (log)"); axes[1].set_ylabel("largest strongly-connected comp. (frac. of nodes)")
axes[1].set_title("Connectivity per trip"); axes[1].legend()
fig.tight_layout()
fig.savefig(FIG / "eda_network.png")
plt.close(fig)

# 5. label-alignment distance (GPS noise proxy) + candidate structure (train)
c, o = load_partition("train")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].hist(np.clip(o.lab_dist, 0, 100), bins=100, color=ps.INK)
axes[0].axvline(60, color=ps.MID, ls="--", lw=1.2)
axes[0].annotate("off-route threshold 60 m", (61, axes[0].get_ylim()[1] * 0.8), fontsize=8, color=ps.MID)
axes[0].set_yscale("log")
axes[0].set_xlabel("fix to aligned reference arc (m, clipped at 100)")
axes[0].set_ylabel("observations (log)")
axes[0].set_title("Alignment distance (train)")
pos = c[(c.y == 1) & c.reliable]
rk = pos.d_rank.value_counts().sort_index()
axes[1].bar(rk.index[:12], rk.values[:12] / len(pos), color=ps.INK)
axes[1].set_xlabel("distance rank among candidates (0 = nearest)")
axes[1].set_ylabel("share of observations")
axes[1].set_title("Rank of reference arc (train)")
rel = c[c.reliable]
axes[2].hist(rel[rel.y == 0].cos_head, bins=40, density=True, color=ps.LIGHT, edgecolor=ps.MID, label="other candidates")
axes[2].hist(rel[rel.y == 1].cos_head, bins=40, density=True, histtype="step", color=ps.INK, lw=2, label="reference arc")
axes[2].set_xlabel("cos(GPS heading - arc bearing)")
axes[2].set_ylabel("density")
axes[2].set_title("Heading agreement (train)")
axes[2].legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "eda_labels_candidates.png")
plt.close(fig)

# 6. data-quality summary table -> csv for the report
q = pd.DataFrame({
    "check": ["trips", "GPS observations", "NaN values in .track / .nodes", "non-increasing timestamps",
              "fraction of 1 s steps (median trip)", "trips with a gap > 30 s", "max single gap (s)",
              "isolated nodes", "self-loop arcs", "zero-length arcs", "route arcs head->tail continuous",
              "observations flagged off-route (>60 m)", "candidate miss rate (reliable obs)"],
    "train": [len(tr), int(tr.n_obs.sum()), int(tr.nan_track.sum() + tr.nan_nodes.sum()),
              int((~tr.ts_monotonic).sum()), round(tr.frac_dt1.median(), 4), int((tr.n_gaps_gt30s > 0).sum()),
              int(tr.max_dt.max()), int(tr.isolated_nodes.sum()), int(tr.self_loops.sum()),
              int(tr.zero_len_arcs.sum()), round(tr.route_continuity.mean(), 4), int((~o.reliable).sum()),
              round(1 - o[o.reliable].in_cands.mean(), 4)],
})
c2, o2 = load_partition("student_test")
q["student_test"] = [len(te), int(te.n_obs.sum()), int(te.nan_track.sum() + te.nan_nodes.sum()),
                     int((~te.ts_monotonic).sum()), round(te.frac_dt1.median(), 4), int((te.n_gaps_gt30s > 0).sum()),
                     int(te.max_dt.max()), int(te.isolated_nodes.sum()), int(te.self_loops.sum()),
                     int(te.zero_len_arcs.sum()), round(te.route_continuity.mean(), 4), int((~o2.reliable).sum()),
                     round(1 - o2[o2.reliable].in_cands.mean(), 4)]
q.to_csv(ROOT / "results/eda_quality_table.csv", index=False)
print(q.to_string())
print("rank0 share", rk.values[0] / len(pos))
