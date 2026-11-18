"""Candidate road-arc generation and per-candidate feature extraction.

Uses ONLY track + road network (never the reference route).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import point_segment

RADIUS_M = 80.0     # search radius around each GPS fix
MAX_K = 12          # keep at most K nearest arcs (two-way roads give 2 arcs each)
SAMPLE_STEP = 10.0  # densification step for the spatial index (m)

FEATURES = [
    "d", "d_rank", "d_minus_min", "d_ratio", "n_cand", "n_cand_30",
    "cos_head", "abs_head", "speed", "cos_head_x_speed",
    "t_proj", "t_outside", "arc_len", "log_arc_len",
    "d_prev1", "d_next1", "d_win5", "d_win15", "cos_head_win",
    "conn_prev", "conn_next", "same_prev", "same_next",
    "head_outdeg", "tail_indeg", "twoway", "gap_prev", "gap_next",
]


class ArcIndex:
    """KD-tree over densified arc samples (restricted to the track bbox)."""

    def __init__(self, trip, bbox_pad=RADIUS_M + 200):
        a, b = trip.arc_geometry()
        self.a, self.b = a, b
        lo = trip.xy.min(0) - bbox_pad
        hi = trip.xy.max(0) + bbox_pad
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        keep = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                        (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        L = np.linalg.norm(b[keep] - a[keep], axis=1)
        ns = np.maximum(1, np.ceil(L / SAMPLE_STEP).astype(int)) + 1
        arc_rep = np.repeat(keep, ns)
        frac = np.concatenate([np.linspace(0, 1, k) for k in ns])
        pts = a[arc_rep] + frac[:, None] * (b[arc_rep] - a[arc_rep])
        self.arc_rep = arc_rep
        self.tree = cKDTree(pts)

    def query(self, xy, radius=RADIUS_M, k_samples=96):
        dd, ii = self.tree.query(xy, k=k_samples, distance_upper_bound=radius + SAMPLE_STEP,
                                 workers=2)
        obs = np.repeat(np.arange(len(xy)), k_samples)
        ii = ii.ravel()
        ok = ii < len(self.arc_rep)
        df = pd.DataFrame({"obs": obs[ok], "arc": self.arc_rep[ii[ok]]}).drop_duplicates()
        return df["obs"].to_numpy(), df["arc"].to_numpy()


def _gps_heading(xy, t, half=2):
    n = len(xy)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    v = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.linalg.norm(v, axis=1) / dt
    return v, speed


def build_candidates(trip, radius=RADIUS_M, max_k=MAX_K, index=None):
    """Return a DataFrame with one row per (observation, candidate arc) and the
    feature columns in FEATURES. Original trip id and arc id are preserved."""
    xy, t = trip.xy, trip.track[:, 2]
    n = len(xy)
    idx = index or ArcIndex(trip)
    a, b = idx.a, idx.b
    obs, arc = idx.query(xy, radius)
    d, tc, traw = point_segment(xy[obs, 0], xy[obs, 1], a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1])
    df = pd.DataFrame({"obs": obs, "arc": arc, "d": d, "t_proj": tc,
                       "t_outside": np.abs(traw - tc)})
    df = df[df.d <= radius]
    df = df.sort_values(["obs", "d"], kind="mergesort")
    df["d_rank"] = df.groupby("obs").cumcount()
    df = df[df.d_rank < max_k].reset_index(drop=True)
    obs, arc = df.obs.to_numpy(), df.arc.to_numpy()

    g = df.groupby("obs")["d"]
    dmin = g.transform("min").to_numpy()
    df["d_minus_min"] = df.d - dmin
    df["d_ratio"] = df.d / (dmin + 1.0)
    df["n_cand"] = g.transform("size").to_numpy()
    df["n_cand_30"] = df.assign(c=(df.d <= 30)).groupby("obs")["c"].transform("sum").to_numpy()

    # heading agreement (GPS central-difference heading vs directed arc bearing)
    v, speed = _gps_heading(xy, t, 2)
    av = b[arc] - a[arc]
    alen = np.linalg.norm(av, axis=1)
    gv = v[obs]
    gn = np.linalg.norm(gv, axis=1)
    cosh = np.where((gn > 1e-6) & (alen > 1e-6),
                    (gv * av).sum(1) / np.maximum(gn * alen, 1e-9), 0.0)
    df["cos_head"] = cosh
    df["abs_head"] = np.arccos(np.clip(cosh, -1, 1))
    df["speed"] = speed[obs]
    df["cos_head_x_speed"] = cosh * np.minimum(speed[obs], 30) / 30
    v8, _ = _gps_heading(xy, t, 8)
    g8 = v8[obs]
    g8n = np.linalg.norm(g8, axis=1)
    df["cos_head_win"] = np.where((g8n > 1e-6) & (alen > 1e-6),
                                  (g8 * av).sum(1) / np.maximum(g8n * alen, 1e-9), 0.0)
    df["arc_len"] = alen
    df["log_arc_len"] = np.log1p(alen)

    # temporal context: distances of neighbouring fixes to the same arc
    def dist_to(off):
        j = np.clip(obs + off, 0, n - 1)
        dd, _, _ = point_segment(xy[j, 0], xy[j, 1], a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1])
        return dd
    df["d_prev1"] = dist_to(-1)
    df["d_next1"] = dist_to(1)
    df["d_win5"] = np.mean([dist_to(o) for o in (-5, -3, -1, 0, 1, 3, 5)], axis=0)
    df["d_win15"] = np.mean([dist_to(o) for o in (-15, -10, -5, 0, 5, 10, 15)], axis=0)

    # topology relative to neighbours' nearest arc (route-free)
    nearest = np.full(n, -1)
    first = df.d_rank.to_numpy() == 0
    nearest[obs[first]] = arc[first]
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]

    def conn(off):
        j = np.clip(obs + off, 0, n - 1)
        na = nearest[j]
        valid = na >= 0
        nas = np.where(valid, na, 0)
        same = valid & (nas == arc)
        touch = valid & ((frm[arc] == to[nas]) | (to[arc] == frm[nas]) |
                         (frm[arc] == frm[nas]) | (to[arc] == to[nas]))
        return same.astype(np.int8), (same | touch).astype(np.int8)
    df["same_prev"], df["conn_prev"] = conn(-1)
    df["same_next"], df["conn_next"] = conn(1)

    outdeg = np.bincount(frm, minlength=len(trip.nodes))
    indeg = np.bincount(to, minlength=len(trip.nodes))
    df["head_outdeg"] = outdeg[to[arc]]
    df["tail_indeg"] = indeg[frm[arc]]
    key = frm.astype(np.int64) * (len(trip.nodes) + 1) + to
    rkey = to.astype(np.int64) * (len(trip.nodes) + 1) + frm
    df["twoway"] = np.isin(rkey[arc], key).astype(np.int8)
    dtp = np.diff(t, prepend=t[0])
    dtn = np.diff(t, append=t[-1])
    df["gap_prev"] = dtp[obs]
    df["gap_next"] = dtn[obs]
    df.insert(0, "trip", trip.trip_id)
    df["d_rank"] = df["d_rank"].astype(np.int16)
    for c in FEATURES:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df
