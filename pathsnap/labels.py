"""Observation labels derived from the reference route.

The .route file lists arcs in travel order but has no per-observation
alignment. We derive one label per GPS observation with a *monotone
route-constrained alignment* (a DTW-style dynamic programme):

    choose j_1 <= j_2 <= ... <= j_n (indices into the route sequence)
    minimising sum_i dist(obs_i, route_arc[j_i])

Monotonicity respects the chronological order of the route (so loops and
repeated arcs are handled correctly) and arcs may be skipped (short arcs
traversed between two 1 Hz fixes). Observations whose aligned distance
exceeds OFFROUTE_M are flagged unreliable and excluded from training and
from observation-level metrics (their count is reported).

This module is ONLY used by training-label construction and the evaluation
step. The prediction pipeline (features.py / hmm.py / api predict) never
imports it.
"""
import numpy as np

from .geometry import point_segment

OFFROUTE_M = 60.0


def align_route(xy, node_xy, arcs, route):
    n, m = len(xy), len(route)
    a = node_xy[arcs[route, 0]]
    b = node_xy[arcs[route, 1]]
    back = np.empty((n, m), dtype=np.int32)
    prev = None
    ar = np.arange(m)
    dist_rows = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        d, _, _ = point_segment(xy[i, 0], xy[i, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        dist_rows[i] = d
        if prev is None:
            cur = d.astype(np.float64)
            back[i] = ar
        else:
            cm = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev == cm, ar, 0))
            cur = d + cm
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    idx = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        idx[i] = j
        j = back[i, j]
    lab_arc = route[idx]
    lab_dist = dist_rows[np.arange(n), idx]
    return lab_arc, idx, lab_dist


def observation_labels(trip):
    lab_arc, ridx, lab_dist = align_route(trip.xy, trip.node_xy, trip.arcs, trip.route)
    reliable = lab_dist <= OFFROUTE_M
    return lab_arc, ridx, lab_dist, reliable
