"""Server-side evaluation step (laptop 1 only).

Runs AFTER prediction is complete. It is the only API code that reads the
private reference route. Nothing computed here is fed back into the
prediction, and reference geometry/labels are never returned to the UI --
only aggregate scores.
"""
import numpy as np
from sklearn.metrics import f1_score

from pathsnap.data import load_route
from pathsnap.labels import align_route
from pathsnap.metrics import collapse, continuity, route_overlap


def evaluate_prediction(trip, cls_pred, hmm_pred, route, graph, partition, data_root=None):
    ref = load_route(trip.trip_id, partition, data_root)
    if ref is None or len(ref) == 0:
        return {"available": False, "reason": "no private reference route on server for this trip"}
    lab, _, ldist = align_route(trip.xy, trip.node_xy, trip.arcs, ref)
    rel = ldist <= 60.0
    ro_h = route_overlap(route, ref, graph.alen)
    ro_c = route_overlap(collapse(cls_pred), ref, graph.alen)
    y = lab[rel]
    return {
        "available": True,
        "n_observations": int(len(lab)),
        "n_scored_observations": int(rel.sum()),
        "observation_top1_accuracy_hmm": float((hmm_pred[rel] == y).mean()),
        "observation_top1_accuracy_classifier_only": float((cls_pred[rel] == y).mean()),
        "observation_macro_f1_hmm": float(f1_score(y, hmm_pred[rel], average="macro", zero_division=0)),
        "observation_macro_f1_classifier_only": float(f1_score(y, cls_pred[rel], average="macro", zero_division=0)),
        "route_continuity_hmm": continuity(hmm_pred, trip.arcs),
        "route_continuity_classifier_only": continuity(cls_pred, trip.arcs),
        "route_precision_hmm": ro_h["route_precision"],
        "route_recall_hmm": ro_h["route_recall"],
        "route_f1_hmm": ro_h["route_f1"],
        "route_f1_classifier_only": ro_c["route_f1"],
        "reference_route_length_km": float(graph.alen[ref].sum() / 1000),
    }
