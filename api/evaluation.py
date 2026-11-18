"""Server-side evaluation step (laptop 1 only).

Runs AFTER prediction is complete. It is the only API code that reads the
private reference route. Nothing computed here is fed back into the
prediction, and reference geometry/labels are never returned to the UI --
only aggregate scores.
"""
from sklearn.metrics import f1_score

from pathsnap.data import load_route
from pathsnap.labels import OFFROUTE_M, align_route
from pathsnap.metrics import collapse, continuity, route_overlap


def evaluate_prediction(trip, cls_pred, hmm_pred, final_pred, route, graph, partition, data_root=None):
    ref = load_route(trip.trip_id, partition, data_root)
    if ref is None or len(ref) == 0:
        return {"available": False, "reason": "no private reference route on server for this trip"}
    lab, _, ldist = align_route(trip.xy, trip.node_xy, trip.arcs, ref)
    rel = ldist <= OFFROUTE_M
    y = lab[rel]
    ro = route_overlap(route, ref, graph.alen)
    ro_c = route_overlap(collapse(cls_pred), ref, graph.alen)
    stages = {}
    for name, p in [("classifier_only", cls_pred), ("hmm", hmm_pred), ("final", final_pred)]:
        stages[name] = {
            "observation_top1_accuracy": float((p[rel] == y).mean()),
            "observation_macro_f1": float(f1_score(y, p[rel], average="macro", zero_division=0)),
            "route_continuity": continuity(p, trip.arcs),
            "observation_top1_accuracy_all_fixes": float((p == lab).mean()),
        }
    return {
        "available": True,
        "n_observations": int(len(lab)),
        "n_scored_observations": int(rel.sum()),
        "scoring_note": f"fixes farther than {OFFROUTE_M:.0f} m from the aligned reference route are not scored",
        "stages": stages,
        "observation_top1_accuracy": stages["final"]["observation_top1_accuracy"],
        "route_precision": ro["route_precision"],
        "route_recall": ro["route_recall"],
        "route_f1": ro["route_f1"],
        "route_f1_classifier_only": ro_c["route_f1"],
        "reference_route_length_km": float(graph.alen[ref].sum() / 1000),
    }
