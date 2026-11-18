"""Scorer-only vs scorer + HMM/Viterbi (+ route re-alignment) on the 10 student-test trips, for every model.

Every model's saved student-test scores (results/test_scores_<model>.pkl, 46-feature candidate table) are
decoded with the SAME deployed decoder setting (models/final_config.json). Nothing is tuned here: the
decoder parameters were fixed on training trips (Section 5 of the report), so this is a pure comparison.

usage: python scripts/hmm_ablation.py
outputs: results/hmm_ablation_models.csv (one row per model), results/hmm_test_pertrip.csv (model x trip)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_route, load_trip  # noqa: E402
from pathsnap.dataset import CACHE  # noqa: E402
from pathsnap.hmm import HMMParams, decode, realign_to_route, reconstruct_route  # noqa: E402
from pathsnap.metrics import collapse, continuity, route_overlap  # noqa: E402

RES = ROOT / "results"
CFG = json.load(open(ROOT / "models" / "final_config.json"))
PARAMS = HMMParams(**CFG["hmm_params"])
MODELS = ["baseline", "logreg", "gnb", "knn", "rf", "lgbm", "mlp", CFG["model"]]


def trip_eval(tid, c):
    """Scorer-only, Viterbi and re-aligned metrics of one student-test trip for one model's scores."""
    tr = load_trip(tid, "student_test", with_route=False)
    f = pd.read_pickle(CACHE / f"feat_student_test_{tid}.pkl")[["obs", "arc", "t_proj"]]
    c = c.merge(f, on=["obs", "arc"])
    lab = np.load(CACHE / f"lab_student_test_{tid}.npz")
    rel, y = lab["reliable"], lab["lab"]
    cls = np.full(tr.n_obs, -1, dtype=np.int64)
    b = c.loc[c.groupby("obs").pn.idxmax()]
    cls[b.obs.to_numpy()] = b.arc.to_numpy()
    t0 = time.time()
    pred, seg, g = decode(tr, c, PARAMS)
    route, nfail = reconstruct_route(pred, seg, g)
    final = realign_to_route(tr, pred, route, PARAMS.realign_max_m) if CFG.get("realign", True) else pred
    t_dec = time.time() - t0
    ref = load_route(tid, "student_test")      # evaluation step only
    rc, rh = route_overlap(collapse(cls), ref, g.alen), route_overlap(route, ref, g.alen)
    return dict(trip=tid, n_obs=tr.n_obs, n_reliable=int(rel.sum()),
                cls_top1=float((cls == y)[rel].mean()), hmm_top1=float((pred == y)[rel].mean()),
                final_top1=float((final == y)[rel].mean()),
                cls_continuity=continuity(cls, tr.arcs), hmm_continuity=continuity(pred, tr.arcs),
                cls_route_precision=rc["route_precision"], cls_route_recall=rc["route_recall"], cls_route_f1=rc["route_f1"],
                hmm_route_precision=rh["route_precision"], hmm_route_recall=rh["route_recall"], hmm_route_f1=rh["route_f1"],
                hmm_segments=int(seg.max() + 1), fill_failures=nfail, decode_time_s=t_dec)


if __name__ == "__main__":
    abl, per = [], []
    for name in MODELS:
        f = RES / f"test_scores_{name}.pkl"
        if not f.exists():
            print("skip (no saved scores):", name)
            continue
        sc = pd.read_pickle(f)
        rows = Parallel(n_jobs=4)(delayed(trip_eval)(t, g) for t, g in sc.groupby("trip"))
        d = pd.DataFrame(rows)
        d.insert(0, "model", name)
        per.append(d)
        wt = d.n_reliable
        abl.append(dict(model=name, cls_top1=np.average(d.cls_top1, weights=wt), hmm_top1=np.average(d.hmm_top1, weights=wt),
                        final_top1=np.average(d.final_top1, weights=wt),
                        cls_continuity=d.cls_continuity.mean(), hmm_continuity=d.hmm_continuity.mean(),
                        cls_route_f1=d.cls_route_f1.mean(), hmm_route_f1=d.hmm_route_f1.mean(),
                        cls_route_precision=d.cls_route_precision.mean(), hmm_route_precision=d.hmm_route_precision.mean(),
                        cls_route_recall=d.cls_route_recall.mean(), hmm_route_recall=d.hmm_route_recall.mean(),
                        trips_hmm_hurt_top1=int((d.hmm_top1 < d.cls_top1).sum()),
                        trips_final_hurt_top1=int((d.final_top1 < d.cls_top1).sum()),
                        trips_hmm_hurt_route_f1=int((d.hmm_route_f1 < d.cls_route_f1).sum()),
                        mean_decode_time_s=d.decode_time_s.mean()))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in abl[-1].items()}, flush=True)
    pd.DataFrame(abl).to_csv(RES / "hmm_ablation_models.csv", index=False)
    pd.concat(per).to_csv(RES / "hmm_test_pertrip.csv", index=False)
