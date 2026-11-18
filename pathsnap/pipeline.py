"""End-to-end, route-free prediction pipeline (raw trip files -> matched route).

Used by the FastAPI server. Nothing here imports labels.py or reads .route.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .candidates import build_candidates
from .data import load_trip
from .dataset import X_of
from .hmm import HMMParams, decode, reconstruct_route
from .metrics import normalise_scores

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, model_dir=ROOT / "models"):
        cfg = json.load(open(Path(model_dir) / "final_config.json"))
        self.cfg = cfg
        self.model = joblib.load(Path(model_dir) / f"{cfg['model']}.joblib")
        self.hmm = HMMParams(**cfg["hmm_params"])

    def score(self, cands):
        cands = cands.copy()
        cands["p"] = self.model.predict_proba(X_of(cands))[:, 1] if len(cands) else []
        cands["pn"] = normalise_scores(cands) if len(cands) else []
        return cands

    def predict_trip(self, trip_id, partition, data_root=None, use_hmm=True):
        timings = {}
        t0 = time.time()
        trip = load_trip(trip_id, partition, data_root, with_route=False)
        timings["load_s"] = time.time() - t0
        t0 = time.time()
        cands = build_candidates(trip)
        timings["candidates_features_s"] = time.time() - t0
        t0 = time.time()
        cands = self.score(cands)
        timings["classifier_s"] = time.time() - t0
        n = trip.n_obs
        cls = np.full(n, -1, dtype=np.int64)
        if len(cands):
            best = cands.loc[cands.groupby("obs").pn.idxmax()]
            cls[best.obs.to_numpy()] = best.arc.to_numpy()
        out = dict(trip=trip, cands=cands, cls_pred=cls, timings=timings)
        if use_hmm:
            t0 = time.time()
            pred, seg, g = decode(trip, cands, self.hmm)
            route, nfail = reconstruct_route(pred, seg, g)
            timings["hmm_s"] = time.time() - t0
            out.update(hmm_pred=pred, segments=seg, route=route, graph=g, fill_failures=nfail)
        timings["total_s"] = sum(v for k, v in timings.items() if k != "total_s")
        return out
