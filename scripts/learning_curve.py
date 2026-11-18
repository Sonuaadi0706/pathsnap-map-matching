"""Learning curves: top-1 accuracy against the number of TRAINING TRIPS.

For each of the 5 trip-grouped folds (results/folds.json) the validation fold (16 trips) is fixed and the
model is trained on nested random subsets of the other 64 trips (4, 8, 16, 32, 48, 64 trips). Reported per
size: in-sample top-1 on the trips used for training and top-1 on the validation fold (every 3rd fix, as in
all CV tables), mean and std over the 5 folds. Two models on the candidate table (46 features, K=24):
  * logreg  : logistic regression (configuration of Section 3)       -> high-bias reference
  * lgbm_v2 : LightGBM LambdaRank with the deployed configuration    -> deployed scorer
Student-test is not used.

usage: python scripts/learning_curve.py          (about 5 minutes)
outputs: results/learning_curve.csv, figures/learning_curve.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tune_v2 as T  # noqa: E402
from pathsnap import plotstyle as S  # noqa: E402
from pathsnap.candidates import FEATURES_V2  # noqa: E402
from pathsnap.dataset import X_of  # noqa: E402
from pathsnap.metrics import normalise_scores, observation_metrics, top1  # noqa: E402
from pathsnap.models import SEED, CandidateScorer, fit_model, make_model  # noqa: E402

SIZES = [4, 8, 16, 32, 48, 64]
STRIDE = 3
NAMES = {"logreg": "Logistic regression", "lgbm_v2": "LightGBM LambdaRank (deployed scorer)"}
c, o = T.DATA["v2"]


def acc(cands, obs):
    return observation_metrics(cands, obs, top1(cands))[0]["obs_top1"]


def run_logreg(trn, ev):
    m = make_model("logreg")
    fit_model(m, X_of(trn, FEATURES_V2), trn.y.to_numpy())
    out = []
    for cc, oo in ev:
        cc = cc.copy()
        cc["p"] = m.predict_proba(X_of(cc, FEATURES_V2))[:, 1]
        cc["pn"] = normalise_scores(cc)
        out.append(acc(cc, oo))
    return out


def run_lgbm(trn, ev):
    _, m = T.fit_predict("lambdarank", dict(T.BASE), trn, ev[0][0].head(100), FEATURES_V2)
    sc = CandidateScorer(m, FEATURES_V2, "rank")
    out = []
    for cc, oo in ev:
        cc = cc.copy()
        cc["p"], cc["pn"] = sc.score_candidates(cc)
        out.append(acc(cc, oo))
    return out


def main():
    rows, t0 = [], time.time()
    for k in range(5):
        pool = sorted(c.trip[c.fold != k].unique())
        order = list(np.random.default_rng(SEED + k).permutation(pool))
        val = (c[(c.fold == k) & (c.obs % 3 == 0)], o[(o.fold == k) & (o.obs % 3 == 0)])
        for n in SIZES:
            sub = set(order[:n])
            in_sub = c.trip.isin(sub)
            trn = c[in_sub & c.reliable & (c.obs % STRIDE == 0)]
            tr_ev = (c[in_sub & (c.obs % 3 == 0)], o[o.trip.isin(sub) & (o.obs % 3 == 0)])
            for name, fn in [("logreg", run_logreg), ("lgbm_v2", run_lgbm)]:
                a_tr, a_va = fn(trn, [tr_ev, val])
                rows.append(dict(model=name, fold=k, n_trips=n, n_train_rows=len(trn), train_top1=a_tr, val_top1=a_va))
                print(f"fold {k} n={n:>2} {name:<8} train {a_tr:.4f} val {a_va:.4f} | {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(T.RES / "learning_curve.csv", index=False)
    plot(d)


def plot(d):
    import matplotlib.pyplot as plt
    S.apply()
    g = d.groupby(["model", "n_trips"])[["train_top1", "val_top1"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, name in zip(axes, NAMES):
        q = g.loc[name]
        x = q.index.to_numpy()
        ends = []
        for col, colr, ls, lab in [("train_top1", S.MID, "--", "training trips (in-sample)"),
                                   ("val_top1", S.INK, "-", "validation fold (unseen trips)")]:
            mu, sd = q[(col, "mean")].to_numpy(), q[(col, "std")].to_numpy()
            ax.fill_between(x, mu - sd, mu + sd, color=colr, alpha=0.15, linewidth=0)
            ax.plot(x, mu, ls, color=colr, marker="o", markersize=5, label=lab)
            ends.append(mu[-1])
        if abs(ends[0] - ends[1]) < 0.004:      # curves meet: one label instead of two overlapping ones
            ends = [float(np.mean(ends))]
        for e in ends:
            ax.annotate(f"{e:.3f}", (x[-1], e), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=S.INK)
        ax.set_title(NAMES[name])
        ax.set_xticks(x)
        ax.set_xlim(0, 70)
        ax.set_xlabel("training trips used")
    axes[0].set_ylabel("observation top-1 accuracy")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.savefig(ROOT / "figures" / "learning_curve.png")
    print(g.round(4).to_string())


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(pd.read_csv(T.RES / "learning_curve.csv"))
    else:
        main()
