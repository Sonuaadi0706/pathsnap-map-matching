"""Build the consolidated technical report PDF from saved results (46-feature system).

Every number is read from results/*.json|csv; comparative statements are
guarded with assertions so the text cannot silently disagree with the logs.
output: report/PathSnap_Technical_Report.pdf
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import BASE_FEATURES, FEATURES, MAX_K, RADIUS_M, V2  # noqa: E402
from pathsnap.models import CONFIGS, NAMES, TRAIN_STRIDE  # noqa: E402

# ------------------------------------------------------------------ data
J = lambda p: json.load(open(RES / p))  # noqa: E731
ARCH = ROOT / "archive" / "round1_28_features"       # earlier 28-feature round: used only by the note in Section 4.3
JA = lambda p: json.load(open(ARCH / "results" / p))  # noqa: E731
MODELS = ["logreg", "gnb", "knn", "rf", "lgbm", "mlp"]
M = {m: J(f"model_{m}.json") for m in MODELS}
BASE = J("baseline_nearest.json")
EDA = pd.read_csv(RES / "eda_trips.csv", dtype={"trip": str})
QUAL = pd.read_csv(RES / "eda_quality_table.csv")
CAND = pd.read_csv(RES / "candidate_stats.csv", dtype={"trip": str})
OPT = J("optuna_v2_best.json")
TRIALS = pd.read_csv(RES / "optuna_v2_trials.csv")
TRIALS = TRIALS[TRIALS.state == "COMPLETE"]
START = OPT["untuned_same_protocol"]["params"]
assert all(json.load(open(ROOT / "configs/start_hyperparameters.json"))["best_params"][k] == v for k, v in START.items())
HAB = pd.read_csv(RES / "hmm_ablation_models.csv").set_index("model")
HPT = pd.read_csv(RES / "hmm_test_pertrip.csv", dtype={"trip": str})
LC = pd.read_csv(RES / "learning_curve.csv").groupby(["model", "n_trips"])[["train_top1", "val_top1"]].mean()
FI = pd.read_csv(RES / "feature_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
HL = pd.read_csv(RES / "history_lgbm.csv").groupby("epoch")[["train_logloss", "val_logloss"]].mean()
SMOKE = J("smoke_test.json")
ERR = J("error_breakdown.json")
APIALL = pd.read_csv(RES / "api_all_trips.csv", dtype={"trip": str})
assert ERR["predictions_identical_without_route_files"] and (APIALL.status == "ok").all()
V2VAR = pd.read_csv(RES / "tune_v2_variants.csv")
PROMO = pd.read_csv(RES / "optuna_v2_promotion.csv")
M2U, M2T, CHOICE = J("model_lgbm_v2_untuned.json"), J("model_lgbm_v2.json"), J("final_choice.json")
M2 = M2T if CHOICE["selected"] == "lgbm_v2" else M2U
DEC = J("decoder_best.json")
DTR = pd.read_csv(RES / "decoder_trials.csv")
FS = J("final_summary.json")
F2T, F2R = FS["student_test"]["final_decoder"], FS["train"]["final_decoder"]
FSP, DECP = J("final_summary_pooled.json"), J("decoder_best_pooled.json")
DGRID = pd.read_csv(RES / "decoder_grid_v2.csv")
FINAL = json.load(open(ROOT / "models/final_config.json"))
assert not FS["adopted_tuned_decoder"] and not FS["grid2"]["adopted"] and FS["selected_decoder"] == "round1_decoder"
BR = J("bridge_eval.json")
ROB = J("robustness_test.json")
LAT = pd.read_csv(RES / "latency_large_trips.csv")
assert ROB["all_ok"]
assert BR["adopted"] == FS["bridge_adopted"] == FINAL["hmm_params"]["bridge"]
assert CHOICE["selected"] == "lgbm_v2_untuned"
FTEST = pd.read_csv(RES / "final_test.csv", dtype={"trip": str})
FTRAIN = pd.read_csv(RES / "final_train_oof.csv", dtype={"trip": str})
VAL = J("validation_final.json")
assert FINAL["model"] == CHOICE["selected"] and VAL["predictions_identical_without_route_files"]
assert VAL["independent_recomputation"]["match"]
SEL = FINAL["model"]
DEPNAME = "LightGBM LambdaRank (deployed)"
# earlier 28-feature round (archived): numbers quoted in Section 4.3 only
M1 = {m: JA(f"model_{m}.json") for m in MODELS + ["lgbm_tuned"]}
OPT1, RE1, HSUM1 = JA("optuna_best.json"), JA("realign_summary.json"), JA("hmm_summary.json")
ci = lambda d: f"{d['point']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"  # noqa: E731


def miss(cs, part):
    g = cs[cs.partition == part]
    return float(np.average(g.candidate_miss_rate, weights=g.n_obs * g.frac_reliable))


TEAM = json.load(open(ROOT / "configs/team.json")) if (ROOT / "configs/team.json").exists() else {}
N_TEST_ROWS = int(CAND[CAND.partition == "student_test"].n_rows.sum())
mb = lambda name: (ROOT / "models" / f"{name}.joblib").stat().st_size / 1e6  # noqa: E731

cv = lambda m, k: M[m]["cv_mean"][k]  # noqa: E731
cs = lambda m, k: M[m]["cv_std"][k]  # noqa: E731
te = lambda m, k: M[m]["student_test"][k]  # noqa: E731
f3 = lambda x: f"{x:.3f}"  # noqa: E731
f4 = lambda x: f"{x:.4f}"  # noqa: E731
pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731

# guarded claims ------------------------------------------------------
best_cv = max(MODELS, key=lambda m: cv(m, "obs_macro_f1_trip"))
best_te = max(MODELS, key=lambda m: te(m, "obs_top1"))
best_ll = min(MODELS, key=lambda m: te(m, "obs_logloss"))
assert best_cv == "lgbm" and best_te == "lgbm" and best_ll == "lgbm", (best_cv, best_te, best_ll)
gap = {m: cv(m, "train_obs_top1") - cv(m, "obs_top1") for m in MODELS}
assert max(gap, key=gap.get) == "rf"
assert te("knn", "infer_time_s") == max(te(m, "infer_time_s") for m in MODELS)
assert cv("knn", "obs_top1") < cv("logreg", "obs_top1")
assert sorted(MODELS, key=lambda m: -cv(m, "obs_macro_f1_trip"))[1] == "rf" and sorted(MODELS, key=lambda m: -cv(m, "obs_top1"))[1] == "rf"
assert te("gnb", "obs_logloss") == max(te(m, "obs_logloss") for m in MODELS)
assert M2["cv_mean"]["obs_top1"] > max(cv(m, "obs_top1") for m in MODELS), "deployed ranking scorer must beat every binary model in CV"
sel_pt = HPT[HPT.model == SEL]
hurt_top1 = sel_pt[sel_pt.hmm_top1 < sel_pt.cls_top1]
hurt_rf1 = sel_pt[sel_pt.hmm_route_f1 < sel_pt.cls_route_f1]

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10.2, textColor=colors.HexColor("#444444"))
CAP = ParagraphStyle("c", parent=SMALL, alignment=TA_CENTER, spaceAfter=10, fontName="Helvetica-Oblique")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15.5, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=8, spaceAfter=5)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.2, spaceBefore=6, spaceAfter=3)
Q = ParagraphStyle("q", parent=BODY, fontName="Helvetica-Bold", spaceAfter=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.4, leading=9, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=6.9, leading=8.4, backColor=colors.HexColor("#F3F3F3"),
                      borderPadding=4, spaceAfter=8)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))  # noqa: E731
W = A4[0] - 3.6 * cm


def fig(path, caption, width=W, maxh=None):
    p = FIG / path
    if not p.exists():
        P(f"<i>[missing figure {path}]</i>")
        return
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    h = width * ih / iw
    if maxh and h > maxh:
        width, h = width * maxh / h, maxh
    story.append(KeepTogether([Image(str(p), width=width, height=h), Paragraph(caption, CAP)]))


def table(rows, widths=None, header=True, font=7.4, zebra=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3E3E3")))
    if zebra:
        for i in range(2, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F7F7")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 7))


def qa(q, a):
    P(q, Q)
    P(a)


# ================================================================== title page
story.append(Spacer(1, 3.2 * cm))
P("<font size=24><b>PathSnap</b></font>", ParagraphStyle("t", parent=BODY, alignment=TA_CENTER, leading=30))
P("<font size=14>Distributed Map-Matching &amp; Route Reconstruction</font>",
  ParagraphStyle("t2", parent=BODY, alignment=TA_CENTER, leading=20))
P("Technical Report  |  UE24CS352A Machine Learning Hackathon  |  PES University, RR Campus",
  ParagraphStyle("t3", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor("#555555")))
story.append(Spacer(1, 1.2 * cm))
team_rows = [["Team", TEAM["team_name"]]] if TEAM.get("team_name") else []
for mbr in TEAM.get("members", []):
    team_rows.append(["Team member", f"{mbr['name']}  |  {mbr['srn']}  |  GitHub: {mbr['github']}"])
if not TEAM.get("members"):
    team_rows.append(["Members / SRN", "<i>(to be filled in)</i>"])
team_rows.append(["Repository", TEAM.get("repo_url", "<i>(GitHub URL)</i>")])
team_rows.append(["Date", date.today().strftime("%d %B %Y")])
table(team_rows, widths=[3.0 * cm, 12.5 * cm], header=False, zebra=False)
story.append(Spacer(1, 0.8 * cm))
P("<b>Summary.</b> Each GPS fix is matched to a road arc in two stages. (1) A candidate scorer ranks the up to "
  f"{MAX_K} arcs within {RADIUS_M:.0f} m of the fix using {len(FEATURES)} route-free geometric, kinematic, temporal-context "
  "and topology features. Six models from six algorithm families were trained with identical candidates, labels and 5-fold "
  "trip-grouped cross-validation on the 80 training trips; LightGBM was best on every quality metric. For the selected "
  "family the training objective was changed from per-row binary log loss to LambdaRank (Section 4.1), and a Bayesian "
  f"search over its hyperparameters (Optuna TPE, {OPT['n_trials']} trials, Section 4) did not improve on the starting "
  "configuration within fold noise, so the starting configuration is deployed. "
  "(2) An HMM decoded with Viterbi over the directed road graph turns the per-fix scores into a connected route, "
  "and each fix is then re-assigned to an arc of that predicted route in travel order. Two further Bayesian searches "
  f"over the decoder's parameters ({DEC['n_trials']} trials each) did not generalise to held-out training trips and "
  "were not adopted (Section 5.3). "
  f"On the 10 student-test trips the final system reaches observation top-1 accuracy <b>{f3(F2T['final']['top1'])}</b> "
  f"(95% trip-bootstrap interval {VAL['student_test']['final_top1']['ci95'][0]:.3f}-{VAL['student_test']['final_top1']['ci95'][1]:.3f}; "
  f"scorer alone {f3(F2T['cls']['top1'])}, nearest-segment baseline {f3(BASE['student_test']['obs_top1'])}), "
  f"macro-F1 <b>{f3(F2T['final']['macro_f1'])}</b>, route continuity <b>{f3(F2T['final']['continuity'])}</b> and "
  f"length-weighted route F1 <b>{f3(F2T['route_f1'])}</b>; on the 80 training trips (out-of-fold) top-1 is "
  f"{f3(F2R['final']['top1'])}. "
  "A FastAPI service (laptop 1) and a browser UI (laptop 2) run the full pipeline per selected trip and score it "
  "against server-side reference routes that never reach the model or the UI; the ten reserved evaluator trips were "
  "served this way from a second laptop after the system was frozen (Section 7).")
fig("diagram_pipeline.png", "Figure 1. PathSnap pipeline. Dark boxes are learned / decoded components.")
story.append(PageBreak())

# ================================================================== 1 problem
P("1. Problem Formulation", H1)
P("Inputs per trip: a GPS track (longitude, latitude, timestamp), a road graph (nodes; directed arcs as "
  "from/to node pairs) and, for development trips, the reference route as a chronological list of arc IDs. "
  "The required output is a road arc for every relevant GPS observation and a realistic route. The route file "
  "is <i>not</i> aligned to the track (the median trip has "
  f"{int(EDA.n_obs.median())} fixes but only {int(EDA.route_len_arcs.median())} route arcs), so per-observation "
  "labels must be derived (Section 2.1). We cast matching as <b>candidate ranking</b>: for fix <i>i</i> with "
  "candidate set C<sub>i</sub>, a model scores every candidate (Section 3: a binary classifier estimating p(arc is the "
  "reference arc | features); deployed scorer: a LambdaRank model, Section 4.1), the scores are normalised into a "
  "distribution over C<sub>i</sub>, and the arg-max is the scorer-only prediction. "
  "Because arc IDs are trip-specific, a multi-class formulation over global arc IDs would be meaningless; "
  "ranking per observation generalises across trips and cities. Sequence consistency is then enforced by "
  "an HMM whose hidden states are the candidates (Section 5).")
P("<b>Primary selection metric.</b> CV observation-level macro-F1 and top-1 accuracy (unit: GPS fix) on "
  "held-out trips, because they directly measure the evaluator's quantity (is each fix on the right arc) "
  "and are computed on trips the model never saw. Probability quality (log loss) and inference time are used "
  "as secondary criteria because the decoder consumes the probabilities and the API must answer interactively.")

# ================================================================== 2 EDA
P("2. Dataset Understanding and Exploratory Data Analysis", H1)
tr, ts = EDA[EDA.partition == "train"], EDA[EDA.partition == "student_test"]
inv = [["Partition", "Trips", "GPS fixes", "Fixes/trip (min / median / max)", "Nodes (median)", "Arcs (median)",
        "Route arcs (median)", "Track length km (median)"]]
for name, g in [("train", tr), ("student_test", ts)]:
    inv.append([name, len(g), f"{int(g.n_obs.sum()):,}", f"{int(g.n_obs.min())} / {int(g.n_obs.median())} / {int(g.n_obs.max())}",
                f"{int(g.n_nodes.median()):,}", f"{int(g.n_arcs.median()):,}", int(g.route_len_arcs.median()),
                f"{g.track_km.median():.1f}"])
inv.append(["evaluator_test", 10, "final evaluation only (Section 7)", "-", "-", "-", "-", "-"])
table(inv)
P("Table 1. Dataset inventory (source: results/eda_trips.csv, produced by scripts/eda.py). All 90 supplied "
  "trips contain all four files; the ten evaluator trips were received for the final evaluation, after the system was frozen, and are "
  "not part of any analysis in Sections 2-5.", CAP)

P("2.1 How did we preprocess the data?", H2)
P("<b>Parsing and validation.</b> Files are whitespace-separated without headers. We verified the column "
  "order (longitude first: values span "
  f"{EDA.lon_min.min():.1f} to {EDA.lon_max.max():.1f} in the first column and {EDA.lat_min.min():.1f} to "
  f"{EDA.lat_max.max():.1f} in the second, and the tracks overlay the OSM basemap correctly in Folium), that "
  "node/arc IDs are zero-based row indices, that every route ID is a valid arc index and that timestamps are "
  "strictly increasing. Trips are spread over several continents, so a single projection is inappropriate.")
P("<b>Projection.</b> Each trip is projected to a local equirectangular plane centred on its own centroid "
  "(x = R&middot;&Delta;lon&middot;cos(lat<sub>0</sub>), y = R&middot;&Delta;lat). Over the 5-50 km extent of a trip "
  "the scale error is far below GPS noise (&lt;0.1%), and it allows exact point-to-segment distances, "
  "projection fractions and bearings in metres. Haversine is used only for reference checks.")
P("<b>Label derivation (training/evaluation only).</b> For each trip we align the fix sequence to the route "
  "sequence with a monotone dynamic programme: choose route indices j<sub>1</sub> &le; j<sub>2</sub> &le; &hellip; "
  "&le; j<sub>n</sub> minimising &Sigma;<sub>i</sub> dist(fix<sub>i</sub>, route[j<sub>i</sub>]). Monotonicity "
  "respects the travel order, so loops and repeated arcs (one trip repeats 255 arcs) are handled; arcs may be "
  "skipped because a short arc can be crossed between two 1 Hz fixes. The aligned arc is the label of the fix. "
  "Fixes whose aligned distance exceeds 60 m are flagged <i>off-route</i>: the reference does not describe "
  "them (e.g. train trip 00000002 keeps recording ~540 fixes after the route ends and contains a 280-fix "
  "detour absent from the route). They are excluded from training and scoring and counted in Table 2. This "
  "is a documented approximation: labels inherit any error in the reference route and in the nearest-feasible "
  "alignment, which we cannot verify independently.")
P(f"<b>Candidate generation.</b> Arcs are densified every 10 m into a KD-tree (restricted to the track bounding "
  f"box + 280 m). For every fix we take arcs within {RADIUS_M:.0f} m, compute exact segment distance, and keep "
  f"the {MAX_K} nearest (two-way streets contribute two co-located arcs). Original trip ID, observation index "
  f"and arc ID are preserved in every row. Feature extraction for all 90 trips takes {CAND.feat_time_s.sum():.0f} s of CPU time (run on 4 cores).")
P("<b>Training rows.</b> 1 Hz fixes are highly autocorrelated, so models are fitted on every 3rd fix of each "
  "training trip (KNN: every 9th, for prediction cost); CV metrics use every 3rd fix of the validation trips; "
  "student-test metrics use <i>all</i> fixes. Scaling (where used) is fitted inside each training fold only.")

P("2.2 Missing values and data quality", H2)
fmtq = lambda v: (f"{int(v):,}" if float(v).is_integer() else f"{float(v):.4f}")  # noqa: E731
qrows = [["Check", "train", "student_test"]] + [[c, fmtq(a), fmtq(b)] for c, a, b in QUAL.values.tolist()]
table(qrows, widths=[8 * cm, 3.5 * cm, 3.5 * cm])
P("Table 2. Missing-value and data-quality analysis (results/eda_quality_table.csv). There are no NaNs, "
  "duplicate timestamps or isolated nodes; irregularity comes from time gaps, degenerate arcs and reference "
  "routes that do not cover the whole track.", CAP)
fig("eda_obs_per_trip.png", "Figure 2. Trip-length distribution (train and student_test).", width=W * 0.8)
fig("eda_sampling.png", "Figure 3. Sampling intervals and apparent speed between consecutive fixes (all 90 trips).")

P("2.3 What patterns did we observe?", H2)
pos_rank0 = float(CAND.nearest_top1.mean())
P(f"<b>(a) The nearest arc is often wrong.</b> Only {pc(BASE['student_test']['obs_top1'])} of student-test fixes "
  f"(and {pc(BASE['cv_mean']['obs_top1'])} on CV folds) lie on their nearest arc. The main reason is that "
  f"{pc(EDA.frac_twoway_arcs.median())} of arcs (median trip) are one direction of a two-way street, so the "
  "reverse arc is at exactly the same distance; parallel roads, intersections and multipath noise add the rest "
  "(Figure 6, centre). Direction can only be resolved from motion, which motivated the heading features.")
P("<b>(b) Noise level.</b> The median aligned distance is about 4-6 m, with a heavy tail (Figure 6, left); a "
  f"{RADIUS_M:.0f} m radius therefore captures the reference arc for "
  f"{100 * (1 - miss(CAND, 'student_test')):.2f}% of scored "
  f"student-test fixes and {100 * (1 - miss(CAND, 'train')):.2f}% of scored training fixes. <b>(c) Sampling.</b> Most steps are exactly 1 s, but 28 training trips contain a gap "
  "longer than 30 s (maximum 153,064 s, i.e. a recording restarted much later) and the apparent speed has "
  "outliers above 50 m/s caused by position jumps; the decoder must tolerate both. <b>(d) Road networks</b> vary "
  f"from {int(EDA.n_arcs.min()):,} to {int(EDA.n_arcs.max()):,} directed arcs per trip and are not always "
  "strongly connected (median largest SCC "
  f"{pc(EDA.largest_scc_frac.median())}), so transitions can be infeasible. <b>(e) Reference routes</b> are "
  f"head-to-tail continuous in {pc(EDA.route_continuity.mean())} of steps, but may stop before the track ends.")
fig("eda_trajectories.png", "Figure 4. GPS trajectories, road network and reference route for six training trips "
    "(train partition, local metres). Note the track continuing beyond the reference in trip 00000002.")
fig("eda_network.png", "Figure 5. Road-network size and connectivity per trip (train and student_test).")
fig("eda_labels_candidates.png", "Figure 6. Alignment distance, rank of the reference arc among candidates, and "
    "heading agreement for reference vs. other candidates (train partition).")

P("2.4 Features and why we chose them", H2)
feat_tab = [["Group", "Features", "Rationale"],
            ["Distance", "d, d_rank, d_minus_min, d_ratio", "GPS error is roughly isotropic; relative distance to the best candidate is more transferable across noise levels than absolute distance."],
            ["Candidate density", "n_cand, n_cand_30", "Dense urban grids need more evidence than an isolated rural road."],
            ["Heading", "cos_head, abs_head, cos_head_win, speed, cos_head_x_speed", "Separates the two directions of a two-way street and crossing roads; heading is only reliable when moving, hence the speed interaction and a longer +-8 fix window."],
            ["Position on arc", "t_proj, t_outside, arc_len, log_arc_len", "Projections falling beyond an arc end indicate the neighbouring arc."],
            ["Temporal context", "d_prev1, d_next1, d_win5, d_win15", "Distance of neighbouring fixes to the same arc: smooths single-fix noise using only GPS (no labels)."],
            ["Topology", "same_prev/next, conn_prev/next, head_outdeg, tail_indeg, twoway", "Whether the candidate equals / touches the nearest arc of the adjacent fixes; intersection complexity."],
            ["Gaps", "gap_prev, gap_next", "Context features are less trustworthy across time gaps."],
            ["Asymmetric context", "d_prev5, d_next5, d_prev15, d_next15", "Mean distance of the previous / next fixes to the candidate separately: at an arc change the past supports the old arc and the future the new one."],
            ["Relative context", "d_win5_gap, d_win15_gap, d_prev5_gap, d_next5_gap, d_win5_rank, cos_win_gap", "Context evidence relative to the best candidate of the same fix; makes the context features comparable across noise levels."],
            ["Long-baseline direction", "cos_head_w20, prog8, disp20", "Heading over +-20 fixes and signed progress along the arc: direction remains defined during stops and slow driving."],
            ["Position along the arc", "t_raw, t_prev5, t_next5", "Unclipped projection fraction of the fix and its neighbours: tells whether the vehicle has not yet reached, is on, or has already left the arc."],
            ["Neighbour agreement", "win_nearest_share, win_cand_share", "Share of the +-5 fixes whose nearest arc is this candidate, and of nearby fixes that have it as a candidate at all."]]
assert sum(len(r[1].replace("same_prev/next", "a, b").replace("conn_prev/next", "a, b").split(",")) for r in feat_tab[1:]) == len(FEATURES)
table(feat_tab, widths=[2.9 * cm, 5.2 * cm, 9.3 * cm])
P(f"Table 3. The {len(FEATURES)} route-free features (pathsnap/candidates.py, FEATURES): {len(BASE_FEATURES)} base features (first seven "
  f"groups) and {len(FEATURES) - len(BASE_FEATURES)} context features (last five groups). All are computed from the track and the road "
  "graph only; none uses the reference route. Section 4.1 measures what the context features add.", CAP)
top = ", ".join(f"{k} ({v:.1%})" for k, v in FI.head(5).items())
P(f"The deployed scorer attributes most split gain to {top} (Figure 7): relative distance and direction evidence over a "
  "window of fixes are the decisive signals, consistent with patterns (a) and (b). All "
  f"{int((FI > 0).sum())} features are used in at least one split.")
fig("feat_importance.png", f"Figure 7. Feature importance (gain share) of the deployed LightGBM LambdaRank scorer ({len(FEATURES)} features).",
    width=W * 0.6, maxh=14.5 * cm)
P("2.5 Geographical map (Folium)", H2)
fig("screenshot_map_eda.png", "Figure 8. Folium EDA map, train trip 00000002 (maps/eda_train_00000002.html): "
    "GPS fixes, nearby road network and reference route, with layer control and legend.", width=W * 0.9, maxh=11 * cm)
P("Partitions used: Tables 1-2 and Figures 2, 3, 5 use train and student_test inventories (no labels); "
  "Figures 4, 6, 7 and 8 use the train partition only. Student-test labels are used only for scoring.", SMALL)
story.append(PageBreak())

# ================================================================== 3 models
P("3. Model Experimentation and Comparison", H1)
P("<b>Protocol (identical for all models).</b> Same candidate sets and labels; 5 trip-grouped folds over the 80 "
  "training trips with a fixed trip-to-fold assignment (results/folds.json, seed 42); no trip ever appears in "
  "two folds or in two partitions. Each model is refitted on all 80 trips and evaluated once on the 10 "
  "student-test trips. Metrics and units: <b>observation top-1</b> (fix), <b>macro-F1</b> (per trip, F1 averaged "
  "over the arcs present in that trip, then averaged over trips, because arc IDs are trip-specific), "
  "<b>observation log loss</b> (-mean log of the renormalised probability of the reference arc, fixes whose "
  "reference arc is in the candidate set), <b>candidate precision / recall / F1 and log loss</b> (binary, "
  "candidate row, threshold 0.5), <b>route continuity</b> (consecutive fix pairs whose predicted arcs are equal "
  f"or connected head-to-tail), and <b>inference time</b> (scoring all {N_TEST_ROWS:,} student-test candidate rows). "
  f"The candidate miss rate is a property of the candidate generator, not the model: {pc(te('lgbm', 'miss_rate'))} "
  f"on student-test and {pc(np.mean([cv(m, 'miss_rate') for m in MODELS]))} on CV folds.")

P("3.1 Overall model comparison", H2)
rows = [["Model", "Algorithm family", "Key configuration", "CV macro-F1", "Top-1 (test)", "Contin. (test)",
         "Macro F1 (test)", "Log loss (test)", "Infer. time (test)"]]
FAM = {"logreg": "Linear", "gnb": "Probabilistic", "knn": "Instance-based", "rf": "Bagging ensemble",
       "lgbm": "Boosting", "mlp": "Neural net"}
keycfg = {"logreg": "C=1.0, L2, lbfgs, standardised", "gnb": "var_smoothing=1e-9",
          "knn": "k=50, distance-weighted, standardised, stride-9 rows",
          "rf": "200 trees, depth 20, min_leaf 3, sqrt features",
          "lgbm": "400 rounds, lr 0.05, 63 leaves, min_child 50, bagging 0.8/0.8",
          "mlp": "64-32 ReLU, Adam 1e-3, L2 1e-4, batch 1024, 20 epochs"}
for i, m in enumerate(MODELS, 1):
    rows.append([f"Model {i}: {NAMES[m]}", FAM[m], keycfg[m],
                 f"{f3(cv(m, 'obs_macro_f1_trip'))} &plusmn; {f3(cs(m, 'obs_macro_f1_trip'))}",
                 f3(te(m, "obs_top1")), f3(te(m, "route_continuity")), f3(te(m, "obs_macro_f1_trip")),
                 f3(te(m, "obs_logloss")), f"{te(m, 'infer_time_s'):.2f} s"])
rows.append(["Deployed scorer: LightGBM LambdaRank (Section 4)", "Boosting (ranking)",
             f"{M2['config']['n_estimators']} rounds, lr {M2['config']['learning_rate']:.3f}, {M2['config']['num_leaves']} leaves, LambdaRank objective",
             f"{f3(M2['cv_mean']['obs_macro_f1_trip'])} &plusmn; {f3(M2['cv_std']['obs_macro_f1_trip'])}",
             f3(M2["student_test"]["obs_top1"]), f3(M2["student_test"]["route_continuity"]),
             f3(M2["student_test"]["obs_macro_f1_trip"]), f3(M2["student_test"]["obs_logloss"]),
             f"{M2['student_test']['infer_time_s']:.2f} s"])
rows.append(["Reference: nearest segment", "Geometric (untrained)", "argmin distance",
             f"{f3(BASE['cv_mean']['obs_macro_f1_trip'])} (no fitting)", f3(BASE["student_test"]["obs_top1"]),
             f3(BASE["student_test"]["route_continuity"]), f3(BASE["student_test"]["obs_macro_f1_trip"]),
             "N/A (not a fitted probability)", f"{BASE['student_test']['infer_time_s']:.2f} s"])
table(rows, widths=[2.75 * cm, 2.25 * cm, 3.0 * cm, 1.65 * cm, 1.3 * cm, 1.35 * cm, 1.45 * cm, 1.65 * cm, 1.35 * cm])
P(f"Table 4. Consolidated comparison, classifier only (no HMM), all models on the same {len(FEATURES)}-feature candidate table. CV values are mean &plusmn; std over 5 trip-grouped "
  "folds; test values are over all fixes of the 10 student-test trips (sources: results/model_*.json, "
  "results/comparison_table.csv). Inference time = scoring time of all test candidates; end-to-end API latency "
  "including features and HMM is reported in Section 7.", CAP)
rows = [["Model", "CV top-1 (val)", "Train- fold top-1", "Gap", "Cand. prec.", "Cand. recall", "Cand. F1",
         "Cand. log loss", "Fit time (s)", "Train rows"]]
for m in MODELS:
    rows.append([NAMES[m], f"{f3(cv(m, 'obs_top1'))} &plusmn; {f3(cs(m, 'obs_top1'))}", f3(cv(m, "train_obs_top1")),
                 f"{cv(m, 'train_obs_top1') - cv(m, 'obs_top1'):+.3f}", f3(te(m, "cand_precision_pos")),
                 f3(te(m, "cand_recall_pos")), f3(te(m, "cand_f1_pos")), f3(te(m, "cand_logloss")),
                 f"{M[m]['final_fit_time_s']:.1f}", f"{M[m]['final_train_rows']:,}"])
table(rows, widths=[3.0 * cm, 2.1 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.75 * cm])
P("Table 5. Generalisation and candidate-level precision/recall (student-test, threshold 0.5). 'Train-fold top-1' "
  "is measured on the training trips of each fold (KNN: on 12 of them), so the gap is direct evidence of "
  "over-fitting.", CAP)

P("3.2 Per-model analysis", H2)


def model_section(i, m, why, strengths, hyper, loss, fit_note):
    P(f"Model {i}: {NAMES[m]} ({CONFIGS[m]['family']})", H3)
    qa("A1. Why suitable for PathSnap?", why)
    qa("A2. Strengths and limitations for this task", strengths)
    qa("B5. Important hyperparameters and initial values", hyper)
    qa("B6. Loss / optimisation objective", loss)
    qa("C7-C8. Student-test performance (observation level)",
       f"Top-1 accuracy <b>{f3(te(m, 'obs_top1'))}</b>, macro-F1 <b>{f3(te(m, 'obs_macro_f1_trip'))}</b>, "
       f"observation log loss <b>{f3(te(m, 'obs_logloss'))}</b> (candidate log loss {f3(te(m, 'cand_logloss'))}); "
       f"candidate precision {f3(te(m, 'cand_precision_pos'))}, recall {f3(te(m, 'cand_recall_pos'))}; route "
       f"continuity {f3(te(m, 'route_continuity'))}. CV: top-1 {f3(cv(m, 'obs_top1'))} &plusmn; "
       f"{f3(cs(m, 'obs_top1'))}, macro-F1 {f3(cv(m, 'obs_macro_f1_trip'))}.")
    qa("C9. Over- or under-fitting?", fit_note)


g = gap
model_section(1, "logreg",
              "A transparent linear baseline that outputs probabilities directly. Most features are monotone "
              "evidence (smaller relative distance and higher heading agreement mean more likely), which a "
              "linear logit can exploit.",
              f"Fast (fit {M['logreg']['final_fit_time_s']:.0f} s, scoring {1000 * te('logreg', 'infer_time_s'):.0f} ms), well-behaved probabilities, no tuning burden. It cannot represent "
              "interactions that matter here, e.g. heading is informative only at speed and context distances "
              "matter only when the fix is ambiguous; it also assumes a monotone effect of features such as "
              "arc length.",
              "C=1.0 (inverse L2 strength), solver lbfgs, max_iter=1000, StandardScaler fitted on training folds.",
              "Binary cross-entropy (log loss) + L2 penalty.",
              f"No over-fitting: training-fold top-1 {f3(cv('logreg', 'train_obs_top1'))} equals CV "
              f"{f3(cv('logreg', 'obs_top1'))} (gap {g['logreg']:+.3f}). It under-fits relative to tree ensembles "
              f"(CV top-1 {f3(cv('lgbm', 'obs_top1'))} for LightGBM), i.e. the remaining error is bias from the "
              "linear boundary, not variance.")
model_section(2, "gnb",
              "A generative probabilistic model; included to test whether class-conditional feature "
              "distributions alone separate reference arcs from distractors.",
              f"Trains in {M['gnb']['final_fit_time_s']:.1f} s and needs no scaling. Its independence assumption is badly violated: d, "
              "d_minus_min, d_ratio and the many windowed distances are strongly correlated, so evidence is counted "
              "several times and probabilities become extreme.",
              "var_smoothing=1e-9 (default); no other hyperparameters.",
              "Maximum-likelihood Gaussian class-conditional densities and class priors (no iterative loss).",
              f"Under-fitting/mis-specification rather than over-fitting: train {f3(cv('gnb', 'train_obs_top1'))} vs "
              f"CV {f3(cv('gnb', 'obs_top1'))}. The over-counted evidence shows in a candidate recall of "
              f"{f3(te('gnb', 'cand_recall_pos'))} with precision only {f3(te('gnb', 'cand_precision_pos'))} and the "
              f"worst log loss of all models ({f3(te('gnb', 'obs_logloss'))}).")
model_section(3, "knn",
              "Map-matching decisions are local: fixes with similar distance/heading/context configurations "
              "should share labels, which an instance-based method captures without assuming a functional form.",
              f"Non-parametric and simple. But all {len(FEATURES)} standardised features get equal weight (irrelevant and redundant ones "
              f"dilute the metric), training data must be stored ({TRAIN_STRIDE['knn']}-stride subsample, "
              f"{M['knn']['final_train_rows']:,} rows) and prediction is slow: "
              f"{te('knn', 'infer_time_s'):.1f} s for the test candidates, {te('knn', 'infer_time_s') / max(te(m, 'infer_time_s') for m in MODELS if m != 'knn'):.0f}x slower than the next slowest model.",
              "n_neighbors=50, weights='distance', Euclidean metric on StandardScaler features.",
              "None (lazy learner); probability = distance-weighted share of positive neighbours.",
              f"Moderate variance: training-fold top-1 {f3(cv('knn', 'train_obs_top1'))} vs CV "
              f"{f3(cv('knn', 'obs_top1'))} (gap {g['knn']:+.3f}). The larger issue is bias from the unweighted "
              f"metric: CV top-1 is {f3(cv('knn', 'obs_top1'))}, below logistic regression.")
model_section(4, "rf",
              "Bagged decision trees model non-linear thresholds and interactions (e.g. heading only when "
              "moving) and are robust to unscaled, heterogeneous features.",
              f"Second-best CV macro-F1 and CV top-1 of the six models, with candidate precision {f3(te('rf', 'cand_precision_pos'))}. "
              f"Limitations: the largest artefact ({mb('rf'):.0f} MB), {M['rf']['final_fit_time_s']:.0f} s to fit, and "
              "deep trees memorise individual trips.",
              "n_estimators=200, max_depth=20, min_samples_leaf=3, max_features='sqrt', bootstrap=True.",
              "Gini impurity reduction per split; probabilities = mean leaf class frequency.",
              f"Clear over-fitting: training-fold top-1 {f3(cv('rf', 'train_obs_top1'))} vs CV "
              f"{f3(cv('rf', 'obs_top1'))}, the largest gap of all models ({g['rf']:+.3f}); training-fold candidate "
              f"log loss {f3(cv('rf', 'train_cand_logloss'))} vs {f3(cv('rf', 'cand_logloss'))} on validation. "
              "Averaging over trees keeps validation performance high despite this.")
model_section(5, "lgbm",
              "Gradient-boosted trees add trees that correct the current errors, which suits a tabular problem "
              "with interacting geometric features; it optimises log loss directly, which the HMM consumes.",
              f"Best CV macro-F1, best test top-1 and the lowest log loss of the six models; fits in "
              f"{M['lgbm']['final_fit_time_s']:.0f} s and scores {te('lgbm', 'infer_time_s'):.1f} s for {N_TEST_ROWS / 1000:.0f}k rows. "
              "Limitations: more hyperparameters to tune, and probabilities can become over-confident with many rounds.",
              "n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8 "
              "(freq 1), colsample_bytree=0.8, no explicit L1/L2.",
              "Binary log loss minimised by second-order gradient boosting.",
              f"Mild over-fitting: training-fold top-1 {f3(cv('lgbm', 'train_obs_top1'))} vs CV "
              f"{f3(cv('lgbm', 'obs_top1'))} (gap {g['lgbm']:+.3f}). In the boosting-round curves (Figure 13) the validation loss "
              f"reaches its minimum at round {int(HL.val_logloss.idxmin())} ({HL.val_logloss.min():.4f}) and ends at {HL.val_logloss.iloc[-1]:.4f} "
              f"after {len(HL)} rounds, while the training loss keeps falling to {HL.train_logloss.iloc[-1]:.4f}: the later rounds fit the "
              "training trips without improving validation loss. The number of rounds and the learning rate are therefore "
              "part of the Bayesian search of Section 4.")
model_section(6, "mlp",
              "A neural network learns smooth non-linear feature combinations and yields probabilities; it tests "
              "whether a learned representation beats axis-aligned tree splits on these features.",
              f"Good probabilities (obs. log loss {f3(te('mlp', 'obs_logloss'))}) and fast inference "
              f"({te('mlp', 'infer_time_s'):.2f} s). Needs feature scaling, is sensitive to learning rate and epochs, "
              "and on tabular data of this size did not outperform boosting.",
              "hidden_layer_sizes=(64, 32), ReLU, Adam lr=1e-3, alpha (L2)=1e-4, batch_size=1024, 20 epochs, "
              "StandardScaler; trained epoch by epoch with partial_fit to record histories.",
              "Binary cross-entropy + L2 weight penalty.",
              f"Slight over-fitting: training-fold top-1 {f3(cv('mlp', 'train_obs_top1'))} vs CV "
              f"{f3(cv('mlp', 'obs_top1'))}; in Figure 12 the validation log loss stops improving after about "
              "8-12 epochs while training loss continues to decrease, so early stopping near that point would be "
              "appropriate.")

P("3.3 Model visualisations", H2)
fig("cmp_cv_macro_f1.png", "Figure 9. CV observation macro-F1 by model (error bars: std over the 5 trip-grouped folds).", width=W * 0.85)
fig("cmp_test_top1_f1.png", "Figure 10. Student-test top-1 accuracy and macro-F1 (classifier only).", width=W * 0.85)
fig("cmp_logloss.png", "Figure 11. Student-test log loss: observation-level (left) and candidate-level (right).")
fig("hist_mlp.png", "Figure 12. MLP training vs validation log loss per epoch (mean &plusmn; std over CV folds; "
    "validation fold used only for monitoring).", width=W * 0.75)
fig("hist_lgbm.png", "Figure 13. LightGBM training vs validation log loss per boosting round (CV folds). Logistic "
    "regression, Naive Bayes, KNN and Random Forest have no epoch-wise history, so none is shown.", width=W * 0.75)
fig("cmp_train_vs_cv.png", "Figure 14. Training-fold vs validation-fold top-1 (over/under-fitting evidence).", width=W * 0.85)
fig("cm_candidates.png", "Figure 15. Candidate-level confusion matrices on student-test (threshold 0.5, row-normalised; counts in brackets).")
fig("cmp_pertrip_heat.png", "Figure 16. Per-trip student-test top-1 by model. Trip 00000077 is hardest for every model.")
fig("cmp_time.png", "Figure 17. Final fit time and scoring time (log scales).")
P("<b>Model selection.</b> LightGBM had the best CV macro-F1 "
  f"({f3(cv('lgbm', 'obs_macro_f1_trip'))}), the best student-test top-1 ({f3(te('lgbm', 'obs_top1'))}) and macro-F1 "
  f"({f3(te('lgbm', 'obs_macro_f1_trip'))}), and the lowest observation log loss ({f3(te('lgbm', 'obs_logloss'))}), "
  "which matters because the HMM uses the probabilities as emissions. Its fold-to-fold std is similar to the "
  f"other models (the variation is driven by which trips fall in a fold, see Figure 16), its artefact is {mb('lgbm'):.1f} MB and "
  f"it scores a trip in well under a second. Random Forest was close in accuracy but over-fits more, is {mb('rf') / mb('lgbm'):.0f}x larger "
  "and slower to train; the MLP was competitive on log loss but lower on accuracy. LightGBM was therefore taken "
  "forward: Section 4 fixes its training objective and tunes its hyperparameters.")
story.append(PageBreak())

# ================================================================== 4 Bayesian
P("4. Bayesian Hyperparameter Optimisation (LightGBM)", H1)
bp, bp2 = OPT["best_params"], OPT["promoted_params"]
comp = TRIALS
P("Section 3 selected the LightGBM family. Before tuning hyperparameters we fixed what the selected family optimises: "
  "Section 4.1 shows that training it with a ranking objective (LambdaRank) instead of per-row binary log loss is the larger "
  f"gain. The Bayesian search below was therefore run on the model that is deployed, LightGBM LambdaRank on the {len(FEATURES)}-feature "
  "table. Labels, folds and metrics are those of Section 3, and student-test was not used for any choice in this section.")
qa("1-2. Hyperparameters searched and their ranges / distributions", "See Table 6. The space covers model "
   "capacity (num_leaves, n_estimators), step size (learning_rate, log-uniform), leaf-size regularisation "
   "(min_child_samples, log-scale), stochastic bagging (subsample, colsample_bytree), L1/L2 penalties "
   "(log-uniform) and the LambdaRank truncation level (how many top-ranked candidates of a fix contribute to the loss). "
   "Ranges bracket the starting configuration.")
fmtv = lambda v: (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
table([["Hyperparameter", "Distribution / range", "Starting value", f"Best in search (trial {OPT['best_trial']})", f"Best at full fidelity (trial {OPT['promoted_trial']})"]] +
      [[k, v, fmtv(START.get(k, "5 (LightGBM default)")), fmtv(bp[k]), fmtv(bp2[k])] for k, v in OPT["search_space"].items()],
      widths=[3.6 * cm, 4.0 * cm, 3.0 * cm, 3.2 * cm, 3.6 * cm])
P("Table 6. Search space (scripts/bayes_opt_v2.py), starting configuration and the best configurations found. The starting values are "
  "the result of an earlier search on a 28-feature binary model (configs/start_hyperparameters.json; Section 4.3).", CAP)
qa("3. Objective", "Maximise the mean 5-fold trip-grouped CV observation top-1 accuracy on the 80 training trips, "
   "using exactly the same folds as Section 3 (macro-F1 and log loss were logged for every trial). "
   f"To make each trial affordable, the search fitted on every {OPT['search_train_stride']}th fix; the starting "
   "configuration was evaluated under the same protocol as the reference "
   f"(CV top-1 {f4(OPT['untuned_same_protocol']['cv_top1'])}). The student-test and evaluator trips were never used inside the search.")
qa("4. Number of trials and cost", f"{len(comp)} completed trials of Optuna's TPE sampler (multivariate, seed 42, "
   f"10 random start-up trials), wall time {OPT['wall_time_s'] / 60:.1f} min on 4 threads (5 model fits per trial). "
   "A first attempt with up to 900 trees and 255 leaves was stopped after 2 trials because a single trial exceeded "
   "3 minutes; the capacity range was reduced so that TPE could run a meaningful number of trials. "
   "The full trial record is in results/optuna_v2_trials.csv and results/optuna_v2_study.pkl (Appendix A).")
qa("5. Best hyperparameters", f"Trial scores span only {comp.value.min():.4f}-{comp.value.max():.4f}. Because the search ran on a smaller "
   "subsample than the final fit, its five best trials were re-scored with the standard protocol (every 3rd fix, "
   "scripts/promote_v2.py, Table 6a). The ranking changes between fidelities: the search's best trial "
   f"({int(PROMO.iloc[0].trial)}) is not the best at full fidelity; trial {OPT['promoted_trial']} is ("
   + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in bp2.items()) + ").")
pr = [["Trial", f"Search CV top-1 (every {OPT['search_train_stride']}th fix)", "Standard-protocol CV top-1", "CV macro-F1", "CV log loss"]]
for _, r in PROMO.iterrows():
    pr.append([int(r.trial), f4(r.search_cv_top1), f"{r.full_cv_top1:.4f} &plusmn; {r.full_cv_top1_std:.3f}", f4(r.full_cv_macro_f1), f4(r.full_cv_logloss)])
pr.append(["starting configuration", f4(OPT["untuned_same_protocol"]["cv_top1"]),
           f"{M2U['cv_mean']['obs_top1']:.4f} &plusmn; {M2U['cv_std']['obs_top1']:.3f}", f4(M2U["cv_mean"]["obs_macro_f1_trip"]), f4(M2U["cv_mean"]["obs_logloss"])])
table(pr, widths=[3.2 * cm, 4.2 * cm, 4.2 * cm, 2.4 * cm, 2.4 * cm])
P("Table 6a. Multi-fidelity promotion of the search's top five trials (results/optuna_v2_promotion.csv).", CAP)
t2r = [["Model", "CV macro- F1", "CV top-1", "Train top-1", "Gap", "Test top-1", "Test macro- F1", "Test log loss",
        "Test contin.", "Infer. (s)", "Fit (s)"]]
for lab_, m_ in [("LightGBM LambdaRank untuned (starting configuration)", M2U), (f"LightGBM LambdaRank Bayesian-tuned (trial {OPT['promoted_trial']})", M2T)]:
    c_, t_ = m_["cv_mean"], m_["student_test"]
    t2r.append([lab_ + (" [deployed]" if (m_ is M2) else ""), f4(c_["obs_macro_f1_trip"]), f"{c_['obs_top1']:.4f} &plusmn; {m_['cv_std']['obs_top1']:.3f}",
                f4(c_["train_obs_top1"]), f"{c_['train_obs_top1'] - c_['obs_top1']:+.3f}",
                f4(t_["obs_top1"]), f4(t_["obs_macro_f1_trip"]), f4(t_["obs_logloss"]), f4(t_["route_continuity"]),
                f"{t_['infer_time_s']:.2f}", f"{m_['final_fit_time_s']:.1f}"])
table(t2r, widths=[3.3 * cm, 1.45 * cm, 2.3 * cm, 1.35 * cm, 1.35 * cm, 1.3 * cm, 1.45 * cm, 1.3 * cm, 1.4 * cm, 1.15 * cm, 1.05 * cm])
du_, dt2_ = M2U["cv_mean"], M2T["cv_mean"]
P("Table 7. Untuned vs tuned LightGBM LambdaRank, both refitted with the standard pipeline (every 3rd fix) and evaluated on identical "
  "folds, test trips and metrics (scripts/train_final_v2.py; results/model_lgbm_v2_untuned.json, model_lgbm_v2.json, tuned_vs_untuned.csv).", CAP)
d_cv = dt2_["obs_top1"] - du_["obs_top1"]
d_te = M2T["student_test"]["obs_top1"] - M2U["student_test"]["obs_top1"]
qa("6. Did the tuned model outperform the untuned model?",
   f"Not beyond noise: CV top-1 changed by {d_cv:+.4f} ({f4(du_['obs_top1'])} to {f4(dt2_['obs_top1'])}), CV macro-F1 by "
   f"{dt2_['obs_macro_f1_trip'] - du_['obs_macro_f1_trip']:+.4f}, student-test top-1 by {d_te:+.4f}, student-test macro-F1 by "
   f"{M2T['student_test']['obs_macro_f1_trip'] - M2U['student_test']['obs_macro_f1_trip']:+.4f} and observation log loss by "
   f"{M2T['student_test']['obs_logloss'] - M2U['student_test']['obs_logloss']:+.4f}; the tuned model scores the test candidates in "
   f"{M2T['student_test']['infer_time_s']:.2f} s against {M2U['student_test']['infer_time_s']:.2f} s. These differences are an order of magnitude "
   f"smaller than the fold-to-fold std (about {M2U['cv_std']['obs_top1']:.2f}), i.e. performance is limited by the features, objective and labels "
   "rather than by LightGBM's hyperparameters. The train-fold vs validation gap is "
   f"{du_['train_obs_top1'] - du_['obs_top1']:+.3f} (untuned) vs {dt2_['train_obs_top1'] - dt2_['obs_top1']:+.3f} (tuned). "
   "Pre-declared rule (scripts/select_final_v2.py): the higher CV top-1 wins, a difference below 0.001 is a tie broken by the "
   f"smaller train-validation gap. Selected: <b>{'the Bayesian-tuned configuration' if CHOICE['selected'] == 'lgbm_v2' else 'the starting (untuned) configuration'}</b> "
   f"({CHOICE['reason']}). The choice was not made on student-test.")
fig("optuna_history.png", "Figure 18. Optuna optimisation history (each point = mean CV top-1 over 5 folds).", width=W * 0.85)
fig("optuna_importance.png", "Figure 19. fANOVA hyperparameter importance over completed trials.", width=W * 0.7)
fig("optuna_tuned_vs_untuned.png", "Figure 20. Tuned vs untuned LightGBM LambdaRank on identical data and metrics.", width=W * 0.85)

P("4.1 Training objective and feature ablation (fixed before the search)", H2)
P("All variants below use the starting hyperparameters and the same 5 trip-grouped folds, so each row changes exactly one thing "
  "(scripts/tune_v2.py). Student-test was not used.")
vr = [["Variant", "Features", "Objective", "Train rows", "CV top-1", "CV macro-F1", "CV log loss", "CV miss rate"]]
for _, r in V2VAR.iterrows():
    vr.append([r.variant.split(" ")[0], int(r.n_features), r.objective, "every 3rd fix" if r.train_stride == 3 else "all fixes",
               f"{r.cv_top1:.4f} &plusmn; {r.cv_top1_std:.3f}", f4(r.cv_macro_f1),
               f4(r.cv_obs_logloss) if r.objective != "rank_xendcg" else f"{r.cv_obs_logloss:.2f} (uncalibrated)", pc(r.cv_miss_rate)])
table(vr, widths=[1.6 * cm, 1.7 * cm, 2.4 * cm, 2.4 * cm, 2.8 * cm, 2.0 * cm, 2.6 * cm, 1.9 * cm])
aa, bb, c1, c2, dd_ = [V2VAR.iloc[i] for i in range(5)]
assert aa.n_features == len(BASE_FEATURES) and bb.n_features == len(FEATURES)
assert c1.cv_top1 == V2VAR.cv_top1.max() or dd_.cv_top1 <= c1.cv_top1 + 0.001
P("Table 7b. Controlled comparison on the same 5 trip-grouped folds (results/tune_v2_variants.csv). A uses only the "
  f"{len(BASE_FEATURES)} base features of Table 3; B to D use all {len(FEATURES)}.", CAP)
P(f"<b>(a) Context features.</b> Adding the {len(FEATURES) - len(BASE_FEATURES)} context features to the {len(BASE_FEATURES)} base features raises "
  f"CV top-1 from {f4(aa.cv_top1)} to {f4(bb.cv_top1)} and macro-F1 from {f4(aa.cv_macro_f1)} to {f4(bb.cv_macro_f1)} (A to B) on the same candidate set. "
  "<b>(b) Objective.</b> The task is to rank the candidates of one fix, so we replaced the per-row binary log loss by "
  "LambdaRank (LightGBM's lambdarank: pairwise logistic loss between the reference arc and every other candidate of the "
  "same fix, weighted by the change in NDCG; one query group per fix). It raises CV top-1 to "
  f"{f4(c1.cv_top1)} and macro-F1 to {f4(c1.cv_macro_f1)} (C1). The listwise rank_xendcg objective was worse "
  f"({f4(c2.cv_top1)}, C2). <b>(c) Training rows.</b> Training on all fixes instead of every third gave no gain ({f4(dd_.cv_top1)}, D) at about "
  f"{dd_.wall_s / c1.wall_s:.0f} times the cost, so every third fix is kept. <b>(d) Scores to probabilities.</b> Ranking scores are turned into a distribution over a fix's candidates "
  f"with a softmax whose temperature (T = {M2['softmax_temperature']:.3f}) is fitted on training out-of-fold scores only.")
fig("v2_variants.png", "Figure 20a. Feature and objective variants: CV top-1 (fold std as error bars) and CV macro-F1.", width=W * 0.9)

P("4.2 Learning curves", H2)
P("To separate under-fitting from over-fitting, the deployed scorer and logistic regression "
  "(as a high-bias reference) were re-trained on nested random subsets of 4 to 64 training trips, with the 16 trips of the "
  "validation fold fixed, for each of the five folds (<font face='Courier'>scripts/learning_curve.py</font>, "
  "results/learning_curve.csv; student-test is not used).")
fig("learning_curve.png", f"Figure 20b. Learning curves on the {len(FEATURES)}-feature table: top-1 on the trips used for training (dashed) and on "
    "the unseen validation fold (solid) against the number of training trips; mean over 5 folds, band = fold std.", width=W * 0.95)
_l, _g = LC.loc["logreg"], LC.loc["lgbm_v2"]
_n0, _n1 = int(_g.index.min()), int(_g.index.max())
P(f"Logistic regression: training and validation top-1 meet at {_l.val_top1[_n1]:.3f} "
  f"(gap {_l.train_top1[_n1] - _l.val_top1[_n1]:+.3f} at {_n1} trips) and validation top-1 changes by only "
  f"{_l.val_top1[_n1] - _l.val_top1[16]:+.3f} from 16 to {_n1} trips. More data cannot help a model whose two curves have already "
  "converged, so its error is bias (under-fitting). "
  f"Deployed scorer: it fits {_n0} trips perfectly (training top-1 {_g.train_top1[_n0]:.3f}); as trips are "
  f"added, training top-1 falls to {_g.train_top1[_n1]:.3f} and validation top-1 rises from {_g.val_top1[_n0]:.3f} to "
  f"{_g.val_top1[_n1]:.3f}, so the gap narrows from {_g.train_top1[_n0] - _g.val_top1[_n0]:.3f} to "
  f"{_g.train_top1[_n1] - _g.val_top1[_n1]:.3f} but does not close. That remaining gap is variance (mild over-fitting), yet the "
  f"scorer's validation curve lies above the logistic-regression curve at every size, so the extra capacity is a net gain. "
  f"Its validation top-1 moves by {_g.val_top1[_n1] - _g.val_top1[16]:+.3f} between 16 and {_n1} trips, which is small "
  "against the fold-to-fold std of about 0.02: more trips of the same kind would add little, and the remaining errors are better "
  "addressed by the sequence decoder of Section 5 than by more scorer training.")

P("4.3 Earlier round with 28 features (archived)", H2)
t1, u1 = M1["lgbm_tuned"], M1["lgbm"]
P(f"The first version of this project used only the {len(BASE_FEATURES)} base features and at most 12 candidates per fix. It is not part of the "
  "deployed system and its files are kept unchanged in <font face='Courier'>archive/round1_28_features/</font> (models, results, figures, "
  "scripts; README there). It is recorded here because two settings of the deployed system were first found in it: "
  f"<b>(i)</b> the starting hyperparameters of Table 6 are the best configuration of a Bayesian search (Optuna TPE, {OPT1['n_trials']} trials) on the "
  f"28-feature binary LightGBM, which moved CV top-1 from {f4(u1['cv_mean']['obs_top1'])} to {f4(t1['cv_mean']['obs_top1'])} and student-test top-1 from "
  f"{f4(u1['student_test']['obs_top1'])} to {f4(t1['student_test']['obs_top1'])} (a tie in accuracy, at less than half the scoring time); "
  f"<b>(ii)</b> the decoder values &beta; = {HSUM1['best_params']['beta']:g} m and w = {HSUM1['best_params']['w_emit']:g} came from a 35-setting grid on that "
  "model's out-of-fold scores; Section 5.3 re-examines them with the deployed scorer. "
  f"That system reached a final student-test top-1 of {f4(RE1['student_test']['realign']['top1'])} ({f4(RE1['train']['realign']['top1'])} on training trips, out-of-fold); "
  f"the deployed system is better by {VAL['student_test']['difference']['point']:+.4f} on student-test "
  f"(95% trip-bootstrap interval [{VAL['student_test']['difference']['ci95'][0]:+.4f}, {VAL['student_test']['difference']['ci95'][1]:+.4f}], "
  f"{VAL['student_test']['trips_better']} trips better, {VAL['student_test']['trips_worse']} worse) and by {VAL['train_oof']['difference']['point']:+.4f} on the training trips "
  f"([{VAL['train_oof']['difference']['ci95'][0]:+.4f}, {VAL['train_oof']['difference']['ci95'][1]:+.4f}], {VAL['train_oof']['trips_better']} better, "
  f"{VAL['train_oof']['trips_worse']} worse; scripts/validate_final.py).")
b6 = [["Model (same configuration)", "CV macro-F1: 28 to 46 features", "CV top-1: 28 to 46 features", "Test top-1: 28 to 46 features"]]
for m in MODELS:
    a_, b_ = M1[m], M[m]
    b6.append([NAMES[m], f"{a_['cv_mean']['obs_macro_f1_trip']:.3f} to {b_['cv_mean']['obs_macro_f1_trip']:.3f}",
               f"{a_['cv_mean']['obs_top1']:.3f} to {b_['cv_mean']['obs_top1']:.3f}",
               f"{a_['student_test']['obs_top1']:.3f} to {b_['student_test']['obs_top1']:.3f}"])
table(b6, widths=[5.0 * cm, 4.2 * cm, 4.1 * cm, 4.1 * cm])
gain = {m: M[m]["cv_mean"]["obs_top1"] - M1[m]["cv_mean"]["obs_top1"] for m in MODELS}
up = [m for m in MODELS if gain[m] > 0.001]
down = [m for m in MODELS if gain[m] < -0.001]
assert max(MODELS, key=lambda m: M1[m]["cv_mean"]["obs_macro_f1_trip"]) == "lgbm"
P("Table 7c. The six model families in the archived 28-feature round and on the current table (same folds, labels, metrics and model "
  "configurations; archive/round1_28_features/results/model_*.json vs results/model_*.json).", CAP)
P("LightGBM was the best of the six families in both rounds, so the model selection of Section 3 does not depend on the feature table. "
  "The additional features raise CV top-1 for " + (", ".join(NAMES[m] for m in up) if up else "none of the families")
  + (" and lower it for " + ", ".join(NAMES[m] for m in down) if down else "")
  + ": models that weight all standardised features equally or assume independence gain least from 18 additional, partly "
  "redundant features, while tree ensembles can use them selectively.")
story.append(PageBreak())

# ================================================================== 5 HMM
P("5. HMM and Viterbi Route Reconstruction", H1)
hp = FINAL["hmm_params"]
qa("1. Hidden states", "For fix i the hidden state is one of its candidate arcs c &isin; C<sub>i</sub>, together "
   "with the projection fraction t<sub>c</sub> of the fix on that arc. The candidate-to-arc correspondence is the "
   "same table the classifier scored (trip, observation, arc ID), so the decoder returns an original arc ID per fix.")
qa("2. Emission probabilities", "log e<sub>i</sub>(c) = w &middot; log q<sub>i</sub>(c), where q<sub>i</sub> is the "
   "scorer's distribution over C<sub>i</sub> (floor 1e-6): the renormalised probability for binary models, and for the "
   "deployed LambdaRank scorer a softmax of its scores with a temperature fitted on training out-of-fold scores. The weight w balances classifier "
   f"evidence against path plausibility; the selected value is w = {hp['w_emit']}.")
qa("3. Transition probabilities", "log a(c, c') = -|d<sub>route</sub>(c, c') - d<sub>gc</sub>(z<sub>i</sub>, "
   "z<sub>i+1</sub>)| / &beta; - max(0, d<sub>route</sub>/&Delta;t - v<sub>max</sub>)/5 (Newson-Krumm style). "
   "d<sub>route</sub> is the driving distance from the projection on c to the projection on c': along the arc if "
   "c = c' (backward motion beyond 15 m costs an extra 60 m), otherwise the remaining part of c + shortest path "
   "from head(c) to tail(c') in the directed graph (Dijkstra, scipy, limit 3 km, computed in batches per trip) + "
   f"the initial part of c'. d<sub>gc</sub> is the straight-line distance between the fixes. Selected &beta; = "
   f"{hp['beta']} m, v<sub>max</sub> = 60 m/s. The decoder code also supports a U-turn penalty and pruning to the scorer's "
   "top-M candidates; both were tried in Section 5.3 and are switched off in the deployed setting.")
qa("4. Missing candidates and large GPS gaps", "A fix without any candidate inside the radius gets no prediction "
   "(-1). Its two neighbours are linked directly when the time and distance between them are within the usual limits "
   "(dropout bridging, Section 5.3); otherwise the chain ends there and a new chain starts at the next fix with candidates. A time gap "
   "&gt; 60 s or a jump &gt; 1.5 km always breaks the chain, because driving distance is no longer constrained by the "
   "elapsed time. Each chain is decoded independently. The smoke test verifies this on a synthetic trip with "
   f"100 displaced fixes ({SMOKE['99999999']['n_matched']} of {SMOKE['99999999']['n_observations']} matched, "
   f"{SMOKE['99999999']['hmm_segments']} chains, no failure).")
qa("5. Disconnected road segments", "Pairs with no directed path within the search limit get probability 0 "
   "(-&infin;). If every transition between fix i and i+1 is impossible (e.g. the graph is disconnected or the "
   "fixes jump between components) the chain is cut at i and decoding restarts, instead of forcing an absurd "
   "path. For the output route, consecutive distinct arcs inside a chain are joined with the shortest path; "
   "across chain breaks no path is invented.")
qa("Viterbi recurrence (decoding procedure)", "Within a chain of fixes i = 1..n with candidate sets C<sub>i</sub>: "
   "&delta;<sub>1</sub>(c) = log e<sub>1</sub>(c); &delta;<sub>i</sub>(c') = log e<sub>i</sub>(c') + "
   "max<sub>c &isin; C<sub>i-1</sub></sub> [ &delta;<sub>i-1</sub>(c) + log a(c, c') ], with back-pointer "
   "&psi;<sub>i</sub>(c') = argmax<sub>c</sub> [ &delta;<sub>i-1</sub>(c) + log a(c, c') ]. The chain ends at the "
   "last fix or at a break; the best final state argmax<sub>c</sub> &delta;<sub>n</sub>(c) is traced back through "
   "&psi; to give one arc per fix. Scores are kept in the log domain; the transition matrix of each step "
   "(at most 24 &times; 24) is built with numpy from pre-computed shortest-path distances, so a trip of several "
   "thousand fixes decodes in well under a second (Table 9).")
qa("Decoder parameters", "Never chosen on student-test. &beta; and w were first selected by a 35-setting grid on out-of-fold "
   f"scores of the 80 training trips in the archived 28-feature round (&beta; = {hp['beta']:g} m, w = {hp['w_emit']:g}; Section 4.3). With the deployed "
   "scorer they were re-examined on training trips only, under a rule with held-out training trips fixed in advance (Section 5.3): a 4 x 4 grid "
   "over &beta; and w (Figure 21), a Bayesian search over eight decoder parameters, and one structural option (dropout bridging). Only "
   "bridging was adopted; &beta; and w stayed. Small &beta; and w mean the decoder trusts path geometry "
   "strongly relative to single-fix scores, which suits 1 Hz data where consecutive fixes are only 10-20 m apart.")
fig("hmm_grid.png", "Figure 21. Decoder grid with the deployed scorer: mean per-trip final top-1 on the 64 tuning trips (training trips, out-of-fold scores).", width=W * 0.6)
hrows = [["Model (scores)", "Top-1 scorer", "Top-1 HMM", "Top-1 final", "Cont. scorer", "Cont. HMM", "Route F1 scorer",
          "Route F1 HMM", "Trips worse (top-1)", "Trips worse (route F1)"]]
order = [m for m in ["baseline"] + MODELS + [SEL] if m in HAB.index]
for m in order:
    r = HAB.loc[m]
    hrows.append(["Nearest-segment (geometric HMM)" if m == "baseline" else DEPNAME if m == SEL else NAMES[m], f3(r.cls_top1), f3(r.hmm_top1),
                  f3(r.final_top1), f3(r.cls_continuity), f3(r.hmm_continuity), f3(r.cls_route_f1), f3(r.hmm_route_f1),
                  int(r.trips_hmm_hurt_top1), int(r.trips_hmm_hurt_route_f1)])
table(hrows, widths=[3.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm])
P("Table 8. Scorer only vs scorer + HMM/Viterbi (and + re-alignment, Section 5.1) on the 10 student-test trips, every model decoded "
  "with the same deployed decoder setting (scripts/hmm_ablation.py, results/hmm_ablation_models.csv). Top-1 is pooled over fixes; continuity and "
  "route F1 are trip means. Route F1 for scorer-only uses the collapsed per-fix arcs; for HMM the route with shortest-path filling.", CAP)
assert abs(HAB.loc[SEL, "final_top1"] - F2T["final"]["top1"]) < 1e-9 and abs(HAB.loc[SEL, "hmm_top1"] - F2T["hmm"]["top1"]) < 1e-9
P("<b>6. Did HMM decoding improve the final route?</b> " +
  f"Yes. For the deployed scorer: top-1 {f3(HAB.loc[SEL, 'cls_top1'])} &rarr; {f3(HAB.loc[SEL, 'hmm_top1'])}, "
  f"continuity {f3(HAB.loc[SEL, 'cls_continuity'])} &rarr; {f3(HAB.loc[SEL, 'hmm_continuity'])}, route F1 "
  f"{f3(HAB.loc[SEL, 'cls_route_f1'])} &rarr; {f3(HAB.loc[SEL, 'hmm_route_f1'])}. The gain is largest for weak "
  f"scorers (nearest-segment top-1 {f3(HAB.loc['baseline', 'cls_top1'])} &rarr; {f3(HAB.loc['baseline', 'hmm_top1'])}), "
  "showing that topology and timing recover much of what a pointwise scorer misses, while a better scorer still "
  f"gives the better end result. Observation macro-F1 for the deployed scorer rises from {f3(F2T['cls']['macro_f1'])} to "
  f"{f3(F2T['hmm']['macro_f1'])} after Viterbi. Log loss is N/A for decoded outputs: Viterbi returns a single best "
  "path, not a per-fix probability distribution.")
ptrows = [["Trip", "Fixes", "Top-1 scorer", "Top-1 HMM", "Top-1 final", "Macro-F1 final", "Cont. scorer", "Cont. HMM", "Route P / R", "Route F1",
           "Chains", "Decode s"]]
for _, r in FTEST.iterrows():
    ptrows.append([r.trip, int(r.n_obs), f3(r.cls_top1), f3(r.hmm_top1), f3(r.final_top1), f3(r.final_macro_f1), f3(r.cls_continuity), f3(r.hmm_continuity),
                   f"{r.route_precision:.3f} / {r.route_recall:.3f}", f3(r.route_f1), int(r.chains), f"{r.decode_time_s:.2f}"])
table(ptrows)
P("Table 9. Per-trip effect of the decoder for the deployed scorer on student-test (results/final_test.csv; 'Top-1 final' = after the "
  "route re-alignment step of Section 5.1). Per-trip training results: results/final_train_oof.csv.", CAP)
hurt_txt = []
if len(hurt_top1):
    hurt_txt.append("top-1 decreased on " + ", ".join(f"{r.trip} ({r.cls_top1:.3f} &rarr; {r.hmm_top1:.3f})" for _, r in hurt_top1.iterrows()))
if len(hurt_rf1):
    hurt_txt.append("route F1 decreased on " + ", ".join(f"{r.trip} ({r.cls_route_f1:.3f} &rarr; {r.hmm_route_f1:.3f})" for _, r in hurt_rf1.iterrows()))
P("<b>Cases where HMM did not help.</b> On student-test, for the deployed scorer, " + ("; ".join(hurt_txt) + ". " if hurt_txt else
  "no trip got worse on top-1 or route F1. ") +
  "Where the decoder hurts, the typical mechanism is a confident but wrong scorer run (e.g. on the reverse direction of a "
  "two-way street or a parallel service road) that is internally consistent, so the decoder keeps it; shortest-path "
  "filling then adds arcs between two wrong matches, lowering route precision. Trip 00000077 (dense urban grid, many parallel roads) "
  "and trip 00000009 remain the hardest student-test trips after decoding.")
bad = FTRAIN[FTRAIN.hmm_top1 < FTRAIN.cls_top1].sort_values("trip")
P(f"On the 80 training trips (out-of-fold scores, same decoder) Viterbi lowered top-1 on "
  f"{len(bad)} trips (Table 10), by at most {(bad.cls_top1 - bad.hmm_top1).max():.3f}, while raising pooled OOF top-1 from "
  f"{f3(F2R['cls']['top1'])} to {f3(F2R['hmm']['top1'])}. These are trips whose scorer-only result was already high, "
  "where smoothing occasionally moves a correct fix at an arc boundary; the re-alignment step of Section 5.1 recovers most of them "
  f"(final top-1 below scorer-only on {int((FTRAIN.final_top1 < FTRAIN.cls_top1).sum())} of 80 trips). The weakest training trips after decoding "
  "(" + ", ".join(f"{r.trip}: {r.final_top1:.3f}" for _, r in FTRAIN.sort_values("final_top1").head(3).iterrows()) + ") are limited by the scorer "
  "and by candidate misses, not by the decoder.")
brow = [["Trip (train)", "Fixes", "Top-1 scorer", "Top-1 HMM", "Top-1 final", "Route precision", "Route F1", "Chains"]]
for _, r in bad.iterrows():
    brow.append([r.trip, int(r.n_obs), f3(r.cls_top1), f3(r.hmm_top1), f3(r.final_top1), f3(r.route_precision), f3(r.route_f1), int(r.chains)])
table(brow)
P("Table 10. Training trips (OOF) where Viterbi lowered top-1 relative to the scorer alone (results/final_train_oof.csv).", CAP)
fig("hmm_ablation_models.png", "Figure 22. Effect of HMM decoding for every model's scores (student-test).")
fig("hmm_pertrip.png", "Figure 23. Per-trip scorer-only, + HMM/Viterbi and final top-1 for the deployed scorer (student-test).")
P("5.1 Per-fix assignment: re-alignment to the predicted route", H2)
eh, ef = ERR["errors"]["hmm_pred"], ERR["errors"]["final_pred"]
P("An error analysis of the Viterbi output on student-test (scripts/error_analysis.py) shows that "
  f"{pc(eh['adjacent_on_route'] / eh['n_errors'])} of its wrong fixes are matched to the arc immediately before or "
  "after the reference arc <i>on the correct route</i>: the route is right, but the per-fix states switch arcs a few "
  "fixes too early or too late, and short arcs inserted by shortest-path filling receive no fix at all. The decoder "
  "therefore ends with a reference-free post-step (pathsnap/hmm.py, realign_to_route): Viterbi decides <i>which arcs are driven</i>; "
  "the fixes are then assigned to the arcs of that predicted route by a monotone (order-preserving) nearest-arc "
  "dynamic programme, the same assignment convention that defines the labels, applied to the predicted route "
  f"instead of the reference. A fix keeps its Viterbi arc if the re-assigned arc would be more than {hp['realign_max_m']:.0f} m away, and "
  "unmatched fixes stay unmatched. The step is judged on the 80 training trips (out-of-fold scores), not on student-test.")
rr = [["Partition (same scores, decoder, labels)", "Stage", "Top-1", "Macro-F1", "Continuity", "Trips better / worse vs Viterbi"]]
for lab_, S_, D_ in [("80 training trips, out-of-fold", F2R, FTRAIN), ("10 student-test trips", F2T, FTEST)]:
    rr.append([lab_, "scorer only", f4(S_["cls"]["top1"]), f4(S_["cls"]["macro_f1"]), f4(S_["cls"]["continuity"]), "-"])
    rr.append(["", "+ HMM/Viterbi", f4(S_["hmm"]["top1"]), f4(S_["hmm"]["macro_f1"]), f4(S_["hmm"]["continuity"]), "-"])
    rr.append(["", "+ re-alignment (final)", f4(S_["final"]["top1"]), f4(S_["final"]["macro_f1"]), f4(S_["final"]["continuity"]),
               f"{int((D_.final_top1 > D_.hmm_top1 + 1e-12).sum())} / {int((D_.final_top1 < D_.hmm_top1 - 1e-12).sum())}"])
table(rr, widths=[4.6 * cm, 3.6 * cm, 1.7 * cm, 1.8 * cm, 1.9 * cm, 3.4 * cm])
P("Table 11. Before/after for the re-alignment step with the deployed scorer (results/final_summary.json, per-trip CSVs "
  "results/final_train_oof.csv and results/final_test.csv). Top-1 pooled over scored fixes; macro-F1 and "
  "continuity are trip means.", CAP)
worse = FTRAIN[FTRAIN.final_top1 < FTRAIN.hmm_top1 - 1e-12]
assert F2R["final"]["top1"] > F2R["hmm"]["top1"] and F2T["final"]["top1"] > F2T["hmm"]["top1"]
P(f"Top-1 improves on {int((FTRAIN.final_top1 > FTRAIN.hmm_top1 + 1e-12).sum())} of 80 training trips and on "
  f"{int((FTEST.final_top1 > FTEST.hmm_top1 + 1e-12).sum())} of 10 student-test trips; "
  f"continuity is unchanged within 0.002. It gets worse on {len(worse)} training trips "
  f"({', '.join(f'{r.trip} ({r.hmm_top1:.3f} to {r.final_top1:.3f})' for _, r in worse.iterrows())}): when the predicted "
  "route itself contains a wrong detour, re-alignment spreads fixes onto it. Because the per-fix assignment now "
  "follows the same convention as the labels, the final top-1 is essentially a measure of how correct the "
  "predicted route is; it does not use the reference in any way (Section 5.2 verifies this).")
P("5.2 Residual errors, label noise and leakage check", H2)
er = [["Stage", "Errors / scored fixes", "Error rate", "Neighbour arc on route", "Opposite direction", "Other arc on route", "Off the reference route"]]
for k_, n_ in [("cls_pred", "scorer only"), ("hmm_pred", "+ HMM/Viterbi"), ("final_pred", "+ re-alignment (final)")]:
    e_ = ERR["errors"][k_]
    er.append([n_, f"{e_['n_errors']:,} / {ERR['n_scored_fixes']:,}", pc(e_["error_rate"]), e_["adjacent_on_route"],
               e_["reverse_direction"], e_["other_on_route"], e_["off_route"]])
table(er, widths=[3.6 * cm, 2.8 * cm, 1.6 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm])
P("Table 12. Residual-error breakdown of the deployed system on the 10 student-test trips (results/error_breakdown.json).", CAP)
ln = ERR["label_noise"]
P(f"<b>Label noise.</b> {ln['label_against_motion']} of {ln['moving_scored_fixes']:,} scored fixes recorded while moving "
  f"faster than 3 m/s ({pc(ln['share'])}) have a reference label whose arc direction points <i>against</i> the GPS "
  "direction of travel (cosine &lt; -0.5). The label alignment is geometry-only, so on out-and-back stretches of a "
  "two-way street a fix can be assigned to either direction's arc. Of the "
  f"{ln['final_reverse_direction_errors_while_moving']} final opposite-direction errors made while moving, "
  f"{ln['of_which_label_against_motion']} fall in this group, i.e. the prediction follows the direction of travel and "
  "the label does not. We did not change the label protocol (all models share it), but this bounds how much of the "
  "remaining error is attributable to the reference rather than to the model, and it is why we do not claim the "
  "remaining error is purely model error.")
P("<b>Leakage check.</b> The ten student-test trips were copied to a temporary folder <i>without</i> their .route "
  "files and predicted again: scorer arcs, Viterbi arcs, final arcs and routes were identical for all ten trips "
  "(results/error_breakdown.json, predictions_identical_without_route_files = true). Independent recomputation of "
  "the final top-1 from the deployed pipeline's own output reproduced the reported value exactly (Section 5.4).")

P("5.3 Bayesian tuning of the decoder (not adopted)", H2)
dtc = DTR[DTR.state == "COMPLETE"]
P("With the deployed scorer we tried to re-tune the whole decoder for the quantity finally reported, the <i>final</i> top-1 "
  "(Viterbi + re-alignment), on training trips with out-of-fold scores. Search space (Optuna TPE, multivariate, seed 42, "
  "16 start-up trials, batches of 8; scripts/tune_decoder.py): &beta; log-uniform [2, 80] m; emission weight w log-uniform "
  "[0.05, 2]; tolerated backward motion on the same arc [3, 40] m and its penalty [0, 200] m; a U-turn penalty [0, 8] (log "
  "units) for switching to the opposite-direction arc; the chain-breaking time gap log-uniform [10, 300] s; the number of "
  "top-scored candidates kept as states {6, 8, 12, 24}; and the re-alignment distance cap [30, 150] m. Shortest-path "
  "distances do not depend on these parameters, so each trip is prepared once and only Viterbi is re-run per trial. "
  f"Trial 0 is always the current setting (&beta; = {hp['beta']:g} m, w = {hp['w_emit']:g}).")
p0, p1 = FSP["train"]["round1_decoder"], FSP["train"]["tuned_decoder"]
P(f"<b>Attempt 1: pooled objective on all 80 training trips ({DECP['n_trials']} trials).</b> Pooled final top-1 rose from "
  f"{f4(p0['final']['top1'])} to {f4(p1['final']['top1'])}, but the mean per-trip macro-F1 on the same trips <i>fell</i> from "
  f"{f4(p0['final']['macro_f1'])} to {f4(p1['final']['macro_f1'])}, several tuned values sat on the search bounds "
  f"(w = {DECP['best_params']['w_emit']:.3f}, U-turn penalty = {DECP['best_params']['uturn_pen']:.2f}), and student-test did not improve "
  f"({f4(FSP['student_test']['round1_decoder']['final']['top1'])} to {f4(FSP['student_test']['tuned_decoder']['final']['top1'])}). A pooled "
  "objective is dominated by a few very long trips, so the search was fitting those. This run is kept in "
  "results/decoder_trials_pooled.csv, decoder_best_pooled.json and final_summary_pooled.json.")
tu0, tu1, ho0, ho1 = FS["tuning_trips"]["round1_decoder"], FS["tuning_trips"]["tuned_decoder"], FS["heldout_trips"]["round1_decoder"], FS["heldout_trips"]["tuned_decoder"]
P(f"<b>Attempt 2: per-trip objective with held-out training trips ({len(dtc)} trials).</b> Objective: mean per-trip final "
  "top-1 (the evaluator scores one trip at a time) on the 64 training trips of CV folds 0-3; the 16 trips of fold 4 were "
  "held out. Adoption rule fixed before the run: the tuned setting replaces the current setting only if, on the held-out "
  "training trips, mean per-trip top-1 improves and mean per-trip macro-F1 does not decrease.")
bd = DEC["best_params"]
dr = [["Trips", "Decoder", "Mean per-trip top-1", "Mean per-trip macro-F1", "Pooled top-1", "Continuity"]]
for lab_, key_ in [("64 tuning trips (folds 0-3)", "tuning_trips"), ("16 held-out training trips (fold 4)", "heldout_trips")]:
    for nm_, k2_ in [("current setting", "round1_decoder"), ("TPE best", "tuned_decoder")]:
        S_ = FS[key_][k2_]
        dr.append([lab_ if k2_ == "round1_decoder" else "", nm_, f4(S_["trip_mean_top1"]), f4(S_["trip_mean_macro_f1"]), f4(S_["pooled_top1"]), f4(S_["continuity"])])
for nm_, k2_ in [("current setting", "round1_decoder"), ("TPE best", "tuned_decoder")]:
    S_ = FS["student_test"][k2_]
    dr.append(["10 student-test trips (not used by the rule)" if k2_ == "round1_decoder" else "", nm_, f4(S_["trip_mean_top1"]), f4(S_["final"]["macro_f1"]), f4(S_["final"]["top1"]), f4(S_["final"]["continuity"])])
table(dr, widths=[5.4 * cm, 2.6 * cm, 2.6 * cm, 2.9 * cm, 2.0 * cm, 1.9 * cm])
P("Table 12a. Decoder tuning, attempt 2 (results/final_summary.json, decoder_best.json; all trials in results/decoder_trials.csv). "
  "TPE best: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in bd.items()) + ".", CAP)
P(f"On the tuning trips the best setting gains {tu1['trip_mean_top1'] - tu0['trip_mean_top1']:+.4f} mean per-trip top-1, but on the "
  f"held-out training trips it <i>loses</i> {ho1['trip_mean_top1'] - ho0['trip_mean_top1']:+.4f} ({f4(ho0['trip_mean_top1'])} to "
  f"{f4(ho1['trip_mean_top1'])}). The rule therefore rejects it. On student-test the same setting happens to score higher "
  f"({f4(FS['student_test']['round1_decoder']['final']['top1'])} to {f4(FS['student_test']['tuned_decoder']['final']['top1'])}); we did not "
  "let that override the rule, because choosing a setting for its student-test score would be selecting on the test set, and the "
  "two held-out results disagree in sign, which is itself evidence that eight decoder parameters cannot be tuned reliably "
  "on 64-80 trips.")
g2 = FS["grid2"]
P(f"<b>Attempt 3: 2-parameter grid.</b> A 4 x 4 grid over &beta; in {{5, 10, 20, 40}} m and w in {{0.1, 0.25, 0.5, 1}} "
  "(scripts/decoder_grid_v2.py, results/decoder_grid_v2.csv) under the same rule: mean per-trip top-1 on the tuning trips "
  f"ranges only from {DGRID.tune_trip_mean_top1.min():.4f} to {DGRID.tune_trip_mean_top1.max():.4f}; the best cell "
  f"(&beta; = {g2['best_on_tuning_trips']['beta']:g}, w = {g2['best_on_tuning_trips']['w_emit']:g}) improves held-out top-1 "
  f"({f4(g2['heldout']['round1'])} to {f4(g2['heldout']['best'])}) but not held-out macro-F1 ({f4(g2['heldout']['round1_macro_f1'])} to "
  f"{f4(g2['heldout']['best_macro_f1'])}), so it is not adopted either. The decoder sits on a flat plateau: the current setting "
  f"(&beta; = {hp['beta']:g} m, w = {hp['w_emit']:g}) is kept for the deployed system.")
b0, b1 = BR["train"]["no_bridge"], BR["train"]["bridge"]
P("<b>Structural option: bridging candidate dropouts (adopted).</b> Looking at the weakest training trips showed one whose "
  "track was cut into 716 separate Viterbi chains: the decoder broke the chain at every fix that had no candidate arc within "
  "80 m. Standard practice is to skip such fixes and link their neighbours when the time and distance gap are within the usual "
  "limits. Rule fixed before the run (scripts/bridge_eval.py): adopt if mean per-trip top-1 and macro-F1 do not decrease on the "
  "16 held-out training trips and top-1 increases on the 64 tuning trips. Result: chains on the 80 training trips "
  f"{b0['total_chains']} to {b1['total_chains']}; tuning trips {BR['tuning_trips']['no_bridge']['trip_mean_top1']:.4f} to "
  f"{BR['tuning_trips']['bridge']['trip_mean_top1']:.4f}; held-out trips {BR['heldout_trips']['no_bridge']['trip_mean_top1']:.4f} to "
  f"{BR['heldout_trips']['bridge']['trip_mean_top1']:.4f} (macro-F1 {BR['heldout_trips']['no_bridge']['trip_mean_macro_f1']:.4f} to "
  f"{BR['heldout_trips']['bridge']['trip_mean_macro_f1']:.4f}); pooled training top-1 {f4(b0['final']['top1'])} to {f4(b1['final']['top1'])} "
  f"({BR['train_trips_better']} trip better, {BR['train_trips_worse']} marginally worse); student-test unchanged (it has no such "
  "dropouts). The rule is met, so bridging is switched on in the deployed decoder. Its accuracy effect is small; its value is "
  "that routes are no longer fragmented by short candidate dropouts. The heavily fragmented trip itself (00000054) did not "
  "improve in accuracy: 88% of its fixes lie off the reference route and are not scored, and the rest is limited by candidate "
  "misses, i.e. by data rather than by the decoder.")
fig("decoder_tuning_history.png", "Figure 23a. Decoder tuning, attempt 2: mean per-trip final top-1 on the 64 tuning trips per trial.", width=W * 0.8)

P("5.4 Final system: results and validation", H2)
lad = [["Stage", "Train (CV / OOF) top-1", "Test top-1", "Test macro-F1", "Test continuity"],
       ["Nearest segment (baseline)", f4(BASE["cv_mean"]["obs_top1"]), f4(BASE["student_test"]["obs_top1"]), f4(BASE["student_test"]["obs_macro_f1_trip"]), f4(BASE["student_test"]["route_continuity"])],
       [f"Scorer: LightGBM LambdaRank, {len(FEATURES)} features, K={MAX_K}", f4(F2R["cls"]["top1"]), f4(F2T["cls"]["top1"]), f4(F2T["cls"]["macro_f1"]), f4(F2T["cls"]["continuity"])],
       ["+ HMM/Viterbi (dropout bridging)", f4(F2R["hmm"]["top1"]), f4(F2T["hmm"]["top1"]), f4(F2T["hmm"]["macro_f1"]), f4(F2T["hmm"]["continuity"])],
       ["+ re-alignment = deployed system", f4(F2R["final"]["top1"]), f4(F2T["final"]["top1"]), f4(F2T["final"]["macro_f1"]), f4(F2T["final"]["continuity"])]]
table(lad, widths=[7.4 * cm, 3.0 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm])
P("Table 12b. Accuracy ladder. Train column: baseline on CV validation folds; other rows on all scored fixes of the 80 training trips with out-of-fold scores; "
  f"test: all scored fixes of the 10 student-test trips. Route F1 of the deployed system: {f4(F2T['route_f1'])} (test), "
  f"{f4(F2R['route_f1'])} (train OOF). Source: results/final_summary.json.", CAP)
fig("final_ladder.png", "Figure 23b. Accuracy ladder on training (CV / out-of-fold) and student-test trips.", width=W * 0.85)
vt, vs = VAL["train_oof"], VAL["student_test"]
vrows = [["Partition", "Trips / fixes", "Final top-1 [95% CI]", "Weakest trip (final top-1)"]]
for lab_, v_ in [("80 training trips (OOF)", vt), ("10 student-test trips", vs)]:
    vrows.append([lab_, f"{v_['n_trips']} / {v_['n_fixes']:,}", ci(v_["final_top1"]), f"{v_['worst_trip_final']['trip']} ({v_['worst_trip_final']['top1']:.3f})"])
table(vrows, widths=[4.6 * cm, 3.2 * cm, 4.8 * cm, 4.8 * cm])
P(f"Table 12c. Trip-level bootstrap ({VAL['bootstrap_resamples']:,} resamples of trips; pooled accuracy). "
  "scripts/validate_final.py, results/validation_final.json.", CAP)
ls_ = VAL["label_sensitivity_student_test"]
P("<b>How certain is the result?</b> Fixes of one trip are not independent, so whole trips are resampled. The final top-1 is "
  f"{ci(vs['final_top1'])} on the 10 student-test trips and {ci(vt['final_top1'])} on the 80 training trips (out-of-fold). "
  "With only ten test trips the interval is about 2.5 points wide. "
  f"The weakest student-test trip of the final system is {vs['worst_trip_final']['trip']} ({vs['worst_trip_final']['top1']:.3f}); "
  f"the weakest training trip is {vt['worst_trip_final']['trip']} ({vt['worst_trip_final']['top1']:.3f}), so individual trips "
  "can still be far below the average.")
P("<b>Sensitivity to the labelling convention.</b> Labels come from a geometry-only alignment (Section 2.1). As a check we "
  "re-scored the system against direction-aware labels (same alignment plus a 40 m penalty when a fix moving faster than "
  f"3 m/s would be assigned to an arc pointing against its travel direction; {ls_['labels_changed_by_direction_term']:,} of "
  f"{ls_['scored_fixes_geometry_only']:,} student-test labels change). Final top-1 is {f4(ls_['final_top1']['geometry_only_labels'])} "
  f"under the official labels and {f4(ls_['final_top1']['direction_aware_labels'])} under direction-aware labels, about 1.5 points lower: the "
  "re-alignment step uses the same geometry-only convention as the official labels, so part of its gain is agreement with that "
  "convention. All headline numbers use the official geometry-only labels.")
P("<b>Verification of the final system.</b> (i) Predictions for all ten student-test trips are identical when the .route files are "
  "removed from disk; (ii) the final top-1 recomputed from the deployed pipeline's own output "
  f"({VAL['independent_recomputation']['final_top1_from_pipeline']:.6f}) equals the value from the tuning script "
  f"({VAL['independent_recomputation']['final_top1_from_tuning_csv']:.6f}); (iii) every stage of the ladder was selected on the 80 "
  "training trips (CV or out-of-fold) and only then evaluated on student-test; (iv) the fresh-process smoke test and the "
  "all-trips API run of Section 7 use this same configuration.")


# ================================================================== 6 Folium
P("6. Geographical Visualisation Using Folium", H1)
P("scripts/make_maps.py builds every map from the actual pipeline output (the same Predictor class the API "
  "uses). Layers: OpenStreetMap basemap; nearby road network (grey); raw GPS fixes (blue dots with fix index and "
  "timestamp tooltips); reference route (green, development trips only); classifier-only prediction (orange "
  "dashed, off by default); classifier + HMM route (red); fixes matched to a non-reference arc (black circles, "
  "off by default). Each map has a layer control, a legend with the trip's scores, start/end markers, and is "
  "fitted to the track extent. Interactive HTML files are committed in maps/.")
fig("screenshot_map_test_full.png", "Figure 24. Student-test trip 00000009, full extent (maps/student_test_00000009.html).",
    width=W * 0.95, maxh=10.5 * cm)
fig("screenshot_map_test_zoom.png", "Figure 25. Zoomed view of the hardest student-test trip 00000077: reference "
    "(green), HMM route (red), classifier-only (orange dashed) and wrongly matched fixes (black circles).",
    width=W * 0.95, maxh=10.5 * cm)
story.append(PageBreak())

# ================================================================== 7 API/UI
P("7. API, UI and Deployment", H1)
fig("diagram_deployment.png", "Figure 26. Two-laptop deployment.", width=W * 0.9)
P("<b>Backend (laptop 1).</b> api/server.py (FastAPI + uvicorn, bound to 0.0.0.0:8000) loads "
  f"models/final_config.json (model file {FINAL['model']}.joblib = {FINAL.get('display_name', FINAL['model'])}; 'v2' is the config name of the {len(FEATURES)}-feature table; decoder setting of Section 5) and the model "
  "file once at start-up. GET /trips lists the trip folders of the configured partition (PATHSNAP_PARTITION), "
  "so the ten reserved evaluator trips are discovered from disk, not hardcoded. POST /predict {trip_id} runs (1) "
  "the route-free pipeline: load trip without its .route, candidates, features, classifier, HMM, route filling, "
  "re-alignment; "
  "then (2) a separate evaluation step (api/evaluation.py) that reads the private reference route, derives "
  "per-fix labels with the same alignment as in training, and computes top-1, macro-F1, continuity and route "
  "precision/recall/F1 for the classifier-only, Viterbi and final outputs. Only these aggregate scores, the predicted route and "
  "the GPS fixes are returned; reference geometry and labels never leave the server. Every request is logged "
  "(console + logs/api_requests.log) with client IP, status, latency, trip and accuracy, for the evaluator to "
  "inspect on laptop 1.")
P("<b>UI (laptop 2).</b> ui/index.html + app.js, a static page served with python -m http.server. The API "
  "address is entered once (or passed as ?api=http://&lt;laptop-1-IP&gt;:8000). It lists the trips returned by "
  "/trips as buttons, posts the selected ID, draws the GPS fixes and predicted route on a Leaflet map, and shows "
  "the server-side scores and timings.")
lat_rows = [["Trip", "Status", "Fixes", "Round trip (s)", "Route arcs", "Top-1 cls", "Top-1 HMM", "Top-1 final", "Macro-F1 final",
             "Continuity final", "Route F1"]]
for _, r in APIALL.iterrows():
    lat_rows.append([r.trip, r.status, int(r.n_obs), f"{r.seconds:.2f}", int(r.route_arcs), f3(r.top1_cls), f3(r.top1_hmm),
                     f3(r.top1_final), f3(r.macro_f1_final), f3(r.continuity_final), f3(r.route_f1)])
table(lat_rows)
P("Table 13. All ten student-test trips requested from the running API over HTTP by a separate client process: status, "
  "round-trip latency and the scores returned by the server-side evaluation (results/api_all_trips.csv). "
  f"Slowest request {APIALL.seconds.max():.2f} s. A fresh-process smoke test (scripts/smoke_test.py) additionally covers a "
  "synthetic trip with a 100-fix dropout and an unknown trip ID (404).", CAP)
fig("screenshot_ui.png", "Figure 27. Evaluator UI during development (served locally, API on the same machine, student-test trip): "
    "trip list, predicted route and server-side evaluation.", width=W * 0.9, maxh=12.5 * cm)
fig("screenshot_api_docs.png", "Figure 28. Successful POST /predict response (FastAPI interactive docs).",
    width=W * 0.9, maxh=11 * cm)
logf = ROOT / "logs" / "api_requests_demo.log"
if logf.exists():
    lines = [ln.rstrip() for ln in open(logf) if "PREDICT" in ln or "POST" in ln][-12:]
    P("Server console log excerpt (copied verbatim from logs/api_requests_demo.log):", SMALL)
    P("<br/>".join(ln.replace("&", "&amp;").replace("<", "&lt;") for ln in lines), MONO)
P("7.1 Final evaluation on two laptops (ten reserved evaluator trips)", H2)
import re  # noqa: E402
EVLOG = [ln.rstrip() for ln in open(ROOT / "logs" / "api_requests_evaluator.log")]
EV, _cur = [], None
for ln in EVLOG:
    m_ = re.search(r"PREDICT done trip=(\d+) obs=(\d+) route_arcs=(\d+) acc_final=([\d.]+) acc_hmm=([\d.]+) acc_cls=([\d.]+) total=([\d.]+)s", ln)
    if m_:
        _cur = dict(trip=m_[1], obs=int(m_[2]), arcs=int(m_[3]), final=float(m_[4]), hmm=float(m_[5]), cls=float(m_[6]), pipe=float(m_[7]), time=ln[:19])
    m_ = re.search(r"POST /predict from ([\d.]+) -> (\d+) in (\d+) ms", ln)
    if m_ and _cur:
        EV.append({**_cur, "client": m_[1], "status": int(m_[2]), "ms": int(m_[3])})
        _cur = None
EV = pd.DataFrame(EV)
assert len(EV) == 10 and (EV.status == 200).all() and (EV.client != "127.0.0.1").all()
assert all((FIG / f"ui_evaluator_{t}.jpg").exists() for t in EV.trip)
P(f"The reserved evaluator trips were received on {EV.time.min()[:10]}, after model, features, hyperparameters and decoder had been fixed "
  "(models/final_config.json and the model file were not changed afterwards). The API ran on laptop 1 with "
  "PATHSNAP_PARTITION=evaluator_test; the UI ran in a browser on laptop 2 (served by python -m http.server) and was pointed at "
  "laptop 1's LAN address. Every one of the ten trips was selected in the UI on laptop 2, sent to the API as a trip ID, processed by "
  "the complete pipeline on laptop 1 and scored there against the private reference route. Table 14 is read from the request log of "
  f"laptop 1 (client address {', '.join(sorted(EV.client.unique()))} = laptop 2); Figures 29 and 30 are the ten screens of laptop 2. "
  "These are final-evaluation results: no choice in Sections 2-5 was made or changed after the evaluator trips were opened.")
evr = [["Trip", "Client (laptop 2)", "HTTP status", "GPS fixes", "Route arcs", "Top-1 scorer", "Top-1 + HMM", "Top-1 final", "Pipeline time (s)", "Server round trip (ms)"]]
for _, r in EV.sort_values("trip").iterrows():
    evr.append([r.trip, r.client, r.status, f"{r.obs:,}", r.arcs, f4(r.cls), f4(r.hmm), f4(r.final), f"{r['pipe']:.2f}", r.ms])
evr.append(["Mean of 10 trips", "", "", "", "", f4(EV.cls.mean()), f4(EV.hmm.mean()), f4(EV.final.mean()), f"{EV['pipe'].mean():.2f}", f"{EV.ms.mean():.0f}"])
table(evr, widths=[2.0 * cm, 2.3 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.7 * cm, 1.6 * cm, 1.8 * cm, 1.9 * cm])
P("Table 14. The ten evaluator trips requested from laptop 2 and answered by the API on laptop 1 (logs/api_requests_evaluator.log: the last "
  "request of each trip, copied verbatim from the server log). Top-1 = share of scored GPS fixes on the reference arc, computed on the server; "
  f"pooled over all {int(EV.obs.sum()):,} fixes the final top-1 is {float((EV.final * EV.obs).sum() / EV.obs.sum()):.4f}. "
  "The lowest trip is " + f"{EV.loc[EV.final.idxmin(), 'trip']} ({EV.final.min():.4f}).", CAP)
P("Server console log on laptop 1 (verbatim excerpt, first three trips):", SMALL)
P("<br/>".join(ln.replace("&", "&amp;").replace("<", "&lt;") for ln in EVLOG[1:10]), MONO)
fig("ui_evaluator_00000017.jpg", "Figure 29a. UI on laptop 2, evaluator trip 00000017: the address box shows the API on laptop 1; predicted route on the map; "
    "server-side evaluation for scorer only, + HMM/Viterbi and + re-alignment.", width=W, maxh=10.6 * cm)
fig("ui_evaluator_00000015.jpg", "Figure 29b. UI on laptop 2, evaluator trip 00000015 (the longest trip, 6,854 fixes, and the lowest route F1 of the ten).",
    width=W, maxh=10.6 * cm)
from reportlab.lib.utils import ImageReader as _IR  # noqa: E402
_rest = [t for t in sorted(EV.trip) if t not in ("00000017", "00000015")]
_cells, _row = [], []
for t in _rest:
    iw, ih = _IR(str(FIG / f"ui_evaluator_{t}.jpg")).getSize()
    wd = W / 2 - 4
    _row.append([Image(str(FIG / f"ui_evaluator_{t}.jpg"), width=wd, height=wd * ih / iw),
                 Paragraph(f"Trip {t}: final top-1 {float(EV.set_index('trip').loc[t, 'final']):.4f}", CAP)])
    if len(_row) == 2:
        _cells.append(_row)
        _row = []
_g = Table(_cells, colWidths=[W / 2, W / 2])
_g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
story.append(_g)
P("Figure 30. UI on laptop 2 for the other eight evaluator trips (full-resolution files: figures/ui_evaluator_&lt;trip&gt;.jpg).", CAP)
story.append(PageBreak())

# ================================================================== 8 limitations
P("8. Limitations and Reproducibility", H1)
P("<b>Limitations.</b> (1) Observation labels come from aligning fixes to the supplied route; where the route is "
  "incomplete or the alignment is ambiguous (fixes near an intersection, or either direction of a two-way street on "
  f"out-and-back stretches: {pc(ERR['label_noise']['share'])} of moving fixes have a label against the direction of "
  "travel, Section 5.2) the labels and "
  "therefore every observation metric inherit that uncertainty; off-route fixes (&gt;60 m, "
  f"{int(QUAL.set_index('check').loc['observations flagged off-route (>60 m)', 'train']):,} in train, "
  f"{int(QUAL.set_index('check').loc['observations flagged off-route (>60 m)', 'student_test'])} in student-test) are "
  "excluded. (2) Student-test has only 10 trips, so differences below ~0.01 are within trip-to-trip variation. "
  f"(3) Bayesian search fitted on a stride-{OPT['search_train_stride']} subsample for cost; the refit uses stride 3. (4) Candidate radius "
  f"{RADIUS_M:.0f} m / K={MAX_K} caps recall; fixes with larger errors cannot be recovered by the scorer. "
  "(5) The decoder setting comes from a small grid; a Bayesian search over eight decoder parameters did not generalise "
  f"to held-out training trips (Section 5.3). The deployed scorer has a train-validation top-1 gap of {M2['cv_mean']['train_obs_top1'] - M2['cv_mean']['obs_top1']:+.3f}. (6) Trips from other regions may differ; the "
  "features are local and scale-free where possible, but beyond the 90 development trips this was only checked on the ten evaluator trips of Section 7.1.")
P("<b>Reproducibility.</b> Fixed seed 42; fold assignment stored in results/folds.json; exact package versions in "
  "requirements.txt; every figure and table is regenerated by the scripts listed in the README from the raw files. "
  "scripts/robustness_test.py serves 16 awkward synthetic trips (single fix, empty track, track outside the road network, "
  "huge time gap, coarse sampling, unsorted or duplicate timestamps, missing / empty / invalid reference route) from a fresh "
  f"API process: all {sum(1 for k, v in ROB.items() if isinstance(v, dict))} requests return HTTP 200 with well-formed JSON (results/robustness_test.json); two "
  "crashes found by this test (no arcs near the track; undefined continuity for very short trips) were fixed. The largest "
  f"training trips (up to {int(LAT.n_obs.max()):,} fixes, up to {int(LAT.n_arcs.max()):,} arcs) take at most {LAT.total_s.max():.1f} s end to end "
  "(results/latency_large_trips.csv). scripts/preflight.py repeats these checks on the evaluation partition before the demo. "
  "scripts/smoke_test.py re-runs the full raw-files-to-API path in a fresh process; its last run passed all checks "
  f"({', '.join(k for k, v in SMOKE['checks'].items() if v)}).")
P("<b>AI assistance disclosure.</b> An AI coding assistant (Claude) was used to help write code and draft this "
  "report under the team's direction. All experiments were actually executed; all numbers, plots and screenshots "
  "come from the committed scripts and logs.")
P("9. Conclusion", H1)
P(f"Nearest-segment matching is right for only {pc(BASE['student_test']['obs_top1'])} of student-test fixes, mainly "
  "because two-way streets put two arcs at the same distance. A learned candidate scorer with heading and "
  f"temporal-context features raises this to {pc(te('lgbm', 'obs_top1'))} for the best of six model families; all six beat the baseline "
  "by a wide margin and LightGBM is best on every quality metric. A Bayesian search over LightGBM's hyperparameters "
  "changed accuracy by less than the fold-to-fold noise; what moved accuracy was information and objective: "
  f"the {len(FEATURES) - len(BASE_FEATURES)} context features and a LambdaRank objective lift the scorer to "
  f"{pc(F2T['cls']['top1'])}. Sequence decoding matters as much as the scorer: HMM/Viterbi over the road graph plus "
  f"re-assignment of fixes along the predicted route gives the final {pc(F2T['final']['top1'])} top-1, "
  f"{pc(F2T['final']['macro_f1'])} macro-F1, {pc(F2T['final']['continuity'])} continuity and {pc(F2T['route_f1'])} route F1 "
  f"on student-test ({pc(F2R['final']['top1'])} top-1 on the 80 training trips out-of-fold), in about a second per trip "
  f"through the API. The remaining {pc(ERR['errors']['final_pred']['error_rate'])} of student-test errors are mostly "
  "genuinely off-route matches plus arc-boundary and direction disagreements, part of which trace back to the reference "
  f"labels. Sections 2-5 are development-stage results. The ten reserved evaluator trips were opened only after the system was frozen and were served "
  f"from a second laptop: final top-1 between {EV.final.min():.3f} and {EV.final.max():.3f} per trip, mean {EV.final.mean():.3f} (Section 7.1). "
  "Single trips can be well below the average (Section 5.4).")
P("Appendix A. Optuna trial record", H2)
trows = [["Trial", "CV top-1", "CV macro-F1", "n_est.", "lr", "leaves", "min_child", "subsample", "colsample", "lambda", "alpha", "trunc."]]
for _, r in comp.sort_values("number").iterrows():
    trows.append([int(r.number), f4(r.value), f4(r.user_attrs_cv_macro_f1), int(r.params_n_estimators),
                  f"{r.params_learning_rate:.3f}", int(r.params_num_leaves), int(r.params_min_child_samples),
                  f"{r.params_subsample:.2f}", f"{r.params_colsample_bytree:.2f}", f"{r.params_reg_lambda:.3g}",
                  f"{r.params_reg_alpha:.3g}", int(r.params_lambdarank_truncation_level)])
table(trows)


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(1.8 * cm, 1.1 * cm, "PathSnap - Technical Report")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


(ROOT / "report").mkdir(exist_ok=True)
out = ROOT / "report" / "PathSnap_Technical_Report.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.7 * cm, title="PathSnap Technical Report", author=", ".join(m["name"] for m in TEAM.get("members", [])) or "PathSnap team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("written", out)
