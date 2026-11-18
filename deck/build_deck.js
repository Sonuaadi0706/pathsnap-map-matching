// PathSnap pitch deck (black & white). Numbers come from deck_data.json, which
// scripts/deck_data.py exports from results/. Run: node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
// optional: writes the monochrome palette + name into the deck theme (helper from the pptx tooling);
// the deck uses explicit hex colours, so it renders identically without it.
let applyTheme = null;
try { if (process.env.PPTX_SKILL) ({ applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js")); } catch (e) { applyTheme = null; }

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const FIG = path.join(__dirname, "..", "figures");
const OUT = path.join(__dirname, "PathSnap_Pitch_Deck.pptx");

const THEME = {
  name: "PathSnap Mono", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: { dk1: "000000", lt1: "FFFFFF", dk2: "1F1F1F", lt2: "F2F2F2", accent1: "111111", accent2: "595959",
    accent3: "8C8C8C", accent4: "BFBFBF", accent5: "D9D9D9", accent6: "404040", hlink: "000000", folHlink: "595959" },
};
const HEX = { ink: "111111", mid: "595959", light: "A6A6A6", pale: "F2F2F2", grid: "E3E3E3" };
const f3 = (x) => Number(x).toFixed(3);
const f4 = (x) => Number(x).toFixed(4);
const pct = (x) => (100 * x).toFixed(1) + "%";

function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// fit image into box (x,y,w,h) keeping aspect, centred
function fitImage(slide, file, x, y, w, h, name) {
  const p = path.join(FIG, file);
  const s = pngSize(p);
  let iw = w, ih = (w * s.h) / s.w;
  if (ih > h) { ih = h; iw = (h * s.w) / s.h; }
  slide.addImage({ path: p, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name, altText: name });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "PathSnap pitch deck";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = { text1: "000000", background1: "FFFFFF", accent2: "595959", accent3: "8C8C8C", accent4: "BFBFBF" };

pres.defineSlideMaster({
  title: "DARK", background: { color: C.text1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.3, fontSize: 48, bold: true,
    color: C.background1, valign: "bottom" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.75, w: 11.7, h: 1.4, fontSize: 20,
    color: C.accent4, valign: "top" }, text: "" } }],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true,
      color: C.text1, valign: "middle" }, text: "" } },
    { text: { text: "PathSnap  |  map-matching & route reconstruction", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent3 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8C8C8C" },
});

function stat(slide, x, y, w, big, label, dark = false, name = "stat", size = 40) {
  slide.addText(big, { x, y, w, h: 0.9, fontSize: size, bold: true, color: dark ? C.background1 : C.text1, margin: 0,
    isTextBox: true, objectName: name + "_value" });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.75, fontSize: 13, color: dark ? C.accent4 : C.accent2, margin: 0,
    valign: "top", isTextBox: true, objectName: name + "_label" });
}
function bullets(slide, items, x, y, w, h, size = 16, name = "bullets") {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize: size, color: C.text1, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: name });
}
const chartBase = (title) => ({
  showTitle: true, title, titleFontSize: 13, titleColor: "111111", titleFontFace: "+mn-lt",
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelColor: "404040", valAxisLabelColor: "595959", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  valGridLine: { color: "E3E3E3", size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 9, dataLabelColor: "111111", dataLabelFormatCode: "0.000",
});

const models = D.models;
const sel = D.hmm.rows[D.hmm.sel];
const lg = models.find((m) => m.key === "lgbm");

// 1 ---------------------------------------------------------------- title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("PathSnap", { placeholder: "title" });
s.addText("Matching noisy GPS to the road network with a learned scorer + HMM/Viterbi decoding" +
  (D.team.team_name ? "\n" + D.team.team_name : ""), { placeholder: "body" });
s.addNotes("PathSnap reconstructs the road route a vehicle actually drove from noisy GPS fixes. Two stages: a classifier scores candidate road arcs per fix, an HMM decoded with Viterbi turns those scores into a connected route. Everything shown is from our runs; numbers are on the 10 student-test trips unless stated.");

// 2 ---------------------------------------------------------------- problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("The nearest road is wrong almost half the time", { placeholder: "title" });
stat(s, 0.6, 1.55, 3.6, pct(D.baseline.test_top1), "of student-test fixes lie on their nearest arc (geometric baseline)", false, "s1");
stat(s, 0.6, 3.35, 3.6, pct(D.data.twoway), "of arcs (median trip) are one direction of a two-way street, at identical distance", false, "s2");
stat(s, 0.6, 5.15, 3.6, pct(D.baseline.test_cont), "route continuity of nearest-arc matching: the path jumps", false, "s3");
fitImage(s, "bw_screenshot_map_test_zoom.png", 4.6, 1.4, 8.1, 5.45, "map_zoom");
s.addNotes(`Nearest-segment matching gets ${pct(D.baseline.test_top1)} of fixes right. The main reason is two-way streets: both directions are separate arcs at the same distance, so only motion (heading) can tell them apart. Parallel roads and urban multipath add the rest. Right: our Folium map of the hardest test trip.`);

// 3 ---------------------------------------------------------------- data
pres.addSection({ title: "Data" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
s.addText("90 trips, worldwide, every trip with its own road graph", { placeholder: "title" });
stat(s, 0.6, 1.5, 2.9, `${D.data.train_trips} / ${D.data.test_trips} / 10`, "train / student-test / evaluator trips", false, "d1", 32);
stat(s, 3.6, 1.5, 2.9, (D.data.train_fixes / 1000).toFixed(0) + "k", "training GPS fixes (" + (D.data.test_fixes / 1000).toFixed(1) + "k test)", false, "d2", 32);
stat(s, 6.6, 1.5, 2.9, (D.data.arcs_min / 1000).toFixed(1) + "k-" + (D.data.arcs_max / 1000).toFixed(0) + "k", "directed arcs per trip graph", false, "d3", 32);
stat(s, 9.6, 1.5, 3.1, pct(D.data.frac_dt1), "of steps at 1 Hz (median trip), but gaps up to " + (D.data.max_gap / 3600).toFixed(0) + " h", false, "d4", 32);
fitImage(s, "eda_trajectories.png", 0.6, 3.3, 12.1, 3.6, "trajectories");
s.addNotes(`No missing values or duplicate timestamps. Irregularity comes from time gaps, degenerate arcs and reference routes that stop before the track ends: ${D.data.offroute_train.toLocaleString()} training fixes are more than 60 m from the aligned route and are excluded from scoring. Each trip is projected to local metres around its own centroid.`);

// 4 ---------------------------------------------------------------- pipeline
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Pipeline: candidates, learned scores, sequence decoding", { placeholder: "title" });
fitImage(s, "diagram_pipeline.png", 0.6, 1.35, 12.1, 3.2, "pipeline");
bullets(s, [
  "Labels: the .route has no per-fix alignment, so fixes are aligned to the route sequence with a monotone dynamic programme",
  "Candidates: arcs within 80 m, up to 24 per fix (12 in round 1); the reference arc is among them for every scored test fix",
  "Prediction never reads .route; evaluation reads it afterwards, on the server only",
], 0.6, 4.7, 12.1, 2.1, 16, "pipeline_notes");
s.addNotes("Why ranking instead of multi-class: arc IDs are trip-specific, so we score each candidate arc of a fix with a binary classifier and renormalise within the fix. Labels come from aligning the fix sequence to the route sequence with a monotone DP, which handles loops and skipped short arcs.");

// 5 ---------------------------------------------------------------- features
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Relative distance and heading decide the match", { placeholder: "title" });
const fi = D.features.slice().reverse();
s.addChart(pres.charts.BAR, [{ name: "gain share", labels: fi.map((f) => f[0]), values: fi.map((f) => f[1]) }], {
  x: 0.6, y: 1.4, w: 7.0, h: 5.4, barDir: "bar", chartColors: ["111111"], showLegend: false,
  ...chartBase("Round-1 model: share of split gain (top 8 of 28 features)"), dataLabelFormatCode: "0.0%", valAxisLabelFormatCode: "0%",
});
bullets(s, [
  "Distance: absolute, rank, gap to the best candidate",
  "Heading vs arc bearing, over ±2 and ±8 fixes, weighted by speed",
  "Context: distance of neighbouring fixes to the same arc",
  "Topology: touches neighbours' nearest arc, node degrees, two-way",
  "Round 2 adds 18 more context features; all are route-free",
], 8.0, 1.6, 4.7, 5.0, 15, "feature_groups");
s.addNotes("Feature importance from the tuned LightGBM. Relative distance (how much farther than the best candidate) generalises across noise levels better than raw distance; the windowed heading resolves the direction of two-way streets.");

// 6 ---------------------------------------------------------------- model comparison
pres.addSection({ title: "Models" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Six model families, identical data: LightGBM wins", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "test top-1", labels: models.map((m) => m.label), values: models.map((m) => m.test_top1) },
  { name: "test macro-F1", labels: models.map((m) => m.label), values: models.map((m) => m.test_f1) },
], { x: 0.6, y: 1.35, w: 7.4, h: 5.5, barDir: "col", barGrouping: "clustered", chartColors: ["111111", "A6A6A6"],
  showLegend: true, legendPos: "b", legendFontSize: 11, valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00",
  ...chartBase("Student-test, classifier only (observation level)"), dataLabelFontSize: 8 });
const rows = [[{ text: "Model", options: { bold: true } }, { text: "CV macro-F1", options: { bold: true } },
  { text: "Log loss", options: { bold: true } }, { text: "Score time", options: { bold: true } }]];
models.forEach((m) => rows.push([m.label, `${f3(m.cv_f1)} ± ${f3(m.cv_f1_std)}`, f3(m.test_ll), m.infer_s.toFixed(2) + " s"]));
s.addTable(rows, { x: 8.3, y: 1.5, w: 4.4, colW: [1.35, 1.25, 0.85, 0.95], fontSize: 11, color: "111111",
  border: { type: "solid", pt: 0.75, color: "BFBFBF" }, fill: { color: "FFFFFF" }, rowH: 0.42, objectName: "model_table" });
s.addText("CV: 5 trip-grouped folds on 80 training trips. Log loss: student-test, observation level.", {
  x: 8.3, y: 4.75, w: 4.4, h: 0.8, fontSize: 11, color: C.accent2, isTextBox: true, objectName: "table_note" });
s.addNotes(`Same candidates, labels and folds for every model. LightGBM has the best CV macro-F1 (${f3(lg.cv_f1)}), test top-1 (${f3(lg.test_top1)}) and lowest log loss (${f3(lg.test_ll)}), which matters because the HMM consumes the probabilities. KNN is slowest; Gaussian NB has poorly calibrated probabilities because its features are correlated. Likely question: why binary ranking and not multi-class? Arc IDs are trip-specific, so each candidate arc of a fix is scored and the scores are renormalised within the fix.`);

// 7 ---------------------------------------------------------------- generalisation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Trees fit training trips; boosting generalises best", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "training folds", labels: models.map((m) => m.label), values: models.map((m) => m.train_top1) },
  { name: "validation folds", labels: models.map((m) => m.label), values: models.map((m) => m.cv_top1) },
], { x: 0.6, y: 1.35, w: 6.6, h: 5.5, barDir: "col", barGrouping: "clustered", chartColors: ["BFBFBF", "111111"],
  showLegend: true, legendPos: "b", legendFontSize: 11, valAxisMinVal: 0.8, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00",
  ...chartBase("Observation top-1, train vs CV validation"), dataLabelFontSize: 8 });
fitImage(s, "hist_lgbm.png", 7.5, 1.45, 5.2, 3.1, "lgbm_curve");
bullets(s, [
  "Random Forest: largest train-validation gap (memorises trips)",
  "LightGBM: validation loss flattens after ~120 rounds but does not rise",
  "LogReg and Naive Bayes: no gap, but under-fit",
], 7.5, 4.75, 5.2, 2.0, 14, "fit_notes");
s.addNotes("Over/under-fitting evidence from the same CV folds. Epoch-wise curves are shown only for models that actually produce them (LightGBM rounds, MLP epochs).");

// 8 ---------------------------------------------------------------- bayesian
pres.addSection({ title: "Tuning & decoding" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Tuning & decoding" });
s.addText("Bayesian search: flat optimum, lighter model", { placeholder: "title" });
fitImage(s, "optuna_history.png", 0.6, 1.4, 7.4, 5.3, "optuna_history");
const tu = D.optuna.tv.lgbm, tt = D.optuna.tv.lgbm_tuned;
stat(s, 8.4, 1.5, 4.3, String(D.optuna.n_trials), "Optuna TPE trials, 5-fold trip-grouped CV each, " + D.optuna.wall_min.toFixed(0) + " min", false, "o1");
stat(s, 8.4, 3.3, 4.3, `${f4(tu.cv_top1)} → ${f4(tt.cv_top1)}`, "CV top-1, untuned → tuned (same folds); test is a tie", false, "o2", 34);
stat(s, 8.4, 5.1, 4.3, `${f4(tu.test_top1)} → ${f4(tt.test_top1)}`, "student-test top-1, untuned → tuned; tuned overfits less and scores 2x faster", false, "o3", 34);
s.addNotes(`Search space: trees, learning rate, leaves, min child samples, bagging fractions, L1/L2. Objective: mean CV top-1 on the 80 training trips only. Every trial lies between ${f3(D.optuna.min)} and ${f3(D.optuna.max)}, so the features, not the hyperparameters, limit accuracy. The gain is smaller than the fold-to-fold variation; we report it as small.`);

// 8b --------------------------------------------------------------- round 2
const R2 = D.r2, V = R2.variants;
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Tuning & decoding" });
s.addText("Round 2: better features and a ranking objective", { placeholder: "title" });
const vl = ["R1 scorer\n28 features", "24 candidates\nsame features", "+ 18 context\nfeatures", "+ LambdaRank\nobjective"];
s.addChart(pres.charts.BAR, [
  { name: "CV top-1", labels: vl, values: [V[0].cv_top1, V[1].cv_top1, V[2].cv_top1, V[3].cv_top1] },
  { name: "CV macro-F1", labels: vl, values: [V[0].cv_macro_f1, V[1].cv_macro_f1, V[2].cv_macro_f1, V[3].cv_macro_f1] },
], { x: 0.6, y: 1.35, w: 7.4, h: 5.5, barDir: "col", barGrouping: "clustered", chartColors: ["111111", "A6A6A6"],
  showLegend: true, legendPos: "b", legendFontSize: 11, valAxisMinVal: 0.84, valAxisMaxVal: 0.94, valAxisLabelFormatCode: "0.00",
  ...chartBase("Same 5 trip-grouped folds, 80 training trips"), dataLabelFontSize: 9 });
stat(s, 8.4, 1.5, 4.3, `${f3(R2.r1_scorer.test_top1)} → ${f3(R2.scorer.test_top1)}`, "scorer top-1 on student-test, round 1 → round 2", false, "v1", 34);
stat(s, 8.4, 3.3, 4.3, `${f3(R2.r1_scorer.test_f1)} → ${f3(R2.scorer.test_f1)}`, "scorer macro-F1 on student-test", false, "v2", 34);
stat(s, 8.4, 5.1, 4.3, `${f3(R2.r1_scorer.test_ll)} → ${f3(R2.scorer.test_ll)}`, "observation log loss on student-test (lower is better)", false, "v3", 34);
s.addNotes(`Round 1 showed hyperparameters were not the limit, so round 2 changed what the model sees and optimises, with labels, folds and metrics fixed. More candidates alone do not help; the 18 context features do; LambdaRank (rank the candidates of one fix) helps again. A second Optuna search of ${R2.optuna2.n_trials} trials (top five re-scored at full fidelity) again tied the existing parameters within fold noise, so they were kept. Likely question: is it overfitting? Train-fold top-1 is ${f3(R2.scorer.train_top1)} vs ${f3(R2.scorer.cv_top1)} on validation folds, a larger gap than round 1, but validation and student-test both improved.`);

// 9 ---------------------------------------------------------------- HMM
const FT = R2.test, FTR = R2.train, VALD = R2.val;
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Tuning & decoding" });
s.addText("Decoding turns point guesses into a drivable route", { placeholder: "title" });
stat(s, 0.6, 1.5, 4.6, `${f3(FT.cls.top1)} → ${f3(FT.hmm.top1)} → ${f3(FT.final.top1)}`, "top-1: scorer → + HMM/Viterbi → + re-alignment to the predicted route", false, "h1", 25);
stat(s, 0.6, 3.3, 4.4, `${f3(FT.cls.continuity)} → ${f3(FT.final.continuity)}`, "route continuity, scorer → final", false, "h2", 30);
stat(s, 0.6, 5.1, 4.4, f3(FT.route_f1), "route F1 (length-weighted) of the decoded route", false, "h3", 30);
const pt = R2.pertrip;
s.addChart(pres.charts.BAR, [
  { name: "scorer only", labels: pt.map((r) => r.trip.slice(-3)), values: pt.map((r) => r.cls) },
  { name: "+ HMM/Viterbi", labels: pt.map((r) => r.trip.slice(-3)), values: pt.map((r) => r.hmm) },
  { name: "+ re-alignment (final)", labels: pt.map((r) => r.trip.slice(-3)), values: pt.map((r) => r.final) },
], { x: 5.3, y: 1.35, w: 7.4, h: 5.5, barDir: "col", barGrouping: "clustered", chartColors: ["D9D9D9", "8C8C8C", "111111"],
  showLegend: true, legendPos: "b", legendFontSize: 11, valAxisMinVal: 0.7, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00",
  ...chartBase("Per student-test trip (last 3 digits of trip ID): observation top-1"), showValue: false });
s.addNotes(`States are the candidate arcs of each fix; emissions are its probabilities; transitions compare driving distance through the directed graph with straight-line distance between fixes, plus speed, backward-motion and U-turn penalties. Chains break at long gaps, fixes without candidates, or infeasible transitions. Final step: Viterbi fixes the route, then each fix is assigned to an arc of that predicted route in travel order (no reference used). We also ran two Bayesian searches over eight decoder parameters (${R2.decoder.n_trials} trials each); the best setting gained on its tuning trips but dropped from ${f3(R2.heldout.round1_decoder.trip_mean_top1)} to ${f3(R2.heldout.tuned_decoder.trip_mean_top1)} on 16 held-out training trips, so it was not adopted and the simple grid setting is kept. Final top-1 on the 80 training trips (out-of-fold): ${f3(FTR.final.top1)}. Likely question: does the last step use the answer? No: predictions are identical when the .route files are deleted.`);

// 10 --------------------------------------------------------------- map
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Tuning & decoding" });
s.addText("Folium: prediction against reference on the real map", { placeholder: "title" });
s.addText("Interactive maps are colour-coded (maps/*.html); shown here in greyscale for print.", { x: 0.6, y: 6.62, w: 12.1, h: 0.35, fontSize: 11, color: C.accent2, isTextBox: true, objectName: "map_note" });
fitImage(s, "bw_screenshot_map_test_full.png", 0.6, 1.3, 12.1, 5.25, "map_full");
s.addNotes("Generated by scripts/make_maps.py from the same predictor the API uses. Layers: road network, raw GPS, reference route (development trips only), classifier-only, HMM route and wrongly matched fixes, with layer control and legend.");

// 11 --------------------------------------------------------------- demo
pres.addSection({ title: "Demo & results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo & results" });
s.addText("Two-laptop demo: the API predicts, then scores privately", { placeholder: "title" });
fitImage(s, "diagram_deployment.png", 0.6, 1.35, 6.0, 2.4, "deployment");
bullets(s, [
  "GET /trips lists the trip folders on laptop 1: nothing hardcoded",
  "POST /predict runs features, model and HMM without the reference",
  "Evaluation runs afterwards; only scores reach the UI",
  "Every request is logged on laptop 1",
], 0.6, 4.0, 6.0, 2.9, 15, "demo_notes");
fitImage(s, "bw_screenshot_ui.png", 6.9, 1.35, 5.8, 5.5, "ui");
s.addNotes(`End-to-end latency measured over HTTP in a fresh server: ${D.smoke["00000009"].seconds} s for trip 00000009 (${D.smoke["00000009"].n_observations} fixes) and ${D.smoke["00000077"].seconds} s for 00000077 (${D.smoke["00000077"].n_observations} fixes). The screenshot is from a local run; the two-laptop screenshots are captured at the live evaluation.`);

// 12 --------------------------------------------------------------- result
const EF = D.final.errors.final_pred, LN = D.final.label_noise, CI = VALD.student_test.final_top1.ci95;
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Demo & results" });
s.addText("Final result on the 10 student-test trips", { x: 0.8, y: 0.7, w: 11.7, h: 0.9, fontSize: 30, bold: true,
  color: C.background1, isTextBox: true, objectName: "result_title" });
stat(s, 0.8, 2.2, 2.8, f3(FT.final.top1), "observation top-1 (nearest-segment baseline " + f3(D.baseline.test_top1) + ")", true, "r1");
stat(s, 3.8, 2.2, 2.8, f3(FT.final.macro_f1), "observation macro-F1 (per arc, per trip)", true, "r2");
stat(s, 6.8, 2.2, 2.8, f3(FT.route_f1), "route F1, length-weighted", true, "r3");
stat(s, 9.8, 2.2, 2.8, f3(FT.final.continuity), "route continuity", true, "r4");
s.addText(`Top-1 95% interval over trips: ${f3(CI[0])} to ${f3(CI[1])}. On the 80 training trips (out-of-fold): ${f3(FTR.final.top1)}.`, {
  x: 0.8, y: 4.6, w: 11.7, h: 0.6, fontSize: 18, color: C.accent4, isTextBox: true, objectName: "result_note" });
s.addText(`All ${D.final.api_ok} trips served by the API in ${D.final.api_max_s.toFixed(1)} s or less. Residual error ${pct(EF.error_rate)}.`, {
  x: 0.8, y: 5.3, w: 11.7, h: 0.6, fontSize: 18, color: C.accent4, isTextBox: true, objectName: "result_note2" });
s.addNotes(`Development-stage numbers on the 10 student-test trips; the evaluator's reserved trips were never available to us, so their score is produced live by the API. The training-trip figure is lower because a few training trips are much harder (weakest ${VALD.train_oof.worst_trip_final.trip}: ${f3(VALD.train_oof.worst_trip_final.top1)}), so expect variation between evaluator trips. Likely question: how trustworthy are the labels? ${pct(LN.share)} of moving fixes have a reference label pointing against the direction of travel (the alignment is geometry-only), so part of the remaining error is label noise.`);

// 13 --------------------------------------------------------------- limitations
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo & results" });
s.addText("Limitations and reproducibility", { placeholder: "title" });
s.addText("Limitations", { x: 0.6, y: 1.5, w: 5.8, h: 0.5, fontSize: 22, bold: true, color: C.text1, isTextBox: true, objectName: "lim_h" });
bullets(s, [
  "Labels come from aligning fixes to the supplied route; " + pct(LN.share) + " of moving fixes are labelled against the direction of travel",
  "Only 10 test trips: differences under ~0.01 are within noise",
  "Hard trips remain: weakest training trip is far below the average",
  "Hyperparameter search barely moved accuracy; features and objective did",
], 0.6, 2.2, 5.9, 4.5, 17, "lim_list");
s.addText("Reproducibility", { x: 6.9, y: 1.5, w: 5.8, h: 0.5, fontSize: 22, bold: true, color: C.text1, isTextBox: true, objectName: "rep_h" });
bullets(s, [
  "Seed 42, fold file committed, pinned requirements",
  "One script per stage; every figure regenerated from logs",
  "Predictions identical with the .route files deleted",
  "Smoke test: raw files → API → checks, all passing",
], 6.9, 2.2, 5.8, 4.5, 17, "rep_list");
s.addNotes("AI assistance (Claude) was used for coding and drafting under the team's direction; all numbers come from executed scripts. Likely question: why not tune more? Forty Optuna trials all landed within 0.004 of each other, so more search would not help; better labels and features would.");

// 14 --------------------------------------------------------------- closing
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Demo & results" });
s.addText("Learned scores choose the arc. Road topology makes it a route.", { placeholder: "title" });
s.addText(`${f3(D.baseline.test_top1)} → ${f3(FT.cls.top1)} → ${f3(FT.hmm.top1)} → ${f3(FT.final.top1)} top-1: nearest segment, scorer, Viterbi, final\ngithub.com/Sonuaadi0706/pathsnap-map-matching`, { placeholder: "body" });
s.addNotes("Close: each stage is measured on the same trips with the same labels. Code, logs, maps and the report are in the public repository; the live demo runs any of the evaluator's ten trips through the API.");

pres.writeFile({ fileName: OUT }).then(async () => {
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("written", OUT, applyTheme ? "(theme applied)" : "(default theme)");
});
