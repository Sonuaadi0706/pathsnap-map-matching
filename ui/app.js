// PathSnap evaluator UI: lists trips from the API, sends the selected trip id,
// renders the returned predicted route and the server-side evaluation scores.
// The UI never receives reference routes or labels -- only aggregate scores.
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let API = params.get("api") || safeGet("pathsnap_api") || "http://127.0.0.1:8000";
let selected = null;
$("api").value = API;

function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function status(msg) { $("status").textContent = msg; }

const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
let layers = L.layerGroup().addTo(map);

async function connect() {
  API = $("api").value.trim().replace(/\/$/, "");
  safeSet("pathsnap_api", API);
  status("Connecting to " + API + " ...");
  try {
    const h = await (await fetch(API + "/health")).json();
    const r = await (await fetch(API + "/trips")).json();
    const box = $("trips");
    box.innerHTML = "";
    r.trips.forEach((t) => {
      const b = document.createElement("div");
      b.className = "trip"; b.textContent = t;
      b.onclick = () => { document.querySelectorAll(".trip").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); selected = t; $("run").disabled = false; };
      box.appendChild(b);
    });
    status(`Connected · model ${h.model} · ${r.count} trips in "${r.partition}"`);
  } catch (e) { status("Connection failed: " + e); }
}

const pct = (x) => (x === undefined || x === null || Number.isNaN(x)) ? "n/a" : (100 * x).toFixed(2) + "%";

async function run() {
  if (!selected) return;
  $("run").disabled = true;
  status(`Sent trip ${selected} to API, waiting for prediction ...`);
  const t0 = performance.now();
  try {
    const resp = await fetch(API + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: selected }) });
    if (!resp.ok) throw new Error(resp.status + " " + (await resp.text()));
    const d = await resp.json();
    draw(d);
    show(d, (performance.now() - t0) / 1000);
    status(`Trip ${d.trip_id} matched in ${((performance.now() - t0) / 1000).toFixed(2)} s (round trip).`);
  } catch (e) { status("Request failed: " + e); }
  $("run").disabled = false;
}

function draw(d) {
  layers.clearLayers();
  const gps = d.gps_points;
  gps.forEach((p) => L.circleMarker(p, { radius: 3, color: "#444", weight: 1, fillColor: "#888", fillOpacity: 0.8 }).addTo(layers));
  const lines = d.predicted_route_lines;
  L.polyline(lines, { color: "#1f4fd1", weight: 5, opacity: 0.8 }).addTo(layers);
  layers.eachLayer((l) => { if (l instanceof L.CircleMarker) l.bringToFront(); });
  if (gps.length) {
    L.circleMarker(gps[0], { radius: 7, color: "#137a3a", fillOpacity: 1 }).bindTooltip("start").addTo(layers);
    L.circleMarker(gps[gps.length - 1], { radius: 7, color: "#b00020", fillOpacity: 1 }).bindTooltip("end").addTo(layers);
  }
  map.invalidateSize();
  const all = gps.concat(lines.flat());
  if (all.length) map.fitBounds(L.latLngBounds(all), { padding: [20, 20] });
}

function row(k, v) { return `<tr><td>${k}</td><td class="num">${v}</td></tr>`; }

function show(d, rt) {
  const e = d.evaluation || {};
  if (e.available) {
    $("kpis").innerHTML = `<div class="kpi"><div class="big">${pct(e.observation_top1_accuracy_hmm)}</div><div class="lbl">Observation top-1 accuracy (HMM)</div></div>
      <div class="kpi"><div class="big">${pct(e.route_f1_hmm)}</div><div class="lbl">Route F1 (length-weighted)</div></div>`;
    $("metrics").innerHTML = `<tr><th>Metric</th><th style="text-align:right">Classifier only</th><th style="text-align:right">Classifier + HMM</th></tr>` +
      `<tr><td>Observation top-1 accuracy</td><td class="num">${pct(e.observation_top1_accuracy_classifier_only)}</td><td class="num">${pct(e.observation_top1_accuracy_hmm)}</td></tr>` +
      `<tr><td>Observation macro-F1 (per arc)</td><td class="num">${pct(e.observation_macro_f1_classifier_only)}</td><td class="num">${pct(e.observation_macro_f1_hmm)}</td></tr>` +
      `<tr><td>Route continuity</td><td class="num">${pct(e.route_continuity_classifier_only)}</td><td class="num">${pct(e.route_continuity_hmm)}</td></tr>` +
      `<tr><td>Route F1</td><td class="num">${pct(e.route_f1_classifier_only)}</td><td class="num">${pct(e.route_f1_hmm)}</td></tr>` +
      `<tr><td>Route precision / recall</td><td class="num">–</td><td class="num">${pct(e.route_precision_hmm)} / ${pct(e.route_recall_hmm)}</td></tr>` +
      `<tr><td>Scored observations</td><td class="num" colspan="2">${e.n_scored_observations} of ${e.n_observations}</td></tr>`;
  } else {
    $("kpis").innerHTML = "";
    $("metrics").innerHTML = row("Evaluation", e.reason || "not available");
  }
  const t = d.timings_s || {};
  $("details").innerHTML = row("Trip", d.trip_id) + row("Model", d.model) + row("GPS observations", d.n_observations) +
    row("Matched observations", d.n_matched_observations) + row("Predicted route arcs", d.n_route_arcs) +
    row("HMM segments", d.n_hmm_segments) + row("Server pipeline time", (t.total_s ?? 0).toFixed(2) + " s") +
    row("Server evaluation time", (t.evaluation_s ?? 0).toFixed(2) + " s") + row("Round trip", rt.toFixed(2) + " s");
  const lite = { ...d, predicted_route_lines: `[${d.predicted_route_lines.length} segments]`, gps_points: `[${d.gps_points.length} points]`, predicted_route_arc_ids: d.predicted_route_arc_ids.slice(0, 20).concat(["..."]) };
  $("raw").textContent = JSON.stringify(lite, null, 2);
}

$("connect").onclick = connect;
$("run").onclick = run;
connect();
