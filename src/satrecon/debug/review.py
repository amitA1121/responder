"""Interactive 2D review page.

A single self-contained HTML file: the source image as a backdrop, footprints
as SVG overlays, layer toggles and a click-to-inspect panel.  This is the
Phase 1 verification surface - confirm the footprints here before any 3D
geometry is generated from them.

The page is deliberately dependency-free so it opens straight from disk.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from ..logging_setup import get_logger
from ..model import Scene

log = get_logger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #14161a; --panel: #1d2026; --line: #2f343d;
    --text: #e8eaed; --muted: #9aa2ad;
    --high: #57d971; --medium: #ffb454; --low: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex;
           flex-wrap: wrap; gap: 12px; align-items: center; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .warn { color: var(--medium); font-size: 12px; }
  .wrap { display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; align-items: flex-start; }
  .stage { position: relative; flex: 1 1 560px; min-width: 320px; background: #000;
           border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .stage img { display: block; width: 100%; height: auto; }
  .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  polygon { cursor: pointer; stroke-width: 2; vector-effect: non-scaling-stroke; }
  polygon.sel { stroke: #fff; stroke-width: 3; }
  .side { flex: 0 1 320px; min-width: 280px; display: flex; flex-direction: column; gap: 12px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
             color: var(--muted); margin: 0 0 8px; }
  label.row { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  td { padding: 3px 0; vertical-align: top; }
  td.k { color: var(--muted); width: 48%; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px;
           font-size: 11px; font-weight: 600; color: #11131a; }
  .HIGH { background: var(--high); } .MEDIUM { background: var(--medium); } .LOW { background: var(--low); }
  ul.notes { margin: 0; padding-left: 16px; color: var(--muted); font-size: 12px; }
  ul.notes li { margin-bottom: 4px; }
  .list { max-height: 260px; overflow: auto; }
  .list div { padding: 4px 6px; border-radius: 4px; cursor: pointer; display: flex;
              justify-content: space-between; gap: 8px; }
  .list div:hover { background: #262a32; }
  .list div.sel { background: #2f3642; }
  code { color: var(--muted); font-size: 11px; word-break: break-all; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__ &mdash; Phase 1 footprint review</h1>
  <span class="warn">Approximate detections from one image. Not measurements. Height, floors and roof type are not estimated in Phase 1.</span>
</header>

<div class="wrap">
  <div class="stage" id="stage">
    <img id="base" src="__IMAGE_SRC__" alt="source image">
    <svg id="ov" viewBox="0 0 __W__ __H__" preserveAspectRatio="none"></svg>
  </div>

  <div class="side">
    <div class="card">
      <h2>Layers</h2>
      <label class="row"><input type="checkbox" id="t-image" checked> Satellite image</label>
      <label class="row"><input type="checkbox" id="t-fill" checked> Footprint fill</label>
      <label class="row"><input type="checkbox" id="t-outline" checked> Footprint outline</label>
      <label class="row"><input type="checkbox" id="t-labels" checked> Building IDs</label>
      <label class="row"><input type="checkbox" id="t-contour"> Raw detected contour</label>
    </div>

    <div class="card">
      <h2>Scene</h2>
      <table id="scene-table"></table>
    </div>

    <div class="card">
      <h2>Selected building</h2>
      <div id="detail"><span style="color:var(--muted)">Click a footprint to inspect it.</span></div>
    </div>

    <div class="card">
      <h2>Candidates (<span id="count"></span>)</h2>
      <div class="list" id="list"></div>
    </div>

    <div class="card">
      <h2>Notes</h2>
      <ul class="notes" id="notes"></ul>
    </div>
  </div>
</div>

<script>
const SCENE = __SCENE_JSON__;
const COLORS = { HIGH: "#57d971", MEDIUM: "#ffb454", LOW: "#ff6b6b" };
const ov = document.getElementById("ov");
const NS = "http://www.w3.org/2000/svg";
let selected = null;

function el(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
}

function draw() {
  ov.innerHTML = "";
  const showContour = document.getElementById("t-contour").checked;
  const showFill = document.getElementById("t-fill").checked;
  const showOutline = document.getElementById("t-outline").checked;
  const showLabels = document.getElementById("t-labels").checked;

  for (const b of SCENE.buildings) {
    const band = b.confidence.band;
    if (showContour && b.contour && b.contour.length) {
      ov.appendChild(el("polygon", {
        points: b.contour.map(p => p.join(",")).join(" "),
        fill: "none", stroke: "#7fd4ff", "stroke-width": 1, "stroke-dasharray": "4 3",
        "pointer-events": "none", opacity: 0.85
      }));
    }
    const poly = el("polygon", {
      points: b.footprint.map(p => p.join(",")).join(" "),
      fill: showFill ? COLORS[band] : "none",
      "fill-opacity": showFill ? 0.22 : 0,
      stroke: showOutline ? COLORS[band] : "none",
    });
    poly.dataset.id = b.id;
    if (b.id === selected) poly.classList.add("sel");
    poly.addEventListener("click", () => select(b.id));
    ov.appendChild(poly);

    if (showLabels) {
      const [x, y, w, h] = b.bounding_box;
      const text = el("text", {
        x: x + w / 2, y: y + h / 2, fill: "#fff", "font-size": 16,
        "text-anchor": "middle", "pointer-events": "none",
        stroke: "#000", "stroke-width": 3, "paint-order": "stroke"
      });
      text.textContent = b.id;
      ov.appendChild(text);
    }
  }
}

function fmt(value, unit) {
  if (value === null || value === undefined) return "<span style='color:var(--muted)'>not estimated</span>";
  return unit ? `${value} ${unit}` : value;
}

function select(id) {
  selected = id;
  const b = SCENE.buildings.find(x => x.id === id);
  const scaleKnown = SCENE.scene.scale.meters_per_pixel != null;
  const dims = scaleKnown
    ? `${b.length_m} m &times; ${b.width_m} m <span style="color:var(--muted)">(estimated)</span>`
    : `${b.length_px} px &times; ${b.width_px} px <span style="color:var(--muted)">(scale unknown)</span>`;
  const c = b.confidence;
  document.getElementById("detail").innerHTML = `
    <table>
      <tr><td class="k">ID</td><td><b>${b.id}</b></td></tr>
      <tr><td class="k">Footprint</td><td>${dims}</td></tr>
      <tr><td class="k">Area</td><td>${scaleKnown ? b.area_m2 + " m&sup2;" : b.area_px + " px&sup2;"}</td></tr>
      <tr><td class="k">Orientation</td><td>${b.orientation_deg}&deg; (image frame)</td></tr>
      <tr><td class="k">Vertices</td><td>${b.footprint.length}</td></tr>
      <tr><td class="k">Height</td><td>${fmt(b.estimated_height_m, "m")}</td></tr>
      <tr><td class="k">Floors</td><td>${fmt(b.estimated_floors)}</td></tr>
      <tr><td class="k">Roof</td><td>${b.roof_type}</td></tr>
      <tr><td class="k">Footprint conf.</td><td>${c.footprint}</td></tr>
      <tr><td class="k">Scale conf.</td><td>${fmt(c.scale)}</td></tr>
      <tr><td class="k">Overall</td><td><span class="badge ${c.band}">${c.band}</span> ${c.overall}</td></tr>
    </table>
    <details style="margin-top:8px"><summary style="cursor:pointer;color:var(--muted);font-size:12px">Detector evidence</summary>
    <code>${JSON.stringify(b.evidence)}</code></details>`;
  draw();
  renderList();
}

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (const b of SCENE.buildings) {
    const row = document.createElement("div");
    if (b.id === selected) row.classList.add("sel");
    row.innerHTML = `<span>${b.id}</span>
      <span><span class="badge ${b.confidence.band}">${b.confidence.band}</span></span>`;
    row.addEventListener("click", () => select(b.id));
    list.appendChild(row);
  }
  document.getElementById("count").textContent = SCENE.buildings.length;
}

function renderScene() {
  const s = SCENE.scene, p = SCENE.provenance;
  const scale = s.scale.meters_per_pixel == null
    ? "<b style='color:var(--medium)'>UNKNOWN</b>"
    : `${s.scale.meters_per_pixel} m/px`;
  document.getElementById("scene-table").innerHTML = `
    <tr><td class="k">Name</td><td>${s.name}</td></tr>
    <tr><td class="k">Image</td><td>${s.image.width} &times; ${s.image.height} px</td></tr>
    <tr><td class="k">Scale</td><td>${scale}</td></tr>
    <tr><td class="k">Scale source</td><td>${s.scale.source}</td></tr>
    <tr><td class="k">Shadow dir.</td><td>${s.shadowDirectionDeg}&deg; (image frame)</td></tr>
    <tr><td class="k">North offset</td><td>${s.northOffsetDeg == null ? "unknown" : s.northOffsetDeg + "&deg;"}</td></tr>
    <tr><td class="k">Detector</td><td>${p.detector}</td></tr>
    <tr><td class="k">Result hash</td><td><code>${p.resultHash.slice(0, 16)}</code></td></tr>`;
  document.getElementById("notes").innerHTML =
    s.notes.map(n => `<li>${n}</li>`).join("");
}

for (const id of ["t-fill", "t-outline", "t-labels", "t-contour"]) {
  document.getElementById(id).addEventListener("change", draw);
}
document.getElementById("t-image").addEventListener("change", e => {
  document.getElementById("base").style.visibility = e.target.checked ? "visible" : "hidden";
});

renderScene();
renderList();
draw();
</script>
</body>
</html>
"""


def write(scene: Scene, out_path: Path, image_path: Path, embed_image: bool = False) -> Path:
    """Render the review page. Copies the image beside it unless embedding."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if embed_image:
        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
        payload = base64.b64encode(image_path.read_bytes()).decode()
        image_src = f"data:image/{mime};base64,{payload}"
    else:
        local = out_path.parent / f"source{image_path.suffix.lower()}"
        if image_path.resolve() != local.resolve():
            shutil.copyfile(image_path, local)
        image_src = local.name

    html = (
        _TEMPLATE
        .replace("__TITLE__", scene.meta.name)
        .replace("__IMAGE_SRC__", image_src)
        .replace("__W__", str(scene.meta.image_width))
        .replace("__H__", str(scene.meta.image_height))
        .replace("__SCENE_JSON__", json.dumps(scene.to_json()))
    )
    out_path.write_text(html)
    log.info("wrote review page %s", out_path)
    return out_path
