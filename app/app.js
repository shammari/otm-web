// OTM Web page: files, settings, run, results. The analysis itself runs in worker.js.

import { TissueView, parseView } from "./viewer.js";
import { CorrectionsEditor, TOOLS } from "./editor.js";

// ============================================================================================
// small helpers
// ============================================================================================
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, props = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c !== null && c !== undefined) e.append(c);
  return e;
};
const num = (v) => (v === "" || v === null || v === undefined ? null : Number(v));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtInt = (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString("en-US"));
const stemOf = (fileName) => fileName.replace(/\.[^.]+$/, "");
const extOf = (fileName) => (fileName.match(/\.[^.]+$/) || [""])[0].toLowerCase();

function fmtParam(v) {
  if (v === null || v === undefined || v === "") return "";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return Number(v).toExponential(3).replace(/\.?0+e/, "e");
  return String(Number(v.toPrecision(6)));
}

function toast(text, warn = false, ms = 4500) {
  if (warn && typeof logEvent === "function") logEvent(`notice: ${text}`);
  const t = el("div", { class: "toast" + (warn ? " warn" : ""), role: "status", text });
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

function download(blob, name) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

// ============================================================================================
// the analysis worker
// ============================================================================================
class Engine {
  constructor() {
    this.ready = false;
    this.failed = null;
    this.pending = new Map();
    this.nextId = 1;
    this.onStatus = () => {};
    this.onProgress = () => {};
    this.onDownload = () => {};
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = ({ data }) => this._message(data);
    this.worker.onerror = (e) => {
      const msg = e.message || "the analysis engine stopped";
      this.failed = msg;
      for (const { reject } of this.pending.values()) reject(new Error(msg));
      this.pending.clear();
      this.onStatus(msg, "failed");
    };
    this.started = this.call("init").then((info) => { this.ready = true; return info; });
  }
  _message(data) {
    if (data.type === "status") return this.onStatus(data.text, "loading");
    if (data.type === "download") return this.onDownload(data);
    if (data.type === "progress") return this.onProgress(data);
    const p = this.pending.get(data.id);
    if (!p) return;
    this.pending.delete(data.id);
    if (data.ok) p.resolve(data.result);
    else {
      const err = new Error(data.error.message);
      err.details = data.error.details;
      p.reject(err);
    }
  }
  call(cmd, args = {}, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, cmd, args }, transfer);
    });
  }
}

// ============================================================================================
// state
// ============================================================================================
const state = {
  info: null,
  files: [],            // {name, mat: File, otm: File|null, check, status, error, width, height, roi, corrections,
                        //  otmEdits: edits in the .otm, edits: corrections made here (null: use the .otm)}
  orphanOtm: new Map(), // stem (lower case) -> File, .otm added before its .mat
  results: new Map(),   // name -> result of otm_web.run
  order: [],            // result names in run order
  selected: null,
  running: false,
  stopRequested: false,
  extras: {},           // settings from a loaded JSON the form does not show (mesh, flux, retouch ...)
  figureCache: new Map(),
  views: new Map(),     // sample -> parsed tissue view (the last few)
  packs: new Map(),     // sample run in an extra worker -> its files (until the main worker has them)
  unpacks: [],          // moves to the main worker still going on
  viewRange: undefined, // PO2 colour range of the tissue views (undefined: the one in the settings)
  viewColormap: null,
  log: [],              // errors and warnings of this visit, for the diagnostic report
};

const engine = new Engine();
// first visit (or after a Pyodide upgrade) Python is downloaded: show how far, instead of a
// status that seems frozen; later visits load it from the browser's cache in a few seconds
const TOTAL_MB = 45;
const pyDownload = { bytes: 0, fromCache: 0, active: true };
engine.onDownload = (d) => {
  pyDownload.bytes = d.downloaded;
  pyDownload.fromCache = d.fromCache;
  if (pyDownload.active && d.downloaded > 5e5) {
    const mb = d.downloaded / 1e6;
    setEngine(`Downloading Python, first time only: ${mb.toFixed(0)} of about ${TOTAL_MB} MB…`, "loading");
  }
};
engine.onStatus = (text, kind) => {
  if (text.startsWith("Loading OTM") || kind !== "loading") pyDownload.active = false;
  if (pyDownload.active && pyDownload.bytes > 5e5) return;      // keep showing the download progress
  setEngine(text, kind);
};
engine.onProgress = (p) => updateProgress(p.name, p.f, p.msg);

function setEngine(text, kind) {
  const box = $("#engine");
  box.className = "engine " + (kind || "");
  $("#engine-text").textContent = text;
}

// ============================================================================================
// 1 FILES
// ============================================================================================
function uniqueName(stem) {
  let name = stem.replace(/[\\/:*?"<>|]/g, "_");
  const taken = new Set(state.files.map((f) => f.name.toLowerCase()));
  if (!taken.has(name.toLowerCase())) return name;
  for (let i = 2; ; i++) if (!taken.has(`${name}_${i}`.toLowerCase())) return `${name}_${i}`;
}

function addFiles(list) {
  let ignored = 0;
  const added = [];
  for (const file of list) {
    const ext = extOf(file.name);
    if (ext === ".mat") {
      const entry = { name: uniqueName(stemOf(file.name)), stem: stemOf(file.name), mat: file, otm: null,
                      check: null, status: "waiting", error: null, given: state.defaultSize ? state.defaultSize.given : null,
                      value: state.defaultSize ? state.defaultSize.value : "", roi: null, corrections: null,
                      otmEdits: null, edits: null };
      const orphan = state.orphanOtm.get(entry.stem.toLowerCase());
      if (orphan) { entry.otm = orphan; state.orphanOtm.delete(entry.stem.toLowerCase()); }
      state.files.push(entry);
      added.push(entry);
    } else if (ext === ".otm") {
      const target = state.files.find((f) => f.stem.toLowerCase() === stemOf(file.name).toLowerCase());
      if (target) { target.otm = file; target.corrections = null; target.otmEdits = null; if (!added.includes(target)) added.push(target); }
      else state.orphanOtm.set(stemOf(file.name).toLowerCase(), file);
    } else ignored++;
  }
  if (ignored) toast(`${ignored} file(s) left out: only .mat and .otm files can be analysed.`, true);
  if (state.orphanOtm.size) toast(`Session file(s) waiting for a .mat with the same name: ${[...state.orphanOtm.keys()].join(", ")}`);
  renderFiles();
  for (const f of added) checkFile(f);
}

async function putFile(entry, eng = engine) {
  await eng.call("put", { name: entry.name, ext: ".mat", buffer: await entry.mat.arrayBuffer() });
  if (entry.otm) await eng.call("put", { name: entry.name, ext: ".otm", buffer: await entry.otm.arrayBuffer() });
}

async function checkFile(entry) {
  entry.status = "checking";
  renderFiles();
  try {
    await engine.started;
    await putFile(entry);
    const res = await engine.call("inspect", { name: entry.name, opts: checkOptions() });
    if (res.ok) {
      entry.check = res;
      entry.status = "ok";
      entry.error = null;
      if (entry.otm) {
        const c = await engine.call("corrections", { name: entry.name });
        entry.corrections = c.ok ? c.n_edits : null;
        entry.otmEdits = c.ok ? c.edits : null;
        if (!c.ok) entry.error = `session file not used: ${c.error}`;
      }
    } else {
      entry.status = "bad";
      entry.error = res.error;
    }
  } catch (e) {
    entry.status = "bad";
    entry.error = e.message;
  } finally {
    if (engine.ready) engine.call("drop", { name: entry.name }).catch(() => {});
  }
  renderFiles();
}

// what the data check and the editor depend on
function checkOptions() {
  return { tissue: radio("tissue"), use_fibre_types: $("#use-types").checked, data_check: radio("dcheck") };
}

// the tissue type changes what the data check looks for: check every readable file again
let recheckTimer = 0;
function recheckFiles() {
  clearTimeout(recheckTimer);
  recheckTimer = setTimeout(() => { for (const f of state.files) if (f.status === "ok") checkFile(f); }, 300);
}

// corrections a run of this file will use: made here, else from its .otm
function editsOf(entry) { return entry.edits !== null ? entry.edits : (entry.otmEdits || []); }

const DC_ICON = { ok: "✓", warning: "⚠", error: "✗", fatal: "✗" };

// the data-check cell of the file table
function checkCell(f) {
  const c = f.check, q = c && c.check;
  const bits = [];
  if (!q) bits.push(el("span", { class: "ok-text", text: "✓ readable" }));
  else {
    const n = (sev) => q.issues.filter((i) => i.severity === sev).length;
    const policy = radio("dcheck");
    let text, cls;
    if (q.status === "ok") { text = "✓ passed"; cls = "ok-text"; }
    else if (q.status === "warning") { text = `⚠ ${n("warning")} warning${n("warning") > 1 ? "s" : ""}`; cls = "note-text"; }
    else if (q.status === "fatal") { text = "✗ cannot be analysed"; cls = "warn-text"; }
    else {
      const what = policy === "skip" ? "file will be skipped" : (policy === "keep" && q.can_keep) ? "flagged items kept" : "flagged items will be removed";
      text = `✗ ${n("error")} error${n("error") > 1 ? "s" : ""}: ${what}`;
      cls = "warn-text";
    }
    bits.push(el("span", { class: cls, title: q.headline + (q.issues.length ? "\n" + q.issues.map((i) => `${i.code} ${i.title}: ${i.message}`).join("\n") : ""), text }));
    if (q.issues.length) bits.push(el("div", { class: "hint small dc-codes", text: q.issues.map((i) => `${i.code} ${i.title}`).join(" · ") }));
  }
  const nEd = editsOf(f).length;
  if (nEd) bits.push(el("div", { class: "hint small", text: `${nEd} correction${nEd > 1 ? "s" : ""}${f.edits !== null ? " made here" : " from the .otm"}` }));
  if (f.error) bits.push(el("div", { class: "warn-text small", text: f.error }));
  if (c && c.notes && c.notes.length) bits.push(el("div", { class: "hint small", text: c.notes.join("; ") }));
  return el("span", {}, ...bits);
}

function fileSize(entry) {
  const own = num(entry.value);
  if (entry.given && own > 0) return entry.given === "width" ? { width_um: own, height_um: null } : { width_um: null, height_um: own };
  return null;
}

// the size all runnable files share (saved as width_um / height_um in the settings JSON), else null
function commonSize() {
  const files = runnable();
  if (!files.length || files.some((f) => !fileSize(f))) return null;
  const first = files[0];
  return files.every((f) => f.given === first.given && num(f.value) === num(first.value)) ? fileSize(first) : null;
}

function applySizeToAll(src) {
  const others = state.files.filter((f) => f !== src && f.status !== "bad");
  for (const f of others) { f.given = src.given; f.value = src.value; showSize(f); }
  updateRunState();
  toast(`${src.given === "width" ? "Width" : "Height"} ${src.value} µm given to ${others.length} other file(s); `
    + "the other side is calculated for each image.");
}

// width / height of the image (as LengthScale: the other side follows the image's pixel shape)
function aspectOf(f) { return f.check ? f.check.image_cols / f.check.image_rows : null; }
function sizeOf(f) {
  const s = fileSize(f), a = aspectOf(f);
  if (!s || !a) return null;
  return s.width_um ? [s.width_um, s.width_um / a] : [s.height_um * a, s.height_um];
}
const fmtUm = (v) => String(Number(v.toFixed(1)));

function showSize(f) {
  const inputs = f._inputs;
  if (!inputs) return;
  const own = f.given && num(f.value) > 0;
  const wh = sizeOf(f);
  for (const key of ["width", "height"]) {
    const inp = inputs[key];
    if (own && f.given === key) { inp.classList.remove("derived"); inp.title = "Typed for this file"; continue; }
    const v = wh ? fmtUm(key === "width" ? wh[0] : wh[1]) : "";
    if (own) {
      inp.value = v;
      inp.placeholder = "–";
      inp.classList.add("derived");
      inp.title = "Calculated from the image shape; type here to give this side instead";
    } else {
      inp.value = "";
      inp.placeholder = "–";
      inp.classList.remove("derived");
      inp.title = "Type the width or the height of this image in µm";
    }
  }
  if (f._apply) f._apply.hidden = !(own && state.files.filter((x) => x.status !== "bad").length > 1);
}

function refreshSizes() {
  for (const f of state.files) showSize(f);
}

function renderFiles() {
  const tbody = $("#file-table tbody");
  tbody.textContent = "";
  for (const f of state.files) {
    const c = f.check;
    const types = c && c.fibre_types
      ? Object.entries(c.fibre_types).filter(([k, n]) => n && k !== "unknown").map(([k, n]) => `${k} ${n}`).join(" · ") || "none"
      : (c ? "not stored" : "");
    const inputs = {};
    for (const key of ["width", "height"]) {
      inputs[key] = el("input", {
        type: "number", min: "0", step: "any", placeholder: "–", "aria-label": `${key} of ${f.name} in µm`,
        value: f.given === key ? f.value : "",
        onblur: () => showSize(f),
        oninput: (e) => {
          if (e.target.value) { f.given = key; f.value = e.target.value; }
          else if (f.given === key) { f.given = null; f.value = ""; }
          else return;                      // a calculated side was emptied: keep the given one

          showSize(f);
          updateRunState();
        },
      });
    }
    f._inputs = inputs;
    f._apply = el("button", { class: "ghost small apply-all", type: "button", hidden: true,
      title: "Give every file this side; the other side is calculated for each image",
      onclick: () => applySizeToAll(f) }, "Same for all");
    let check;
    if (f.status === "checking" || f.status === "waiting") check = el("span", { class: "hint small", text: f.status === "waiting" ? "waiting…" : "checking…" });
    else if (f.status === "ok") check = checkCell(f);
    else check = el("span", { class: "warn-text", text: `✗ ${f.error || "cannot be read"} (left out)` });
    const correct = f.status === "ok"
      ? el("button", { class: "ghost small", type: "button", title: "See the data with the flagged items, and correct capillaries, fibres and fibre types",
          onclick: () => openEditor(f) }, "Correct…")
      : null;
    tbody.append(el("tr", { "data-name": f.name },
      el("td", {}, el("b", { text: f.name }), f.otm ? el("span", { class: "hint small", text: " + .otm" }) : null),
      el("td", { class: "num", text: c ? `${c.image_cols} × ${c.image_rows}` : "" }),
      el("td", { class: "num", text: c ? fmtInt(c.n_capillaries) : "" }),
      el("td", { class: "num", text: c ? fmtInt(c.n_fibres) : "" }),
      el("td", { class: "nowrap", text: types }),
      el("td", { class: "num" }, inputs.width),
      el("td", { class: "num" }, inputs.height),
      el("td", {}, f._apply),
      el("td", { class: "dc-cell" }, check, correct),
      el("td", {}, el("button", { class: "remove", type: "button", title: `Remove ${f.name}`, "aria-label": `Remove ${f.name}`,
        onclick: () => removeFile(f) }, "×")),
    ));
  }
  $("#file-table-wrap").hidden = $("#file-actions").hidden = state.files.length === 0;
  refreshSizes();
  updateRunState();
}

function removeFile(f) {
  if (state.running) return toast("Wait until the run has finished.", true);
  state.files = state.files.filter((x) => x !== f);
  renderFiles();
}

function setupFiles() {
  const drop = $("#drop");
  const input = $("#file-input");
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  // dropping a file anywhere else must not navigate away from the page
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); if (!drop.contains(e.target)) addFiles(e.dataTransfer.files); });
  $("#clear-files").addEventListener("click", () => {
    if (state.running) return toast("Wait until the run has finished.", true);
    state.files = [];
    state.orphanOtm.clear();
    renderFiles();
  });
  $("#save-dims").addEventListener("click", () => {
    const rows = state.files.filter((f) => fileSize(f));
    if (!rows.length) return toast("No sizes to save yet: type a width or height in the table.", true);
    const q = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
    const lines = ["file,width_um,height_um,roi_x_min_um,roi_x_max_um,roi_y_min_um,roi_y_max_um"];
    for (const f of rows) {
      const sz = fileSize(f);
      lines.push([q(f.mat.name), sz.width_um ?? "", sz.height_um ?? "", ...(f.roi || ["", "", "", ""])].join(","));
    }
    download(new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv" }), "OTM_dimensions.csv");
  });
  $("#dims-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      await engine.started;
      const res = await engine.call("parseDims", { text: await file.text() });
      if (!res.ok) return toast(`Dimensions CSV not read: ${res.error}`, true);
      let matched = 0;
      for (const f of state.files) {
        const ch = res.table[f.stem.toLowerCase()] || res.table[f.name.toLowerCase()];
        if (!ch) continue;
        matched++;
        f.given = ch.width_um ? "width" : ch.height_um ? "height" : null;
        f.value = String(ch.width_um || ch.height_um || "");
        f.roi = ch.roi_um || null;
      }
      const rows = Object.keys(res.table).length;
      toast(`Sizes set for ${matched} of ${state.files.length} file(s) from ${rows} row(s).`, matched < state.files.length);
      renderFiles();
    } catch (err) { toast(`Dimensions CSV not read: ${err.message}`, true); }
  });
}

// ============================================================================================
// 2 SETTINGS
// ============================================================================================
const radio = (name) => $(`input[name="${name}"]:checked`).value;
const setRadio = (name, value) => { const r = $(`input[name="${name}"][value="${value}"]`); if (r) r.checked = true; };

function paramFields() { return state.info ? state.info.parameters.fields : []; }

function presetValues(tissue) { return state.info.parameters.defaults[tissue]; }

function readParams() {
  const out = {};
  for (const f of paramFields()) out[f.name] = num($(`#p-${f.name}`).value);
  return out;
}

function writeParams(values) {
  for (const f of paramFields()) $(`#p-${f.name}`).value = fmtParam(values[f.name]);
  markParams();
}

function samePreset(values, tissue) {
  const d = presetValues(tissue);
  return paramFields().every((f) => values[f.name] !== null && Math.abs(values[f.name] - d[f.name]) <= 1e-9 * Math.abs(d[f.name]));
}

function markParams() {
  const v = readParams();
  const preset = $("#preset").value;
  const ref = preset === "custom" ? null : presetValues(preset);
  for (const f of paramFields()) {
    const input = $(`#p-${f.name}`);
    input.classList.toggle("changed", !!ref && v[f.name] !== null && Math.abs(v[f.name] - ref[f.name]) > 1e-9 * Math.abs(ref[f.name]));
    input.classList.toggle("bad", v[f.name] === null || !(v[f.name] >= 0));
  }
}

// display only: O2 -> O₂ and subscripted symbols (R_cap -> R<sub>cap</sub>, M0 -> M<sub>0</sub>);
// the parameter names in saved settings and .dat files do not change
const o2 = (text) => text.replace(/\bPO2\b/g, "PO₂").replace(/\bO2\b/g, "O₂");
function symbolNodes(sym) {
  let m = sym.match(/^([^_]+)_(.+)$/) || sym.match(/^([A-Za-z])(\d.*)$/);
  return m ? [m[1], el("sub", { text: m[2] })] : [sym];
}

const LEVEL_NAMES = { resting: "Resting", low: "Low", moderate: "Moderate", high: "High" };
const fmtNum = (v) => fmtParam(Number(v.toPrecision(4)));

// what the exercise level does, from otm_core's own table (ModelSwitches), as the desktop shows it
function updateExercise() {
  if (!state.info) return;
  const tissue = radio("tissue");
  const level = $("#exercise").value || "resting";
  const sw = state.info.exercise[tissue][level];
  const box = $("#exercise-summary");
  box.textContent = "";
  box.append("Michaelis–Menten uptake: ", el("b", { text: sw.michaelis_menten ? "on" : "off" }),
    " · Myoglobin facilitation: ", el("b", { text: sw.myoglobin ? "on" : "off" }),
    " · O₂ demand × ", el("b", { text: fmtNum(sw.demand) }),
    " · capillary permeability × ", el("b", { text: fmtNum(sw.permeability) }));
  const note = $("#m0-note");
  const m0 = num($("#p-M0") && $("#p-M0").value);
  const preset = presetValues(tissue).M0;
  const changed = m0 !== null && !Number.isNaN(m0) && Math.abs(m0 - preset) > 1e-9 * preset;
  note.hidden = !(level !== "resting" && changed);
  if (!note.hidden) {
    const eff = Number((m0 * sw.demand).toPrecision(3)).toExponential().replace(/\.?0+e/, "e");
    note.textContent = `M₀ is multiplied by ${fmtNum(sw.demand)} for ${LEVEL_NAMES[level]} exercise, so this run uses `
      + `M₀ = ${eff} ml O₂ / ml / s. If ${fmtParam(m0)} is already an exercising value, enter the resting value instead.`;
  }
}

// Λ relative to rest before a run: exercise × M0 / (K·Pcap) against the tissue's resting defaults
// (type-I coefficients; the run computes it per fibre, with the section's fibre sizes and types)
function updateDemand() {
  if (!state.info) return;
  $("#dm-options").classList.toggle("off", !$("#dm-enabled").checked);
  for (const e of $$("#dm-options input")) e.disabled = !$("#dm-enabled").checked;
  const tissue = radio("tissue");
  const level = $("#exercise").value || "resting";
  const sw = state.info.exercise[tissue][level];
  const p = readParams(), d = presetValues(tissue);
  const K = (q) => q.alpha * q.D + q.c_Mb * q.D_Mb / q.P50_Mb;
  const ratio = (sw.demand * p.M0 / (K(p) * p.Pcap)) / (d.M0 / (K(d) * d.Pcap));
  const box = $("#dm-estimate");
  box.classList.remove("switch");
  if (!Number.isFinite(ratio)) { box.textContent = ""; return; }
  const cutoff = num($("#dm-cutoff").value) || 1.5;
  const non = ratio > cutoff;
  box.textContent = "";
  box.append("Demand about ", el("b", { text: `${ratio.toFixed(2)} × rest` }), `: ${non ? "non-resting" : "resting"}`);
  if (!$("#dm-enabled").checked) { box.append(" (Λ only reported)."); return; }
  const parts = [];
  if (non && level === "resting" && $("#dm-uptake").checked) {
    parts.push("Michaelis–Menten uptake and myoglobin will be switched on");
    box.classList.add("switch");
  }
  if (non && $("#dm-cap").checked) parts.push(`automatic flux-line cap × ${Math.sqrt(1 / ratio).toFixed(2)}`);
  box.append(parts.length ? ` · ${parts.join(" · ")}.` : ".");
}

function buildParamGrid() {
  const box = $("#params");
  box.textContent = "";
  for (const f of paramFields()) {
    box.append(el("label", { class: "param" },
      el("span", { class: "what" }, `${o2(f.label)} (`, ...symbolNodes(f.symbol), ")", el("small", { text: o2(f.unit) })),
      el("input", { type: "text", inputmode: "decimal", id: `p-${f.name}`, spellcheck: "false",
        oninput: () => {
          const v = readParams();
          const tissue = radio("tissue");
          $("#preset").value = samePreset(v, tissue) ? tissue : (samePreset(v, "skeletal") ? "skeletal"
            : samePreset(v, "cardiac") ? "cardiac" : "custom");
          markParams();
          validate();
        } }),
    ));
  }
}

function applyTissue() {
  const skeletal = radio("tissue") === "skeletal";
  for (const e of $$("[data-skeletal]")) e.hidden = !skeletal;
  if (state.info && $("#preset").value !== "custom") {
    $("#preset").value = radio("tissue");
    writeParams(presetValues(radio("tissue")));
  }
}

function applyFigureOptions() {
  $("#figure-options").hidden = !$("#figures").checked;
  $("#width-label").hidden = radio("layout") !== "publication";
  $("#range-values").hidden = radio("range") !== "fixed";
}

function settingsFromForm() {
  const steps = $("#steps").value.split(",");
  const size = commonSize();
  const s = {
    ...state.extras,
    tissue: radio("tissue"),
    use_fibre_types: $("#use-types").checked,
    width_um: size ? size.width_um : null,
    height_um: size ? size.height_um : null,
    roi_um: radio("roi") === "edges" ? ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1"].map((id) => num($(id).value)) : null,
    roi_fraction: radio("roi") === "shares" ? ["#roi-fx0", "#roi-fx1", "#roi-fy0", "#roi-fy1"].map((id) => num($(id).value) / 100) : null,
    parameters: state.info ? readParams() : null,
    exercise: $("#exercise").value || "resting",
    non_uniform: $("#non-uniform").checked,
    differential_extraction: num($("#diff-extraction").value) ?? 1.0,
    steps,
    figures: $("#figures").checked,
    dpi: Number($("#dpi").value),
    colormap: $("#colormap").value,
    figure_layout: radio("layout"),
    figure_width: $("#fig-width").value || "double",
    figure_formats: $$("#formats input:checked").map((i) => i.value),
    po2_range_mmHg: radio("range") === "fixed" ? [num($("#range-lo").value), num($("#range-hi").value)] : null,
    data_check: radio("dcheck"),
    demand: { enabled: $("#dm-enabled").checked, cutoff: num($("#dm-cutoff").value) ?? 1.5,
              switch_uptake: $("#dm-uptake").checked, scale_flux_cap: $("#dm-cap").checked },
  };
  if (state.info) Object.assign(s, advancedFromForm());
  // a fixed flux-line cap is typed in µm; the settings hold it in model units (half the tissue
  // height), so the saved JSON uses the first file's size, and each run its own (settingsForFile)
  if ($("#flux-cap").value === "fixed" && !$("#flux-legacy").checked) {
    const first = runnable().map(sizeOf).find(Boolean);
    s.flux.max_length = first ? capNdim(first[1]) : "auto";
  }
  return s;
}

const capNdim = (height_um) => num($("#flux-cap-um").value) / (height_um / 2);

// page-only options, sent with a run but not part of the settings JSON (the batch runner rejects unknown keys)
function webOptions() { return { save_session: $("#save-session").checked }; }

function settingsForFile(entry, base) {
  const s = { ...base, ...(fileSize(entry) || { width_um: null, height_um: null }) };
  if (entry.roi) { s.roi_um = entry.roi; s.roi_fraction = null; }
  if ($("#flux-cap").value === "fixed" && !$("#flux-legacy").checked) {
    const wh = sizeOf(entry);
    if (wh) s.flux = { ...s.flux, max_length: capNdim(wh[1]) };
  }
  return s;
}

// ---- advanced settings: retouch, mesh, flux lines ---------------------------------------------
function advancedFromForm() {
  const d = state.info.settings;
  const k = Math.round(num($("#rt-smooth-tol").value) || d.retouch.smooth_tol);
  const retouch = { ...d.retouch, ...(state.extras.retouch || {}),
    smooth: $("#rt-smooth").checked, smooth_tol: k % 2 ? k : k + 1,
    reduce: $("#rt-reduce").checked, reduce_tol: num($("#rt-reduce-tol").value) ?? d.retouch.reduce_tol,
    disjoint: $("#rt-disjoint").checked,
    tangent: $("#rt-tangent").checked, tangent_tol: num($("#rt-tangent-tol").value) ?? d.retouch.tangent_tol };
  const mesh = { ...d.mesh, ...(state.extras.mesh || {}),
    h_max: num($("#mesh-hmax").value) ?? d.mesh.h_max, wall_segments: Math.round(num($("#mesh-wall").value) ?? d.mesh.wall_segments),
    overlap_rule: $("#mesh-overlap").value };
  const legacy = $("#flux-legacy").checked;
  const cap = $("#flux-cap").value;
  const flux = { ...d.flux, ...(state.extras.flux || {}),
    lines_per_capillary: Math.round(num($("#flux-lines").value) ?? d.flux.lines_per_capillary),
    integrator: $("#flux-integrator").value,
    parametrization: legacy ? "time" : "arc_length",
    initial_radius: legacy ? 0.015 : null,
    max_length: legacy || cap === "auto" ? "auto" : cap === "none" ? null : "auto" };
  return { retouch, mesh, flux };
}

function fillAdvanced(d) {
  const def = state.info.settings;
  const rt = { ...def.retouch, ...(d.retouch || {}) };
  $("#rt-smooth").checked = !!rt.smooth;
  $("#rt-smooth-tol").value = rt.smooth_tol;
  $("#rt-reduce").checked = !!rt.reduce;
  $("#rt-reduce-tol").value = rt.reduce_tol;
  $("#rt-disjoint").checked = !!rt.disjoint;
  $("#rt-tangent").checked = !!rt.tangent;
  $("#rt-tangent-tol").value = rt.tangent_tol;
  const m = { ...def.mesh, ...(d.mesh || {}) };
  $("#mesh-hmax").value = m.h_max;
  $("#mesh-wall").value = m.wall_segments;
  $("#mesh-overlap").value = m.overlap_rule;
  const f = { ...def.flux, ...(d.flux || {}) };
  $("#flux-lines").value = f.lines_per_capillary;
  $("#flux-integrator").value = f.integrator === "ivp" ? "ivp" : "heun";
  $("#flux-legacy").checked = f.parametrization === "time";
  if (f.max_length === null) $("#flux-cap").value = "none";
  else if (f.max_length === "auto" || f.max_length === undefined) $("#flux-cap").value = "auto";
  else {
    $("#flux-cap").value = "fixed";
    const first = runnable().map(sizeOf).find(Boolean);   // model units -> µm with the first file's size
    if (first) $("#flux-cap-um").value = +(Number(f.max_length) * first[1] / 2).toFixed(1);
  }
  applyAdvanced();
}

function applyAdvanced() {
  $("#rt-smooth-tol").disabled = !$("#rt-smooth").checked;
  $("#rt-reduce-tol").disabled = !$("#rt-reduce").checked;
  $("#rt-tangent-tol").disabled = !$("#rt-tangent").checked;
  const legacy = $("#flux-legacy").checked;
  $("#flux-cap").disabled = legacy;
  $("#flux-cap-um-row").hidden = legacy || $("#flux-cap").value !== "fixed";
  // tell where the defaults were changed
  const d = state.info ? state.info.settings : null;
  if (d) {
    const a = advancedFromForm();
    const changed = JSON.stringify([a.retouch, a.mesh]) !== JSON.stringify([{ ...d.retouch, ...(state.extras.retouch || {}) }, { ...d.mesh, ...(state.extras.mesh || {}) }])
      || a.flux.lines_per_capillary !== d.flux.lines_per_capillary || a.flux.integrator !== d.flux.integrator || legacy
      || $("#flux-cap").value !== "auto" || !$("#save-session").checked || $("#parallel").value !== "auto";
    $("#advanced summary .hint").textContent = changed ? "changed from the defaults" : "retouch, mesh, flux lines, session files";
    $("#advanced").classList.toggle("changed", changed);
  }
}

function formToDefaults() {
  const d = state.info.settings;
  state.extras = {};
  fillForm({ ...d, parameters: null });
  state.defaultSize = null;
}

function fillForm(d) {
  setRadio("tissue", d.tissue || "skeletal");
  $("#use-types").checked = d.use_fibre_types !== false;
  // a size in the settings JSON goes to every file that has none yet (and to files added later)
  if (d.width_um || d.height_um) {
    state.defaultSize = d.width_um ? { given: "width", value: String(d.width_um) } : { given: "height", value: String(d.height_um) };
    for (const f of state.files) if (!fileSize(f)) { f.given = state.defaultSize.given; f.value = state.defaultSize.value; }
  }
  if (d.roi_um) {
    setRadio("roi", "edges");
    ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1"].forEach((id, i) => { $(id).value = d.roi_um[i]; });
  } else if (d.roi_fraction) {
    setRadio("roi", "shares");
    ["#roi-fx0", "#roi-fx1", "#roi-fy0", "#roi-fy1"].forEach((id, i) => { $(id).value = +(100 * d.roi_fraction[i]).toFixed(3); });
  } else setRadio("roi", "default");
  applyRoi();
  setRadio("dcheck", d.data_check || "fix");
  const dm = { enabled: true, cutoff: 1.5, switch_uptake: true, scale_flux_cap: true, ...(d.demand || {}) };
  $("#dm-enabled").checked = dm.enabled !== false;
  $("#dm-uptake").checked = dm.switch_uptake !== false;
  $("#dm-cap").checked = dm.scale_flux_cap !== false;
  $("#dm-cutoff").value = dm.cutoff;
  fillAdvanced(d);
  const tissue = d.tissue || "skeletal";
  const values = { ...presetValues(tissue), ...(d.parameters || {}) };
  $("#preset").value = samePreset(values, tissue) ? tissue : "custom";
  writeParams(values);
  $("#exercise").value = d.exercise || "resting";
  $("#non-uniform").checked = d.non_uniform !== false;
  $("#diff-extraction").value = d.differential_extraction ?? 1;
  const steps = (d.steps || ["indices", "po2", "flux"]).join(",");
  $("#steps").value = ["indices", "indices,po2", "indices,po2,flux"].includes(steps) ? steps
    : steps.includes("flux") ? "indices,po2,flux" : steps.includes("po2") ? "indices,po2" : "indices";
  $("#figures").checked = d.figures !== false;
  setRadio("layout", d.figure_layout === "legacy" ? "legacy" : "publication");
  $("#fig-width").value = d.figure_width || "double";
  const fmts = new Set(d.figure_formats || ["png", "pdf"]);
  for (const i of $$("#formats input")) i.checked = fmts.has(i.value);
  const dpi = d.dpi || (d.figure_layout === "legacy" ? 300 : 600);
  if (![...$("#dpi").options].some((o) => Number(o.value) === dpi)) $("#dpi").append(el("option", { value: dpi, text: `${dpi} dpi` }));
  $("#dpi").value = String(dpi);
  $("#colormap").value = d.colormap || "turbo";
  if (d.po2_range_mmHg) {
    setRadio("range", "fixed");
    $("#range-lo").value = d.po2_range_mmHg[0];
    $("#range-hi").value = d.po2_range_mmHg[1];
  } else setRadio("range", "auto");
  applyTissue();
  applyFigureOptions();
  refreshSizes();
  validate();
}

const FORM_KEYS = new Set(["tissue", "use_fibre_types", "width_um", "height_um", "roi_um", "roi_fraction", "parameters",
  "exercise", "non_uniform", "differential_extraction", "steps", "figures", "dpi", "colormap", "figure_layout",
  "figure_width", "figure_formats", "po2_range_mmHg", "data_check", "demand"]);
// retouch, mesh and flux are shown in part (Advanced settings): the rest of each is kept as loaded
const PARTLY_SHOWN = ["retouch", "mesh", "flux"];

function applyRoi() {
  $("#roi-edges").hidden = radio("roi") !== "edges";
  $("#roi-shares").hidden = radio("roi") !== "shares";
}

async function loadSettingsFile(file) {
  let d;
  try { d = JSON.parse(await file.text()); } catch { return toast("Settings not loaded: not a JSON file.", true); }
  if (!d || typeof d !== "object" || Array.isArray(d)) return toast("Settings not loaded: not an OTM settings file.", true);
  await engine.started;
  const notes = [];
  if (d.parameter_file) { notes.push("the .dat file it names cannot be opened from the browser; load it under Biophysical parameters"); delete d.parameter_file; }
  const problem = await engine.call("check", { settings: d });
  if (problem) return toast(`Settings not loaded: ${problem}`, true, 7000);
  state.extras = Object.fromEntries(Object.entries(d).filter(([k]) => !FORM_KEYS.has(k) && !PARTLY_SHOWN.includes(k)));
  for (const k of PARTLY_SHOWN) if (d[k]) state.extras[k] = { ...d[k] };
  fillForm(d);
  toast(`Settings loaded from ${file.name}` + (notes.length ? ` (${notes.join("; ")})` : "."), notes.length > 0, 6000);
}

function validate() {
  const msgs = [];
  if (radio("roi") === "edges") {
    const r = ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1"].map((id) => num($(id).value));
    if (r.some((v) => v === null || Number.isNaN(v))) msgs.push("give all four edges of the region of interest");
    else if (!(r[0] < r[1] && r[2] < r[3])) msgs.push("each region edge must be larger than the one before it ('from' < 'to')");
  }
  if (radio("roi") === "shares") {
    const r = ["#roi-fx0", "#roi-fx1", "#roi-fy0", "#roi-fy1"].map((id) => num($(id).value));
    if (r.some((v) => v === null || Number.isNaN(v) || v < 0 || v > 100)) msgs.push("give all four region shares between 0 and 100 %");
    else if (!(r[0] < r[1] && r[2] < r[3])) msgs.push("each region share must be larger than the one before it ('from' < 'to')");
  }
  if (state.info && Object.values(readParams()).some((v) => v === null || Number.isNaN(v) || v < 0)) msgs.push("every biophysical parameter needs a number of 0 or more");
  const adv = [["#rt-smooth-tol", 1, 51, "smoothing window"], ["#rt-reduce-tol", 0.05, 20, "vertex-reduction tolerance"],
    ["#rt-tangent-tol", 0.1, 20, "gap between touching fibres"], ["#mesh-hmax", 0.002, 0.2, "largest mesh element"],
    ["#mesh-wall", 8, 96, "segments per capillary wall"], ["#flux-lines", 1, 256, "flux lines per capillary"]];
  for (const [id, lo, hi, what] of adv) {
    if ($(id).disabled) continue;
    const v = num($(id).value);
    if (v === null || Number.isNaN(v) || v < lo || v > hi) msgs.push(`${what} must be between ${lo} and ${hi} (Advanced settings)`);
  }
  if ($("#flux-cap").value === "fixed" && !$("#flux-legacy").checked) {
    const v = num($("#flux-cap-um").value);
    if (v === null || Number.isNaN(v) || v <= 0) msgs.push("the fixed flux-line cap needs a length in µm (Advanced settings)");
  }
  if ($("#figures").checked && !$$("#formats input:checked").length) msgs.push("choose at least one figure format");
  if (radio("range") === "fixed") {
    const lo = num($("#range-lo").value), hi = num($("#range-hi").value);
    if (lo === null || hi === null || !(hi > lo)) msgs.push("the fixed PO₂ range needs a lowest value below the highest");
  }
  const co = num($("#dm-cutoff").value);
  if ($("#dm-enabled").checked && (co === null || Number.isNaN(co) || co <= 1)) msgs.push("the demand cut-off must be above 1 × the resting Λ");
  const de = num($("#diff-extraction").value);
  if (radio("tissue") === "skeletal" && (de === null || de < 0.1 || de > 10)) msgs.push("differential extraction must be between 0.1 and 10");
  updateExercise();
  updateDemand();
  const box = $("#settings-error");
  box.hidden = !msgs.length;
  box.textContent = msgs.length ? "Fix before running: " + msgs.join("; ") + "." : "";
  updateRunState();
  return msgs.length === 0;
}

function setupSettings(info) {
  $("#exercise").textContent = "";
  for (const lv of info.exercise_levels) $("#exercise").append(el("option", { value: lv, text: lv[0].toUpperCase() + lv.slice(1) }));
  $("#fig-width").textContent = "";
  for (const [k, label] of Object.entries(info.widths)) $("#fig-width").append(el("option", { value: k, text: label }));
  const tips = { png: "PNG image: slides, Word, web", pdf: "PDF (vector): journals, LaTeX, Illustrator",
                 svg: "SVG (vector): Inkscape, Illustrator, PowerPoint", tiff: "TIFF image (LZW): journals that ask for TIFF" };
  $("#formats").textContent = "";
  for (const f of info.formats) $("#formats").append(el("label", { title: tips[f] || "" }, el("input", { type: "checkbox", value: f, onchange: validate }), f.toUpperCase()));
  buildParamGrid();
  formToDefaults();
}

function wireSettings() {
  for (const r of $$('input[name="tissue"]')) r.addEventListener("change", () => { applyTissue(); validate(); recheckFiles(); });
  $("#use-types").addEventListener("change", recheckFiles);
  for (const r of $$('input[name="dcheck"]')) r.addEventListener("change", () => renderFiles());
  for (const r of $$('input[name="roi"]')) r.addEventListener("change", () => { applyRoi(); validate(); });
  for (const id of ["#roi-fx0", "#roi-fx1", "#roi-fy0", "#roi-fy1"]) $(id).addEventListener("input", validate);
  for (const e of $$("#advanced input, #advanced select")) e.addEventListener("change", () => { applyAdvanced(); validate(); });
  for (const r of $$('input[name="layout"]')) r.addEventListener("change", () => {
    const dpi = Number($("#dpi").value);
    if (radio("layout") === "legacy" && dpi === 600) $("#dpi").value = "300";
    if (radio("layout") === "publication" && dpi === 300) $("#dpi").value = "600";
    applyFigureOptions();
  });
  for (const r of $$('input[name="range"]')) r.addEventListener("change", () => { applyFigureOptions(); validate(); });
  $("#figures").addEventListener("change", () => { applyFigureOptions(); validate(); });
  for (const id of ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1", "#range-lo", "#range-hi", "#diff-extraction"]) $(id).addEventListener("input", validate);
  $("#exercise").addEventListener("change", validate);
  for (const id of ["#dm-enabled", "#dm-uptake", "#dm-cap", "#dm-cutoff"]) $(id).addEventListener("input", validate);
  $("#preset").addEventListener("change", () => {
    const p = $("#preset").value;
    if (p !== "custom") writeParams(presetValues(p));
    validate();
  });
  $("#save-dat").addEventListener("click", () => {
    const v = readParams();
    const text = paramFields().map((f) => fmtParam(v[f.name])).join("\n") + "\n";
    download(new Blob([text], { type: "text/plain" }), `${radio("tissue")}_parameters.dat`);
  });
  $("#dat-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    await engine.started;
    const res = await engine.call("parseDat", { text: await file.text() });
    if (!res.ok) return toast(`Parameters not loaded: ${res.error}`, true);
    writeParams(res.values);
    const tissue = radio("tissue");
    $("#preset").value = samePreset(res.values, tissue) ? tissue : "custom";
    markParams();
    validate();
    toast(`Parameters loaded from ${file.name}.`);
  });
  $("#save-settings").addEventListener("click", () => {
    const s = settingsFromForm();
    download(new Blob([JSON.stringify(s, null, 2)], { type: "application/json" }), "OTM_settings.json");
  });
  $("#settings-input").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) loadSettingsFile(f); });
  $("#reset-settings").addEventListener("click", () => { if (state.info) { formToDefaults(); toast("Settings reset to the defaults."); } });
}

// ============================================================================================
// 3 RUN
// ============================================================================================
function runnable() { return state.files.filter((f) => f.status === "ok"); }

function updateRunState() {
  const btn = $("#run-button");
  const files = runnable();
  const missingSize = files.filter((f) => !fileSize(f));
  const checking = state.files.some((f) => f.status === "checking" || f.status === "waiting");
  let hint = "";
  if (!engine.ready) hint = engine.failed ? "The analysis engine could not start (see the top right)." : "Loading the analysis engine…";
  else if (!state.files.length) hint = "Add files to start.";
  else if (checking) hint = "Checking the files…";
  else if (!files.length) hint = "None of the files can be read.";
  else if (missingSize.length) hint = `Type the width or the height in the file table for: ${missingSize.map((f) => f.name).join(", ")}.`;
  else if (!$("#settings-error").hidden) hint = "Fix the settings first.";
  else {
    hint = `${files.length} file(s) ready.` + (state.files.length > files.length ? ` ${state.files.length - files.length} unreadable file(s) will be left out.` : "");
    const flagged = files.filter((f) => f.check && f.check.check && (f.check.check.status === "error" || f.check.check.status === "fatal"));
    if (flagged.length) {
      const pol = radio("dcheck");
      hint += ` Data check: ${flagged.length} file(s) with segmentation errors` + (pol === "skip" ? " will be skipped." : pol === "keep" ? " keep their flagged items where allowed." : ": the flagged items will be removed first.");
    }
  }
  if (state.running) hint = "Running… you can keep using this tab; leaving the page stops the run.";
  $("#run-hint").textContent = hint;
  btn.disabled = state.running || !engine.ready || !files.length || missingSize.length > 0 || checking || !$("#settings-error").hidden;
  btn.textContent = files.length > 1 ? `Run ${files.length} files` : "Run";
  $("#stop-button").hidden = !state.running;
  $("#zip-button").disabled = state.running || state.results.size === 0;
  const steps = $$(".step-link");
  steps[0].classList.toggle("done", files.length > 0);
  steps[1].classList.toggle("done", files.length > 0 && missingSize.length === 0 && $("#settings-error").hidden && engine.ready);
  steps[2].classList.toggle("running", state.running);
  steps[2].classList.toggle("done", !state.running && state.results.size > 0);
  steps[3].classList.toggle("done", state.results.size > 0);
}

const STEP_LABELS = { load: "Reading the file", geometry: "Tissue size and region", indices: "Supply indices",
  mesh: "Meshing", po2: "Solving PO₂", flux: "Flux lines", export: "Writing tables", figures: "Writing figures (results shown)", done: "Done",
  "finished with errors": "Finished with errors" };
const progressRows = new Map();

function progressRow(name) {
  let row = progressRows.get(name);
  if (!row) {
    row = el("li", {},
      el("div", { class: "pname", text: name }),
      el("div", {}, el("div", { class: "pstep", text: "Waiting" }), el("div", { class: "bar" }, el("i"))),
      el("div", { class: "ptime", text: "" }));
    $("#progress-list").append(row);
    progressRows.set(name, row);
  }
  return row;
}

let runClock = null;
function updateProgress(name, f, msg) {
  const row = progressRow(name);
  const key = String(msg || "").split(":")[0].trim();
  $(".pstep", row).textContent = STEP_LABELS[key] || (msg ? msg[0].toUpperCase() + msg.slice(1) : "");
  $(".bar i", row).style.width = `${Math.round(Math.max(0, Math.min(1, f)) * 100)}%`;
}

// ---- several samples at once: extra workers, each with its own Python ------------------------
function parallelCount(nFiles) {
  const choice = $("#parallel").value;
  if (choice !== "auto") return Math.max(1, Math.min(Number(choice), nFiles));
  const cores = navigator.hardwareConcurrency || 2;
  const mem = navigator.deviceMemory;                 // GB, Chrome and Edge only (capped at 8)
  if (nFiles < 2 || cores < 4 || (mem !== undefined && mem < 8)) return 1;
  return Math.min(nFiles, cores >= 8 ? 3 : 2);
}

function extraEngine() {
  const e = new Engine();
  e.onProgress = (p) => updateProgress(p.name, p.f, p.msg);
  return e;
}

// a ZIP written with ZIP_STORED by otm_web.pack: name -> bytes (no copy)
function readStoredZip(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Map();
  let i = 0;
  const dec = new TextDecoder();
  while (i + 30 <= u8.length && dv.getUint32(i, true) === 0x04034b50) {
    const method = dv.getUint16(i + 8, true), size = dv.getUint32(i + 18, true);
    const nlen = dv.getUint16(i + 26, true), xlen = dv.getUint16(i + 28, true);
    const name = dec.decode(u8.subarray(i + 30, i + 30 + nlen));
    const start = i + 30 + nlen + xlen;
    if (method !== 0) throw new Error("packed sample is compressed");
    out.set(name, u8.subarray(start, start + size));
    i = start + size;
  }
  return out;
}

async function runOne(f, base, eng) {
  const row = progressRow(f.name);
  row.className = "";
  const started = performance.now();
  const clock = setInterval(() => { $(".ptime", row).textContent = `${((performance.now() - started) / 1000).toFixed(0)} s`; }, 500);
  setOverviewRow(f.name, { name: f.name, running: true });
  try {
    await putFile(f, eng);
    const later = $("#figures").checked;
    const res = await eng.call("run", { name: f.name, settings: { ...settingsForFile(f, base), ...webOptions(), figures_later: later }, edits: f.edits });
    res.file = f.mat.name;
    res.seconds_tables = res.seconds;
    res.worker = eng === engine ? "main" : "extra";
    if (eng !== engine) res.owner = "extra";
    state.packs.delete(f.name);
    storeResult(res);                                 // the numbers now, the figures below
    if (res.figures_pending) {
      $(".pstep", row).textContent = "Results shown · writing figures…";
      const fr = await eng.call("figures", { name: f.name });
      res.files = fr.files;
      if (!fr.ok) { res.ok = false; res.errors = [...(res.errors || []), { step: "export", message: fr.error }]; }
      res.seconds = (performance.now() - started) / 1000;
    }
    if (eng !== engine) {                             // the main worker keeps every result (files, ZIP)
      const bytes = await eng.call("pack", { name: f.name });
      const files = readStoredZip(bytes);
      state.packs.set(f.name, files);
      delete res.owner;
      const copy = bytes.slice();
      state.unpacks.push(engine.call("unpack", { name: f.name, buffer: copy.buffer }, [copy.buffer])
        .then(() => state.packs.delete(f.name))
        .catch((e) => logEvent(`moving ${f.name} to the main worker: ${e.message}`)));
      eng.call("forget", { name: f.name }).catch(() => {});
    }
    updateResult(res);
    row.classList.add(res.ok ? "done" : "failed");
    $(".pstep", row).textContent = res.ok ? `Done · ${res.files.length} files written` : res.errors.map((e) => e.message).join(" · ");
    $(".bar i", row).style.width = "100%";
  } catch (e) {
    row.classList.add("failed");
    $(".pstep", row).textContent = `Failed: ${e.message}`;
    storeResult({ name: f.name, ok: false, errors: [{ step: "run", message: e.message }], files: [], summary: null, seconds: (performance.now() - started) / 1000 });
    logEvent(`run ${f.name}: ${e.message}`, e.details);
    if (eng.failed) throw e;
  } finally {
    clearInterval(clock);
    $(".ptime", row).textContent = `${((performance.now() - started) / 1000).toFixed(0)} s`;
    eng.call("drop", { name: f.name }).catch(() => {});
  }
}

async function runAll() {
  if (!validate()) return;
  const files = runnable();
  const base = settingsFromForm();
  // check every file's settings with otm_core before starting
  for (const f of files) {
    const problem = await engine.call("check", { settings: settingsForFile(f, base) });
    if (problem) { toast(`${f.name}: ${problem}`, true, 8000); return; }
  }
  state.running = true;
  state.stopRequested = false;
  state.lastSettings = base;
  progressRows.clear();
  $("#progress-list").textContent = "";
  for (const f of files) progressRow(f.name);
  updateRunState();
  const t0 = performance.now();
  const queue = files.slice();
  const loop = async (eng) => {
    while (queue.length && !state.stopRequested) {
      const f = queue.shift();
      try { await runOne(f, base, eng); } catch { return; }       // this worker stopped working
    }
  };
  const n = parallelCount(files.length);
  const extras = [];
  const loops = [loop(engine)];
  for (let k = 1; k < n; k++) {
    const e = extraEngine();
    extras.push(e);
    loops.push(e.started.then(() => loop(e)).catch((err) => logEvent(`extra worker ${k}: ${err.message}`)));
  }
  if (n > 1) toast(`Running ${n} samples at a time (each in its own worker).`);
  await Promise.all(loops);
  for (const f of queue) $(".pstep", progressRow(f.name)).textContent = "Not run (stopped)";
  await Promise.all(state.unpacks);
  state.unpacks = [];
  for (const e of extras) e.worker.terminate();     // give their memory back
  state.running = false;
  updateRunState();
  const failed = files.filter((f) => state.results.get(f.name) && !state.results.get(f.name).ok).length;
  const done = files.filter((f) => state.results.has(f.name)).length;
  toast(`Run finished in ${((performance.now() - t0) / 1000).toFixed(0)} s: ${done - failed} of ${files.length} sample(s) without errors.`, failed > 0, 6000);
}

// ============================================================================================
// 4 RESULTS
// ============================================================================================
function storeResult(res) {
  if (!state.results.has(res.name)) state.order.push(res.name);
  state.results.set(res.name, res);
  state.views.delete(res.name);
  for (const e of res.errors || []) logEvent(`${res.name}: ${e.message}`);
  for (const k of [...state.figureCache.keys()]) if (k.startsWith(res.name + "/")) { URL.revokeObjectURL(state.figureCache.get(k)); state.figureCache.delete(k); }
  setOverviewRow(res.name, res);
  if (!state.selected || state.selected === res.name) selectSample(res.name);
  $("#results-hint").textContent = "Click a sample to see its tables and figures. The ZIP holds, per sample, the INDICES and PO2 folders the desktop app writes, plus the batch summary spreadsheet.";
}

// the figures of a result arrived (or it changed after the run): refresh what shows its files
function updateResult(res) {
  state.results.set(res.name, res);
  setOverviewRow(res.name, res);
  if (state.selected !== res.name) return;
  const node = $("#sample-view .sample");
  if (!node) return;
  const fresh = $("#tpl-sample").content.firstElementChild.cloneNode(true);
  fillFiles(fresh, res);
  for (const sel of [".block-figures", ".files-list"]) $(sel, node).replaceWith($(sel, fresh));
  const otm = (res.files || []).find((p) => p.toLowerCase().endsWith(".otm"));
  const otmBtn = $(".save-otm", node);
  if (otm && otmBtn.hidden) { otmBtn.hidden = false; otmBtn.addEventListener("click", () => saveOutput(otm)); }
  $(".sample-meta", node).textContent = sampleMeta(res);
  loadThumbnails($(".gallery", node));
  const host = $(".tv-host", node);
  if (host && !currentView && res.view) showTissueView($(".block-view", node), res);   // it was waiting for the pack
}

function setOverviewRow(name, res) {
  $("#overview-wrap").hidden = false;
  const tbody = $("#overview tbody");
  let tr = $(`tr[data-name="${CSS.escape(name)}"]`, tbody);
  if (!tr) {
    tr = el("tr", { "data-name": name, tabindex: "0", onclick: () => selectSample(name),
      onkeydown: (e) => { if (e.key === "Enter") selectSample(name); } });
    tbody.append(tr);
  }
  const s = res.summary || {};
  const status = res.running ? el("span", { class: "badge running", text: "running" })
    : el("span", { class: "badge " + (res.ok ? "ok" : "error"), text: res.ok ? "ok" : "error" });
  const hyp = s.po2_tissue_hypoxic_pct;
  tr.textContent = "";
  tr.append(
    el("td", {}, el("b", { text: name })),
    el("td", {}, status),
    el("td", { class: "num", text: s.n_capillaries !== undefined ? fmtInt(s.n_capillaries) : "" }),
    el("td", { class: "num", text: s.LCFR_mean !== undefined ? fmt(s.LCFR_mean, 3) : "" }),
    el("td", { class: "num" + (s.uptake_switched_by_lambda === "yes" ? " warn-text" : ""), title: s.demand_regime || "",
      text: s.lambda_ratio !== undefined && s.lambda_ratio !== null ? fmt(s.lambda_ratio, 2) : "" }),
    el("td", { class: "num", text: s.po2_tissue_mean_mmHg !== undefined ? fmt(s.po2_tissue_mean_mmHg, 2) : "" }),
    el("td", { class: "num" + (hyp > 0 ? " warn-text" : ""), text: hyp !== undefined ? fmt(hyp, 1) : "" }),
    el("td", { class: "num", text: res.seconds ? fmt(res.seconds, 0) : "" }),
  );
  tr.classList.toggle("selected", state.selected === name);
}

function fillTable(table, rows, numCols, hypoxicRows = []) {
  const tbody = $("tbody", table);
  tbody.textContent = "";
  rows.forEach((r, i) => {
    tbody.append(el("tr", { class: hypoxicRows.includes(i) ? "hypoxic" : "" },
      ...r.map((v, j) => el("td", { class: numCols.includes(j) ? "num" : "", text: v }))));
  });
}

function tableText(table) {
  return $$("tr", table).map((tr) => $$("th,td", tr).map((c) => c.textContent).join("\t")).join("\n");
}

function sampleMeta(res) {
  const meta = [];
  if (res.size_um) meta.push(`${fmt(res.size_um[0], 0)} × ${fmt(res.size_um[1], 0)} µm`);
  if (res.roi_um) meta.push(`ROI x ${fmt(res.roi_um[0], 0)}–${fmt(res.roi_um[1], 0)}, y ${fmt(res.roi_um[2], 0)}–${fmt(res.roi_um[3], 0)} µm`);
  if (res.mesh) meta.push(`mesh ${fmtInt(res.mesh.nodes)} nodes (${res.mesh.mesher})`);
  if (res.solve) meta.push(`${res.solve.converged ? "converged" : "NOT converged"} in ${res.solve.iterations} Newton iteration(s), PO₂ ${fmt(res.solve.po2_min_mmHg)}–${fmt(res.solve.po2_max_mmHg)} mmHg`);
  if (res.flux) meta.push(`flux lines from ${res.flux.capillaries_with_lines} of ${res.flux.seed_capillaries} ROI capillaries`);
  if (res.switches && res.po2) meta.push(`exercise ${LEVEL_NAMES[res.switches.exercise] || res.switches.exercise}: O₂ demand × ${fmtNum(res.switches.demand)}`);
  if (res.flux && res.flux.cap_um) meta.push(`flux-line cap ${fmt(res.flux.cap_um, 1)} µm`);
  if (res.corrections) meta.push(`${res.corrections} correction(s) applied${res.corrections_from === "page" ? " (made here)" : " (from the .otm)"}`);
  if (res.figures_pending && res.seconds_tables === res.seconds) meta.push("figures being written…");
  else if (res.seconds) meta.push(`${fmt(res.seconds, 0)} s`);
  return meta.join(" · ");
}

function fillFiles(node, res) {
  const files = res.files || [];
  const pngs = files.filter((p) => p.toLowerCase().endsWith(".png"));
  const gallery = $(".gallery", node);
  if (!pngs.length) $(".block-figures", node).replaceChildren(el("h4", { text: "Figures" }),
    el("p", { class: "empty", text: res.figures_pending && res.seconds_tables === res.seconds ? "Writing the figures…" : "No figures in this run." }));
  for (const p of pngs) gallery.append(figureCard(p, files));
  const ul = $(".files-list ul", node);
  for (const p of files) ul.append(el("li", {}, el("a", { href: "#", onclick: (e) => { e.preventDefault(); saveOutput(p); }, text: p.split("/").slice(1).join("/") })));
  $(".files-list", node).hidden = !files.length;
}

function selectSample(name) {
  state.selected = name;
  for (const tr of $$("#overview tbody tr")) tr.classList.toggle("selected", tr.dataset.name === name);
  const res = state.results.get(name);
  const view = $("#sample-view");
  view.textContent = "";
  if (!res) return;
  const node = $("#tpl-sample").content.firstElementChild.cloneNode(true);
  $(".sample-name", node).textContent = name;
  const badge = $(".badge", node);
  badge.textContent = res.ok ? "ok" : "error";
  badge.classList.add(res.ok ? "ok" : "error");
  $(".sample-meta", node).textContent = sampleMeta(res);
  const errs = $(".errors", node);
  for (const e of res.errors || []) errs.append(el("p", { text: e.message }));
  for (const w of res.corrections_skipped || []) errs.append(el("p", { text: `Correction skipped: ${w}` }));
  showCheckResult($(".dc-result", node), res.check);
  if (res.demand) {
    const dm = $(".demand-result", node);
    dm.hidden = false;
    dm.classList.toggle("switch", !!res.demand.uptake_switched_by_lambda);
    dm.textContent = res.demand.text;
  }
  const otm = (res.files || []).find((p) => p.toLowerCase().endsWith(".otm"));
  const otmBtn = $(".save-otm", node);
  if (otm) { otmBtn.hidden = false; otmBtn.addEventListener("click", () => saveOutput(otm)); }

  const ind = $(".block-indices", node);
  if (res.indices) fillTable($("table", ind), res.indices, [1]);
  else ind.replaceChildren(el("h4", { text: "Supply indices" }), el("p", { class: "empty", text: "Not computed in this run." }));
  const po2 = $(".block-po2", node);
  if (res.po2) {
    fillTable($("table", po2), res.po2, [1, 2, 3], res.po2_hypoxic_rows || []);
    fillTable($("table.mo2", po2), res.mo2 || [], [1, 2, 3]);
    if (res.hypoxia) { const h = $(".hypoxia", po2); h.hidden = false; h.textContent = res.hypoxia; }
  } else po2.replaceChildren(el("h4", { text: "PO₂" }), el("p", { class: "empty", text: "Not computed in this run." }));
  for (const btn of $$(".copy", node)) {
    btn.addEventListener("click", async () => {
      const table = btn.closest(".block-head").nextElementSibling.matches("table") ? btn.closest(".block-head").nextElementSibling
        : btn.closest(".block-head").nextElementSibling.nextElementSibling;
      try { await navigator.clipboard.writeText(tableText(table)); toast("Table copied: paste it into Excel or Word."); }
      catch { toast("Copy is blocked by the browser here; select the table and press Ctrl+C.", true); }
    });
  }
  fillFiles(node, res);
  view.append(node);
  loadThumbnails($(".gallery", node));
  showTissueView($(".block-view", node), res);
}

// what the data check found and did, under a sample's name (nothing when it passed)
function showCheckResult(box, q) {
  box.textContent = "";
  if (!q || (q.status === "ok" && !q.fixed.length)) { box.hidden = true; return; }
  box.hidden = false;
  box.className = "dc-result " + (q.status === "ok" || q.status === "warning" ? "note" : "warn");
  box.append(el("b", { text: q.headline }));
  const items = [...q.fixed.map((i) => `${i.code} ${i.title}: ${i.message} (removed)`), ...q.issues.map((i) => `${i.code} ${i.title}: ${i.message}`)];
  if (items.length) box.append(el("ul", {}, ...items.map((t) => el("li", { text: t }))));
  if (q.actions.length) box.append(el("p", { class: "small", text: q.actions.join(" · ") }));
}

// ---- tissue view ------------------------------------------------------------------------------
async function viewData(name) {
  if (state.views.has(name)) return state.views.get(name);
  const pk = state.packs.get(name);
  const res = state.results.get(name);
  if (!pk && res && res.owner === "extra") return "pending";        // still in its own worker
  const bytes = pk ? (pk.get("view.bin") || null) : await engine.call("view", { name });
  if (!bytes) return null;
  const v = parseView(bytes.buffer.byteLength === bytes.byteLength ? bytes.buffer : bytes.slice().buffer);
  state.views.set(name, v);
  while (state.views.size > 4) state.views.delete(state.views.keys().next().value);   // keep the last few
  return v;
}

let currentView = null;
async function showTissueView(block, res) {
  if (currentView) { currentView.destroy(); currentView = null; }
  const host = $(".tv-host", block);
  if (!res.view) {
    block.hidden = !res.ok ? true : false;
    host.replaceChildren(el("p", { class: "empty", text: res.view_error ? `The tissue view could not be made: ${res.view_error}` : "No tissue view for this run." }));
    return;
  }
  host.replaceChildren(el("p", { class: "empty", text: "Loading the tissue view…" }));
  let data;
  try { data = await viewData(res.name); }
  catch (e) { host.replaceChildren(el("p", { class: "empty", text: `The tissue view could not be loaded: ${e.message}` })); return; }
  if (!host.isConnected || state.selected !== res.name) return;   // another sample was selected meanwhile
  if (data === "pending") { host.replaceChildren(el("p", { class: "empty", text: "The tissue view appears when this sample's figures are written." })); return; }
  if (!data) { host.replaceChildren(el("p", { class: "empty", text: "No tissue view for this run." })); return; }
  host.textContent = "";
  const settings = state.lastSettings || {};
  const range = state.viewRange !== undefined ? state.viewRange : (settings.po2_range_mmHg || null);
  currentView = new TissueView(host, {
    colormap: state.viewColormap || settings.colormap || "turbo", range,
    onRange: (r) => { state.viewRange = r; },
  });
  currentView.el.cmap.addEventListener("change", () => { state.viewColormap = currentView.colormap; });
  currentView.setData(data);
}

const baseName = (p) => p.split("/").pop();

function figureCard(path, all) {
  const stem = path.replace(/\.png$/i, "");
  const siblings = all.filter((p) => p !== path && p.replace(/\.[^.]+$/, "") === stem);
  const sample = path.split("/")[0];
  let label = baseName(stem);
  if (label.startsWith(sample + "_") || label.startsWith(sample + " ")) label = label.slice(sample.length + 1);
  label = label.replace(/_/g, " ");
  const links = [path, ...siblings].map((p) => el("a", { href: "#", title: `Save ${baseName(p)}`,
    onclick: (e) => { e.preventDefault(); saveOutput(p); }, text: p.split(".").pop().toUpperCase() }));
  const sep = [];
  links.forEach((a, i) => { if (i) sep.push(" "); sep.push(a); });
  return el("figure", { class: "fig", "data-path": path },
    el("button", { class: "open", type: "button", title: "Open larger", "aria-label": `Open ${label}`, onclick: () => openFigure(path, label) },
      el("img", { alt: label, loading: "lazy" })),
    el("figcaption", {}, el("span", { title: baseName(path), text: label }), el("span", {}, ...sep)));
}

// an output file: from the main worker, or from the pack of a sample run in an extra worker
async function outputBytes(path) {
  const pk = state.packs.get(path.split("/")[0]);
  if (pk && pk.has("out/" + path)) return pk.get("out/" + path);
  return engine.call("file", { path });
}

async function figureURL(path) {
  if (state.figureCache.has(path)) return state.figureCache.get(path);
  const bytes = await outputBytes(path);
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
  state.figureCache.set(path, url);
  return url;
}

async function loadThumbnails(gallery) {
  if (!gallery) return;
  for (const fig of $$(".fig", gallery)) {
    if (!fig.isConnected) return;               // another sample was selected meanwhile
    try { $("img", fig).src = await figureURL(fig.dataset.path); }
    catch { $("img", fig).alt = "could not be shown"; }
  }
}

async function openFigure(path, label) {
  const dlg = el("dialog", { class: "viewer" },
    el("div", { class: "vbar" }, el("span", { text: label }), el("button", { class: "small", type: "button", text: "Close", onclick: () => dlg.close() })),
    el("img", { alt: label, src: await figureURL(path) }));
  dlg.addEventListener("close", () => dlg.remove());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  document.body.append(dlg);
  dlg.showModal();
}

const MIME = { png: "image/png", pdf: "application/pdf", svg: "image/svg+xml", tiff: "image/tiff", tif: "image/tiff",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", csv: "text/csv", txt: "text/plain", json: "application/json" };

async function saveOutput(path) {
  try {
    const bytes = await outputBytes(path);
    const ext = path.split(".").pop().toLowerCase();
    download(new Blob([bytes], { type: MIME[ext] || "application/octet-stream" }), baseName(path));
  } catch (e) { toast(`Could not save ${baseName(path)}: ${e.message}`, true); }
}

async function downloadZip() {
  const btn = $("#zip-button");
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "Packing the ZIP…";
  try {
    const bytes = await engine.call("zip", { names: state.order, settings: state.lastSettings || settingsFromForm() });
    const d = new Date();
    const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    download(new Blob([bytes], { type: "application/zip" }), `OTM_results_${stamp}.zip`);
  } catch (e) { toast(`ZIP not made: ${e.message}`, true); }
  btn.textContent = old;
  updateRunState();
}

// ============================================================================================
// CORRECT DATA (the desktop's "Correct data" step, with the data check's flagged items)
// ============================================================================================
const ed = { entry: null, edits: [], redo: [], start: null, editor: null, busy: false, pending: false };

async function openEditor(entry) {
  if (state.running) return toast("Wait until the run has finished.", true);
  const dlg = $("#editor-dialog");
  ed.entry = entry;
  ed.start = editsOf(entry).slice();
  ed.edits = ed.start.slice();
  ed.redo = [];
  $("#ed-title").textContent = `Correct data · ${entry.name}`;
  $("#ed-check").textContent = "Loading…";
  if (!ed.editor) {
    ed.editor = new CorrectionsEditor($("#ed-stage"), {
      onEdit: (e) => { ed.edits.push(e); ed.redo = []; refreshEditor(); },
      onHover: (t) => { $("#ed-hover").textContent = t; },
      onTool: (t) => {
        for (const b of $$("#ed-tools button")) b.setAttribute("aria-pressed", String(b.dataset.tool === t));
        $("#ed-tool-hint").textContent = (TOOLS.find(([k]) => k === t) || [, , ""])[2];
      },
    });
    const box = $("#ed-tools");
    for (const [key, label, tip] of TOOLS) box.append(el("button", { type: "button", class: "ghost small", "data-tool": key, title: tip, onclick: () => { ed.editor.setTool(key); ed.editor.canvas.focus(); } }, label));
    $("#ed-type").addEventListener("change", () => ed.editor.setFiberType($("#ed-type").value));
    $("#ed-undo").addEventListener("click", () => { if (ed.edits.length) { ed.redo.push(ed.edits.pop()); refreshEditor(); } });
    $("#ed-redo").addEventListener("click", () => { if (ed.redo.length) { ed.edits.push(ed.redo.pop()); refreshEditor(); } });
    $("#ed-reset").addEventListener("click", () => { if (ed.edits.length) { ed.redo = ed.edits.reverse(); ed.edits = []; refreshEditor(); } });
    $("#ed-done").addEventListener("click", () => dlg.close());
    $("#ed-save-otm").addEventListener("click", saveEditorSession);
    dlg.addEventListener("close", closeEditor);
    dlg.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); $(e.shiftKey ? "#ed-redo" : "#ed-undo").click(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); $("#ed-redo").click(); }
      else if (e.key === "Escape" && ed.editor.outline.length) { e.preventDefault(); ed.editor.cancelOutline(); }   // Esc cancels the outline, not the dialog
    });
    ed.editor.setTool("pan");
  }
  ed.editor.data = null;
  ed.editor.setFiberType($("#ed-type").value);
  dlg.showModal();
  try {
    await engine.started;
    await putFile(entry);
    await refreshEditor();
  } catch (e) { $("#ed-check").textContent = `The data could not be shown: ${e.message}`; }
}

async function refreshEditor() {
  if (ed.busy) { ed.pending = true; return; }
  ed.busy = true;
  try {
    do {
      ed.pending = false;
      const p = await engine.call("prepare", { name: ed.entry.name, opts: checkOptions(), edits: ed.edits });
      if (!p.ok) { $("#ed-check").textContent = `The data could not be shown: ${p.error}`; return; }
      ed.last = p;
      if (!ed.pending) { ed.editor.setData(p); showEditorPanel(p); }
    } while (ed.pending);
  } finally { ed.busy = false; }
}

function showEditorPanel(p) {
  const box = $("#ed-check");
  box.textContent = "";
  const q = p.check;
  if (q) {
    box.className = "ed-check " + (q.status === "ok" ? "ok" : q.status === "warning" ? "note" : "warn");
    box.append(el("b", { text: q.headline }));
    for (const i of [...q.fixed.map((x) => ({ ...x, removed: true })), ...q.issues]) {
      box.append(el("details", {},
        el("summary", { text: `${i.code} ${i.title}${i.removed ? " (removed)" : ""}` }),
        el("p", { text: i.message }),
        i.items.length ? el("ul", {}, ...i.items.slice(0, 12).map((t) => el("li", { text: t }))) : null,
        el("p", { class: "small", text: `Why it matters: ${i.why}` }),
        el("p", { class: "small", text: `What to do: ${i.advice}` })));
    }
    if (p.policy_note) box.append(el("p", { class: "small", text: p.policy_note + " (Settings → Data check)" }));
    if (p.flags.fibres.length || p.flags.caps.length) box.append(el("p", { class: "small", text: "Flagged items are drawn dashed in coral on the right." }));
  }
  const list = $("#ed-list");
  list.textContent = "";
  p.edits_text.forEach((t, i) => list.append(el("li", { class: p.applied[i] ? "" : "skipped", text: t })));
  list.scrollTop = list.scrollHeight;
  const n = p.edits_text.length;
  $("#ed-info").textContent = n ? `${n} correction${n > 1 ? "s" : ""}; indices, mesh and results are computed from the corrected data.`
    : "No corrections: the data are as in the file" + (p.check && p.check.fixed.length ? " (after the data check)." : ".");
  $("#ed-warn").hidden = !p.warnings.length;
  $("#ed-warn").textContent = p.warnings.join(" ");
  $("#ed-undo").disabled = !ed.edits.length;
  $("#ed-redo").disabled = !ed.redo.length;
  $("#ed-reset").disabled = !ed.edits.length;
}

function closeEditor() {
  const entry = ed.entry;
  if (!entry) return;
  const changed = JSON.stringify(ed.edits) !== JSON.stringify(ed.start);
  if (changed) {
    entry.edits = ed.edits.slice();
    if (state.results.has(entry.name)) toast(`Corrections of ${entry.name} changed: run it again to update its results.`, false, 6000);
  }
  ed.editor.cancelOutline();
  if (engine.ready) engine.call("drop", { name: entry.name }).catch(() => {});
  ed.entry = null;
  renderFiles();
}

async function saveEditorSession() {
  try {
    const bytes = await engine.call("sessionFromEdits", { name: ed.entry.name, opts: checkOptions(), edits: ed.edits });
    download(new Blob([bytes], { type: "application/zip" }), `${ed.entry.name}.otm`);
    toast(`Saved ${ed.entry.name}.otm: open it in the desktop app (File → Open session), or add it here with its .mat later.`, false, 7000);
  } catch (e) { toast(`Session not saved: ${e.message}`, true); }
}

// ============================================================================================
// HELP, SAMPLE, DIAGNOSTIC REPORT
// ============================================================================================
function logEvent(text, details = "") {
  state.log.push({ time: new Date().toISOString(), text: String(text), details: details ? String(details) : "" });
  if (state.log.length > 200) state.log.shift();
}

async function loadSample() {
  try {
    const resp = await fetch(new URL("./sample/OTM_sample_Dtect.mat", import.meta.url));
    if (!resp.ok) throw new Error(`download failed (${resp.status})`);
    const file = new File([await resp.blob()], "OTM_sample_Dtect.mat");
    const before = state.files.length;
    addFiles([file]);
    const entry = state.files[before];
    if (entry) { entry.given = "width"; entry.value = "440"; renderFiles(); }
    $("#help-dialog").close();
    document.getElementById("files").scrollIntoView();
    toast("Sample section added (440 × 330 µm). Press Run when it has been checked.");
  } catch (e) { toast(`The sample could not be loaded: ${e.message}`, true); }
}

function diagnosticReport() {
  const i = state.info || {};
  const L = [
    "OTM Web diagnostic report",
    "Send to: abdullah.alshammari@ku.edu.kw, subject: OTM2026",
    `Created: ${new Date().toString()}`,
    "",
    `Page: ${location.href}`,
    `otm_core ${i.otm_core || "?"} · Python ${i.python || "?"} · Pyodide ${i.pyodide || "?"} · numpy ${i.numpy || "?"} · mesher ${i.mesher || "?"}`,
    `Engine: ${engine.ready ? `ready (loaded in ${(i.load_seconds || 0).toFixed(1)} s)` : engine.failed ? `failed: ${engine.failed}` : "loading"}`,
    `Browser: ${navigator.userAgent}`,
    `Screen: ${screen.width} × ${screen.height} @ ${window.devicePixelRatio}x · memory: ${navigator.deviceMemory ? navigator.deviceMemory + " GB" : "?"} · cores: ${navigator.hardwareConcurrency || "?"}`,
    `WebGL 2: ${(() => { try { return document.createElement("canvas").getContext("webgl2") ? "yes" : "no"; } catch { return "no"; } })()}`,
    "",
    "Files:",
    ...state.files.map((f) => {
      const c = f.check, q = c && c.check;
      return `  ${f.mat.name} (${(f.mat.size / 1e6).toFixed(2)} MB${f.otm ? ", + .otm" : ""}): ${f.status}` +
        (c ? ` · ${c.image_cols}×${c.image_rows} px, ${c.n_capillaries} capillaries, ${c.n_fibres} fibres, ${c.format}` : "") +
        (q ? ` · data check ${q.status}${q.issues.length ? " (" + q.issues.map((x) => x.code).join(", ") + ")" : ""}` : "") +
        (editsOf(f).length ? ` · ${editsOf(f).length} correction(s)` : "") + (f.error ? ` · ${f.error}` : "") +
        (sizeOf(f) ? ` · ${sizeOf(f).map((v) => v.toFixed(1)).join(" × ")} µm` : "");
    }),
    "",
    "Results:",
    ...state.order.map((n) => {
      const r = state.results.get(n);
      return `  ${n}: ${r.ok ? "ok" : "ERRORS"} in ${(r.seconds || 0).toFixed(0)} s` +
        (r.timings ? ` (${Object.entries(r.timings).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", ")})` : "") +
        (r.errors && r.errors.length ? "\n    " + r.errors.map((e) => e.message).join("\n    ") : "");
    }),
    "",
    "Events of this visit:",
    ...(state.log.length ? state.log.map((e) => `  ${e.time}  ${e.text}${e.details ? "\n    " + e.details.split("\n").join("\n    ") : ""}`) : ["  (none)"]),
    "",
    "Settings:",
    JSON.stringify(state.info ? { ...settingsFromForm(), ...webOptions() } : {}, null, 2),
  ];
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  download(new Blob([L.join("\r\n") + "\r\n"], { type: "text/plain" }), `OTM_Web_diagnostic_report_${stamp}.txt`);
  toast("Report saved. Email it to abdullah.alshammari@ku.edu.kw with the subject OTM2026.", false, 7000);
}

function setupHelp() {
  const dlg = $("#help-dialog");
  $("#help-button").addEventListener("click", () => {
    $("#about-version").textContent = state.info ? `otm_core ${state.info.otm_core}` : "";
    dlg.showModal();
  });
  for (const b of $$("[data-close]", dlg)) b.addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  $("#help-sample").addEventListener("click", loadSample);
  $("#try-sample").addEventListener("click", loadSample);
  $("#help-report").addEventListener("click", diagnosticReport);
  window.addEventListener("error", (e) => logEvent(`page error: ${e.message}`, e.error && e.error.stack));
  window.addEventListener("unhandledrejection", (e) => logEvent(`unhandled: ${e.reason && e.reason.message ? e.reason.message : e.reason}`));
}

// ============================================================================================
// start
// ============================================================================================
function start() {
  setupFiles();
  wireSettings();
  setupHelp();
  $("#run-button").addEventListener("click", runAll);
  $("#stop-button").addEventListener("click", () => {
    state.stopRequested = true;
    $("#stop-button").hidden = true;
    toast("The run stops after the current sample.");
  });
  $("#zip-button").addEventListener("click", downloadZip);
  // highlight the step of the section in view
  const links = Object.fromEntries($$(".step-link").map((a) => [a.getAttribute("href").slice(1), a]));
  const seen = new Map();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) seen.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
    let best = null, ratio = 0;
    for (const [id, r] of seen) if (r > ratio) { best = id; ratio = r; }
    for (const [id, a] of Object.entries(links)) a.classList.toggle("current", id === best);
  }, { threshold: [0, 0.1, 0.25, 0.5, 0.75, 1], rootMargin: "-60px 0px -30% 0px" });
  for (const id of Object.keys(links)) io.observe(document.getElementById(id));
  window.addEventListener("beforeunload", (e) => {
    if (state.running || state.results.size) { e.preventDefault(); e.returnValue = ""; }
  });
  applyFigureOptions();
  updateRunState();
  setEngine("Starting the analysis engine…", "loading");
  engine.started.then((info) => {
    state.info = info;
    setupSettings(info);
    setEngine(`Ready · loaded in ${info.load_seconds.toFixed(0)} s`, "ready");
    $("#versions").textContent = `otm_core ${info.otm_core}, Python ${info.python}, Pyodide ${info.pyodide}, ${info.mesher} mesher`;
    updateRunState();
  }).catch((e) => {
    setEngine(`Could not start: ${e.message}`, "failed");
    $("#run-hint").textContent = "The analysis engine could not start. Use an up-to-date Chrome, Edge or Firefox on a computer (not a phone).";
    console.error(e.details || e);
  });
}

if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
  document.addEventListener("DOMContentLoaded", () => setEngine("This browser cannot run OTM Web; use Chrome, Edge or Firefox.", "failed"));
} else start();

// for the browser test
window.__otm = { state, engine, storeResult, selectSample, download: pyDownload, ed, openEditor, get view() { return currentView; } };
