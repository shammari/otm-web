// OTM Web page: files, settings, run, results. The analysis itself runs in worker.js.

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
  files: [],            // {name, mat: File, otm: File|null, check, status, error, width, height, roi, corrections}
  orphanOtm: new Map(), // stem (lower case) -> File, .otm added before its .mat
  results: new Map(),   // name -> result of otm_web.run
  order: [],            // result names in run order
  selected: null,
  running: false,
  stopRequested: false,
  extras: {},           // settings from a loaded JSON the form does not show (mesh, flux, retouch ...)
  figureCache: new Map(),
};

const engine = new Engine();
engine.onStatus = (text, kind) => setEngine(text, kind);
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
                      check: null, status: "waiting", error: null, width: "", height: "", roi: null, corrections: null };
      const orphan = state.orphanOtm.get(entry.stem.toLowerCase());
      if (orphan) { entry.otm = orphan; state.orphanOtm.delete(entry.stem.toLowerCase()); }
      state.files.push(entry);
      added.push(entry);
    } else if (ext === ".otm") {
      const target = state.files.find((f) => f.stem.toLowerCase() === stemOf(file.name).toLowerCase());
      if (target) { target.otm = file; target.corrections = null; if (!added.includes(target)) added.push(target); }
      else state.orphanOtm.set(stemOf(file.name).toLowerCase(), file);
    } else ignored++;
  }
  if (ignored) toast(`${ignored} file(s) left out: only .mat and .otm files can be analysed.`, true);
  if (state.orphanOtm.size) toast(`Session file(s) waiting for a .mat with the same name: ${[...state.orphanOtm.keys()].join(", ")}`);
  renderFiles();
  for (const f of added) checkFile(f);
}

async function putFile(entry) {
  await engine.call("put", { name: entry.name, ext: ".mat", buffer: await entry.mat.arrayBuffer() });
  if (entry.otm) await engine.call("put", { name: entry.name, ext: ".otm", buffer: await entry.otm.arrayBuffer() });
}

async function checkFile(entry) {
  entry.status = "checking";
  renderFiles();
  try {
    await engine.started;
    await putFile(entry);
    const res = await engine.call("inspect", { name: entry.name });
    if (res.ok) {
      entry.check = res;
      entry.status = "ok";
      entry.error = null;
      if (entry.otm) {
        const c = await engine.call("corrections", { name: entry.name });
        entry.corrections = c.ok ? c.n_edits : null;
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

function fileSize(entry) {
  if (num(entry.width)) return { width_um: num(entry.width), height_um: null };
  if (num(entry.height)) return { width_um: null, height_um: num(entry.height) };
  const v = num($("#size-value").value);
  if (v && v > 0) return $("#size-kind").value === "width" ? { width_um: v, height_um: null } : { width_um: null, height_um: v };
  return null;
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
        type: "number", min: "0", step: "any", value: f[key], placeholder: "–", "aria-label": `${key} of ${f.name} in µm`,
        oninput: (e) => {
          f[key] = e.target.value;
          const other = key === "width" ? "height" : "width";
          if (e.target.value && f[other] !== "") { f[other] = ""; inputs[other].value = ""; }
          updateRunState();
        },
      });
    }
    let check;
    if (f.status === "checking" || f.status === "waiting") check = el("span", { class: "hint small", text: f.status === "waiting" ? "waiting…" : "checking…" });
    else if (f.status === "ok") {
      const bits = [el("span", { class: "ok-text", text: "✓ readable" })];
      if (f.corrections !== null && f.corrections !== undefined) bits.push(el("span", { class: "hint small", text: ` + ${f.corrections} correction(s)` }));
      if (f.error) bits.push(el("div", { class: "warn-text small", text: f.error }));
      if (c.notes && c.notes.length) bits.push(el("div", { class: "hint small", text: c.notes.join("; ") }));
      check = el("span", {}, ...bits);
    } else check = el("span", { class: "warn-text", text: `✗ ${f.error || "cannot be read"} (left out)` });
    tbody.append(el("tr", { "data-name": f.name },
      el("td", {}, el("b", { text: f.name }), f.otm ? el("span", { class: "hint small", text: " + .otm" }) : null),
      el("td", { class: "num", text: c ? `${c.image_cols} × ${c.image_rows}` : "" }),
      el("td", { class: "num", text: c ? fmtInt(c.n_capillaries) : "" }),
      el("td", { class: "num", text: c ? fmtInt(c.n_fibres) : "" }),
      el("td", { class: "nowrap", text: types }),
      el("td", { class: "num" }, inputs.width),
      el("td", { class: "num" }, inputs.height),
      el("td", {}, check),
      el("td", {}, el("button", { class: "remove", type: "button", title: `Remove ${f.name}`, "aria-label": `Remove ${f.name}`,
        onclick: () => removeFile(f) }, "×")),
    ));
  }
  $("#file-table-wrap").hidden = $("#file-actions").hidden = state.files.length === 0;
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
        f.width = ch.width_um ?? "";
        f.height = ch.width_um ? "" : (ch.height_um ?? "");
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

function buildParamGrid() {
  const box = $("#params");
  box.textContent = "";
  for (const f of paramFields()) {
    box.append(el("label", { class: "param" },
      el("span", { class: "what" }, `${f.label} (${f.symbol})`, el("small", { text: f.unit })),
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
  const size = num($("#size-value").value);
  const s = {
    ...state.extras,
    tissue: radio("tissue"),
    use_fibre_types: $("#use-types").checked,
    width_um: $("#size-kind").value === "width" && size ? size : null,
    height_um: $("#size-kind").value === "height" && size ? size : null,
    roi_um: radio("roi") === "edges" ? ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1"].map((id) => num($(id).value)) : null,
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
  };
  return s;
}

function settingsForFile(entry, base) {
  const s = { ...base, ...(fileSize(entry) || { width_um: null, height_um: null }) };
  if (entry.roi) s.roi_um = entry.roi;
  return s;
}

function formToDefaults() {
  const d = state.info.settings;
  state.extras = {};
  fillForm({ ...d, parameters: null });
  $("#size-value").value = "";
  $("#size-kind").value = "width";
}

function fillForm(d) {
  setRadio("tissue", d.tissue || "skeletal");
  $("#use-types").checked = d.use_fibre_types !== false;
  if (d.width_um) { $("#size-kind").value = "width"; $("#size-value").value = d.width_um; }
  else if (d.height_um) { $("#size-kind").value = "height"; $("#size-value").value = d.height_um; }
  else $("#size-value").value = "";
  if (d.roi_um) {
    setRadio("roi", "edges");
    ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1"].forEach((id, i) => { $(id).value = d.roi_um[i]; });
  } else setRadio("roi", "default");
  $("#roi-edges").hidden = radio("roi") !== "edges";
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
  validate();
}

const FORM_KEYS = new Set(["tissue", "use_fibre_types", "width_um", "height_um", "roi_um", "parameters", "exercise",
  "non_uniform", "differential_extraction", "steps", "figures", "dpi", "colormap", "figure_layout", "figure_width",
  "figure_formats", "po2_range_mmHg"]);

async function loadSettingsFile(file) {
  let d;
  try { d = JSON.parse(await file.text()); } catch { return toast("Settings not loaded: not a JSON file.", true); }
  if (!d || typeof d !== "object" || Array.isArray(d)) return toast("Settings not loaded: not an OTM settings file.", true);
  await engine.started;
  const notes = [];
  if (d.parameter_file) { notes.push("the .dat file it names cannot be opened from the browser; load it under Biophysical parameters"); delete d.parameter_file; }
  const problem = await engine.call("check", { settings: d });
  if (problem) return toast(`Settings not loaded: ${problem}`, true, 7000);
  state.extras = Object.fromEntries(Object.entries(d).filter(([k]) => !FORM_KEYS.has(k)));
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
  if (state.info && Object.values(readParams()).some((v) => v === null || Number.isNaN(v) || v < 0)) msgs.push("every biophysical parameter needs a number of 0 or more");
  if ($("#figures").checked && !$$("#formats input:checked").length) msgs.push("choose at least one figure format");
  if (radio("range") === "fixed") {
    const lo = num($("#range-lo").value), hi = num($("#range-hi").value);
    if (lo === null || hi === null || !(hi > lo)) msgs.push("the fixed PO₂ range needs a lowest value below the highest");
  }
  const de = num($("#diff-extraction").value);
  if (radio("tissue") === "skeletal" && (de === null || de < 0.1 || de > 10)) msgs.push("differential extraction must be between 0.1 and 10");
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
  for (const r of $$('input[name="tissue"]')) r.addEventListener("change", () => { applyTissue(); validate(); });
  for (const r of $$('input[name="roi"]')) r.addEventListener("change", () => { $("#roi-edges").hidden = radio("roi") !== "edges"; validate(); });
  for (const r of $$('input[name="layout"]')) r.addEventListener("change", () => {
    const dpi = Number($("#dpi").value);
    if (radio("layout") === "legacy" && dpi === 600) $("#dpi").value = "300";
    if (radio("layout") === "publication" && dpi === 300) $("#dpi").value = "600";
    applyFigureOptions();
  });
  for (const r of $$('input[name="range"]')) r.addEventListener("change", () => { applyFigureOptions(); validate(); });
  $("#figures").addEventListener("change", () => { applyFigureOptions(); validate(); });
  for (const id of ["#roi-x0", "#roi-x1", "#roi-y0", "#roi-y1", "#range-lo", "#range-hi", "#diff-extraction", "#size-value"]) $(id).addEventListener("input", validate);
  $("#size-kind").addEventListener("change", validate);
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
  else if (missingSize.length) hint = `Give the tissue size in Settings (or in the file table) for: ${missingSize.map((f) => f.name).join(", ")}.`;
  else if (!$("#settings-error").hidden) hint = "Fix the settings first.";
  else hint = `${files.length} file(s) ready.` + (state.files.length > files.length ? ` ${state.files.length - files.length} unreadable file(s) will be left out.` : "");
  if (state.running) hint = "Running… you can keep using this tab; leaving the page stops the run.";
  $("#run-hint").textContent = hint;
  btn.disabled = state.running || !engine.ready || !files.length || missingSize.length > 0 || checking || !$("#settings-error").hidden;
  btn.textContent = files.length > 1 ? `Run ${files.length} files` : "Run";
  $("#stop-button").hidden = !state.running;
  $("#zip-button").disabled = state.running || state.results.size === 0;
  $$(".step-link")[0].classList.toggle("done", files.length > 0);
  $$(".step-link")[2].classList.toggle("done", state.results.size > 0);
}

const STEP_LABELS = { load: "Reading the file", geometry: "Tissue size and region", indices: "Supply indices",
  mesh: "Meshing", po2: "Solving PO₂", flux: "Flux lines", export: "Writing tables and figures", done: "Done",
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
  for (const f of files) {
    if (state.stopRequested) {
      const row = progressRow(f.name);
      $(".pstep", row).textContent = "Not run (stopped)";
      continue;
    }
    const row = progressRow(f.name);
    row.className = "";
    const started = performance.now();
    runClock = setInterval(() => { $(".ptime", row).textContent = `${((performance.now() - started) / 1000).toFixed(0)} s`; }, 500);
    setOverviewRow(f.name, { name: f.name, running: true });
    try {
      await putFile(f);
      const res = await engine.call("run", { name: f.name, settings: settingsForFile(f, base) });
      res.file = f.mat.name;
      storeResult(res);
      row.classList.add(res.ok ? "done" : "failed");
      $(".pstep", row).textContent = res.ok ? `Done · ${res.files.length} files written` : res.errors.map((e) => e.message).join(" · ");
      $(".bar i", row).style.width = "100%";
    } catch (e) {
      row.classList.add("failed");
      $(".pstep", row).textContent = `Failed: ${e.message}`;
      storeResult({ name: f.name, ok: false, errors: [{ step: "run", message: e.message }], files: [], summary: null, seconds: (performance.now() - started) / 1000 });
      if (engine.failed) { clearInterval(runClock); break; }
    } finally {
      clearInterval(runClock);
      $(".ptime", row).textContent = `${((performance.now() - started) / 1000).toFixed(0)} s`;
      engine.call("drop", { name: f.name }).catch(() => {});
    }
  }
  state.running = false;
  updateRunState();
  const n = files.length;
  const failed = files.filter((f) => state.results.get(f.name) && !state.results.get(f.name).ok).length;
  toast(`Run finished in ${((performance.now() - t0) / 1000).toFixed(0)} s: ${n - failed} of ${n} sample(s) without errors.`, failed > 0, 6000);
}

// ============================================================================================
// 4 RESULTS
// ============================================================================================
function storeResult(res) {
  if (!state.results.has(res.name)) state.order.push(res.name);
  state.results.set(res.name, res);
  for (const k of [...state.figureCache.keys()]) if (k.startsWith(res.name + "/")) { URL.revokeObjectURL(state.figureCache.get(k)); state.figureCache.delete(k); }
  setOverviewRow(res.name, res);
  if (!state.selected || state.selected === res.name) selectSample(res.name);
  $("#results-hint").textContent = "Click a sample to see its tables and figures. The ZIP holds, per sample, the INDICES and PO2 folders the desktop app writes, plus the batch summary spreadsheet.";
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
  const meta = [];
  if (res.size_um) meta.push(`${fmt(res.size_um[0], 0)} × ${fmt(res.size_um[1], 0)} µm`);
  if (res.roi_um) meta.push(`ROI x ${fmt(res.roi_um[0], 0)}–${fmt(res.roi_um[1], 0)}, y ${fmt(res.roi_um[2], 0)}–${fmt(res.roi_um[3], 0)} µm`);
  if (res.mesh) meta.push(`mesh ${fmtInt(res.mesh.nodes)} nodes (${res.mesh.mesher})`);
  if (res.solve) meta.push(`${res.solve.converged ? "converged" : "NOT converged"} in ${res.solve.iterations} Newton iteration(s), PO₂ ${fmt(res.solve.po2_min_mmHg)}–${fmt(res.solve.po2_max_mmHg)} mmHg`);
  if (res.flux) meta.push(`flux lines from ${res.flux.capillaries_with_lines} of ${res.flux.seed_capillaries} ROI capillaries`);
  if (res.corrections) meta.push(`${res.corrections} saved correction(s) applied`);
  if (res.seconds) meta.push(`${fmt(res.seconds, 0)} s`);
  $(".sample-meta", node).textContent = meta.join(" · ");
  const errs = $(".errors", node);
  for (const e of res.errors || []) errs.append(el("p", { text: e.message }));
  for (const w of res.corrections_skipped || []) errs.append(el("p", { text: `Correction skipped: ${w}` }));

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
  const files = res.files || [];
  const pngs = files.filter((p) => p.toLowerCase().endsWith(".png"));
  const gallery = $(".gallery", node);
  if (!pngs.length) $(".block-figures", node).replaceChildren(el("h4", { text: "Figures" }), el("p", { class: "empty", text: "No figures in this run." }));
  for (const p of pngs) gallery.append(figureCard(p, files));
  const ul = $(".files-list ul", node);
  for (const p of files) ul.append(el("li", {}, el("a", { href: "#", onclick: (e) => { e.preventDefault(); saveOutput(p); }, text: p.split("/").slice(1).join("/") })));
  if (!files.length) $(".files-list", node).hidden = true;
  view.append(node);
  loadThumbnails(gallery);
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

async function figureURL(path) {
  if (state.figureCache.has(path)) return state.figureCache.get(path);
  const bytes = await engine.call("file", { path });
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
  state.figureCache.set(path, url);
  return url;
}

async function loadThumbnails(gallery) {
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
    const bytes = await engine.call("file", { path });
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
// start
// ============================================================================================
function start() {
  setupFiles();
  wireSettings();
  $("#run-button").addEventListener("click", runAll);
  $("#stop-button").addEventListener("click", () => {
    state.stopRequested = true;
    $("#stop-button").hidden = true;
    toast("The run stops after the current sample.");
  });
  $("#zip-button").addEventListener("click", downloadZip);
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
window.__otm = { state, engine, storeResult, selectSample };
