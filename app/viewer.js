// OTM Web tissue view: the desktop's 3-D view as an interactive 2-D map.
//
// The PO2 (or VO2) field is drawn with WebGL from the run's own mesh, the colour map is a texture,
// so changing the colour range or map is instant. Fibre outlines, capillaries, flux lines, Voronoi
// domains and the ROI are drawn on a 2-D canvas on top. Wheel to zoom, drag to pan, double-click
// to fit. Data come from otm_web.write_view (a binary file per sample, µm, origin bottom-left).

import { COLORMAPS } from "./colormaps.js";

// ---------------------------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------------------------
export function parseView(buffer) {
  const u8 = new Uint8Array(buffer);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "OTMV") throw new Error("not an OTM view file");
  const n = new DataView(buffer).getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const base = 8 + n;
  const arrays = {};
  const T = { f32: Float32Array, u32: Uint32Array, i32: Int32Array };
  for (const [k, [dt, off, len]] of Object.entries(head.arrays || {})) arrays[k] = new T[dt](buffer, base + off, len);
  return { head, arrays };
}

export function lut(name) {
  const hex = COLORMAPS[name] || COLORMAPS.turbo;
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(2 * i, 2), 16);
  return out;
}

export function lutGradient(name, steps = 16) {
  const c = lut(name);
  const stops = [];
  for (let k = 0; k <= steps; k++) {
    const i = Math.round((k / steps) * 255) * 3;
    stops.push(`rgb(${c[i]},${c[i + 1]},${c[i + 2]}) ${((k / steps) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export const TYPE_COLOURS = { 1: [90, 90, 90], 21: [0, 200, 210], 22: [220, 60, 220], 0: [60, 180, 60] };
export const TYPE_NAMES = { 1: "I", 21: "IIa", 22: "IIb", 0: "unknown" };

// fraction of values below lo / above hi (as the desktop's colour-range check; negatives drawn as 0)
export function coverage(values, lo, hi) {
  let below = 0, above = 0, n = 0;
  const tol = 1e-9 * Math.max(1, Math.abs(hi));
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const w = v < 0 ? 0 : v;
    n++;
    if (w < lo - tol) below++;
    else if (w > hi + tol) above++;
  }
  return n ? { below: below / n, above: above / n, outside: (below + above) / n } : null;
}

const niceStep = (x) => {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / p;
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * p;
};
const fmtNum = (v, d = 2) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : v.toFixed(d));

// ---------------------------------------------------------------------------------------------
// WebGL
// ---------------------------------------------------------------------------------------------
const VS = `#version 300 es
in vec2 a_pos;
in float a_val;
uniform vec2 u_center;
uniform vec2 u_scale;
out float v_val;
void main() {
  gl_Position = vec4((a_pos - u_center) * u_scale, 0.0, 1.0);
  v_val = a_val;
}`;
const FS = `#version 300 es
precision highp float;
in float v_val;
uniform sampler2D u_lut;
uniform vec2 u_range;
uniform vec4 u_solid;
out vec4 o;
void main() {
  if (u_solid.a > 0.0) { o = u_solid; return; }
  float t = clamp((v_val - u_range.x) / max(u_range.y - u_range.x, 1e-12), 0.0, 1.0);
  o = texture(u_lut, vec2(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5));
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

// ---------------------------------------------------------------------------------------------
// the view
// ---------------------------------------------------------------------------------------------
export class TissueView {
  /**
   * @param {HTMLElement} root  element to fill
   * @param {object} opts  {colormap, range: [lo, hi] | null, units: "um" | "ndim", onRange(range)}
   */
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = opts;
    this.colormap = opts.colormap || "turbo";
    this.fixedRange = opts.range || null;          // PO2 range in mmHg (null: automatic)
    this.units = opts.units || "um";
    this.layers = { map: "po2", fibres: true, capillaries: true, flux: true, voronoi: false, roi: true, mesh: false };
    this.data = null;
    this.view = null;                                // {cx, cy, s}: µm at the centre, css px per µm
    this._raf = 0;
    this._build();
  }

  // ---- DOM ----------------------------------------------------------------------------------
  _build() {
    const r = this.root;
    r.classList.add("tv");
    r.innerHTML = `
      <div class="tv-bar">
        <label>Show <select class="tv-map" aria-label="Map">
          <option value="po2">PO₂ map</option><option value="vo2">O₂ consumption map</option>
          <option value="types">Fibre types</option><option value="none">Outlines only</option></select></label>
        <span class="tv-layers">
          <label><input type="checkbox" data-layer="fibres"> Fibres</label>
          <label><input type="checkbox" data-layer="capillaries"> Capillaries</label>
          <label><input type="checkbox" data-layer="flux"> Flux lines</label>
          <label><input type="checkbox" data-layer="voronoi"> Capillary domains</label>
          <label><input type="checkbox" data-layer="roi"> ROI</label>
          <label><input type="checkbox" data-layer="mesh"> Mesh</label>
        </span>
      </div>
      <div class="tv-bar tv-bar2">
        <label>Colours <select class="tv-cmap" aria-label="Colour map"><option>turbo</option><option>viridis</option><option>plasma</option><option>jet</option></select></label>
        <span class="tv-range">
          <span class="label">PO₂ range</span>
          <span class="seg" role="radiogroup"><label><input type="radio" name="tvr" value="auto"><span>Automatic</span></label><label><input type="radio" name="tvr" value="fixed"><span>Fixed</span></label></span>
          <span class="tv-fixed"><input type="number" class="short tv-lo" step="any" min="0" aria-label="Lowest PO2"> to <input type="number" class="short tv-hi" step="any" min="0" aria-label="Highest PO2"> mmHg</span>
        </span>
        <label>Units <select class="tv-units" aria-label="Units"><option value="um">µm</option><option value="ndim">model units</option></select></label>
        <span class="tv-tools">
          <button type="button" class="ghost small tv-zin" title="Zoom in (+)" aria-label="Zoom in">+</button>
          <button type="button" class="ghost small tv-zout" title="Zoom out (−)" aria-label="Zoom out">−</button>
          <button type="button" class="ghost small tv-fit" title="Fit the tissue (double-click, or 0)">Fit</button>
          <button type="button" class="ghost small tv-png" title="Save this view as a PNG image">Save PNG</button>
        </span>
      </div>
      <div class="tv-stage" tabindex="0" aria-label="Tissue view: drag to pan, wheel to zoom, double-click to fit">
        <canvas class="tv-gl"></canvas><canvas class="tv-ov"></canvas>
        <div class="tv-msg" hidden></div>
      </div>
      <div class="tv-foot">
        <div class="tv-cbar"><span class="tv-clo"></span><i></i><span class="tv-chi"></span><span class="tv-cunit"></span></div>
        <div class="tv-legend" hidden></div>
        <div class="tv-readout" aria-live="off"></div>
      </div>
      <p class="tv-cover warn-note" hidden></p>`;
    const q = (s) => r.querySelector(s);
    this.el = {
      map: q(".tv-map"), cmap: q(".tv-cmap"), units: q(".tv-units"), stage: q(".tv-stage"), gl: q(".tv-gl"), ov: q(".tv-ov"),
      msg: q(".tv-msg"), cbar: q(".tv-cbar"), clo: q(".tv-clo"), chi: q(".tv-chi"), cunit: q(".tv-cunit"), grad: q(".tv-cbar i"),
      legend: q(".tv-legend"), readout: q(".tv-readout"), cover: q(".tv-cover"), range: q(".tv-range"), fixed: q(".tv-fixed"),
      lo: q(".tv-lo"), hi: q(".tv-hi"),
    };
    // a unique radio-group name per view
    const gname = "tvr" + Math.random().toString(36).slice(2, 8);
    for (const i of r.querySelectorAll('input[name="tvr"]')) i.name = gname;
    this._rangeRadios = [...r.querySelectorAll(`input[name="${gname}"]`)];

    this.el.cmap.value = this.colormap;
    this.el.units.value = this.units;
    this.el.map.addEventListener("change", () => { this.layers.map = this.el.map.value; this._afterChange(); });
    this.el.cmap.addEventListener("change", () => { this.colormap = this.el.cmap.value; this._lutDirty = true; this._afterChange(); });
    this.el.units.addEventListener("change", () => { this.units = this.el.units.value; this.render(); });
    for (const cb of r.querySelectorAll("[data-layer]")) {
      cb.checked = !!this.layers[cb.dataset.layer];
      cb.addEventListener("change", () => { this.layers[cb.dataset.layer] = cb.checked; this.render(); });
    }
    for (const rb of this._rangeRadios) rb.addEventListener("change", () => this._rangeFromForm());
    for (const inp of [this.el.lo, this.el.hi]) inp.addEventListener("change", () => this._rangeFromForm());
    q(".tv-zin").addEventListener("click", () => this.zoom(1.5));
    q(".tv-zout").addEventListener("click", () => this.zoom(1 / 1.5));
    q(".tv-fit").addEventListener("click", () => this.fit());
    q(".tv-png").addEventListener("click", () => this.savePNG());
    this._wireMouse();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this.el.stage);
    this._initGL();
  }

  destroy() {
    this._ro.disconnect();
    cancelAnimationFrame(this._raf);
    const gl = this.gl;
    if (gl) { const ext = gl.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext(); }
  }

  _message(text) {
    this.el.msg.hidden = !text;
    this.el.msg.textContent = text || "";
  }

  _initGL() {
    const gl = this.el.gl.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true, premultipliedAlpha: false });
    this.gl = gl;
    if (!gl) { this._message("This browser cannot draw the map (no WebGL 2). Outlines are still shown; the figures below have the maps."); return; }
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.prog = p;
    this.loc = {
      pos: gl.getAttribLocation(p, "a_pos"), val: gl.getAttribLocation(p, "a_val"),
      center: gl.getUniformLocation(p, "u_center"), scale: gl.getUniformLocation(p, "u_scale"),
      lut: gl.getUniformLocation(p, "u_lut"), range: gl.getUniformLocation(p, "u_range"), solid: gl.getUniformLocation(p, "u_solid"),
    };
    this.tex = gl.createTexture();
    this._lutDirty = true;
    this.el.gl.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this._message("The map was lost by the graphics driver; reload the page to draw it again."); });
  }

  _uploadLUT() {
    const gl = this.gl;
    if (!gl || !this._lutDirty) return;
    const c = lut(this.colormap);
    const rgba = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) { rgba[4 * i] = c[3 * i]; rgba[4 * i + 1] = c[3 * i + 1]; rgba[4 * i + 2] = c[3 * i + 2]; rgba[4 * i + 3] = 255; }
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._lutDirty = false;
  }

  _buffer(data, target = null) {
    const gl = this.gl;
    const b = gl.createBuffer();
    gl.bindBuffer(target || gl.ARRAY_BUFFER, b);
    gl.bufferData(target || gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  // ---- data ---------------------------------------------------------------------------------
  /** @param {{head, arrays}} view  from parseView */
  setData(view) {
    this.data = view;
    const { head, arrays: a } = view;
    this._fibres = this._fibreRings(a);
    this._grid = null;
    this._gpu = {};
    const hasMesh = !!(a.mesh_pts && a.mesh_tri);
    const hasPo2 = !!(hasMesh && a.po2);
    const hasVo2 = !!(hasMesh && a.vo2_tri);
    // choices that exist for this sample
    for (const o of this.el.map.options) o.disabled = (o.value === "po2" && !hasPo2) || (o.value === "vo2" && !hasVo2) || (o.value === "types" && !head.n_fibres);
    if (this.el.map.selectedOptions[0] && this.el.map.selectedOptions[0].disabled) this.layers.map = hasPo2 ? "po2" : head.n_fibres ? "types" : "none";
    if (!hasPo2 && this.layers.map === "po2") this.layers.map = head.n_fibres ? "types" : "none";
    this.el.map.value = this.layers.map;
    for (const cb of this.root.querySelectorAll("[data-layer]")) {
      const k = cb.dataset.layer;
      const has = { fibres: head.n_fibres > 0, capillaries: !!a.caps, flux: !!a.flux_pts, voronoi: !!a.vor_pts, roi: !!head.roi_um, mesh: hasMesh }[k];
      cb.closest("label").hidden = !has;
    }
    const gl = this.gl;
    if (gl && hasMesh) {
      this._gpu.pos = this._buffer(a.mesh_pts);
      this._gpu.tri = this._buffer(a.mesh_tri, gl.ELEMENT_ARRAY_BUFFER);
      if (hasPo2) this._gpu.po2 = this._buffer(a.po2);
    }
    this._autoRange = head.po2_range || null;
    this._showRangeForm();
    this.fit(false);
    this._afterChange();
  }

  _fibreRings(a) {
    if (!a.fib_pts || !a.fib_off) return [];
    const out = [];
    for (let k = 0; k + 1 < a.fib_off.length; k++) {
      const s = a.fib_off[k], e = a.fib_off[k + 1];
      const pts = a.fib_pts.subarray(2 * s, 2 * e);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, area = 0;
      for (let i = 0, n = e - s; i < n; i++) {
        const x = pts[2 * i], y = pts[2 * i + 1];
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        const j = (i + 1) % n;
        area += x * pts[2 * j + 1] - pts[2 * j] * y;
      }
      const owner = a.fib_owner ? a.fib_owner[k] : k;
      out.push({ pts, owner, type: a.fib_type ? a.fib_type[owner] : 0, box: [x0, x1, y0, y1], area: Math.abs(area) / 2 });
    }
    return out;
  }

  _vo2Buffers() {
    if (this._gpu.vo2pos) return true;
    const a = this.data.arrays;
    if (!a.vo2_tri || !this.gl) return false;
    const nt = a.mesh_tri.length / 3;
    const pos = new Float32Array(nt * 6), val = new Float32Array(nt * 3);
    for (let t = 0; t < nt; t++) {
      for (let c = 0; c < 3; c++) {
        const i = a.mesh_tri[3 * t + c];
        pos[6 * t + 2 * c] = a.mesh_pts[2 * i];
        pos[6 * t + 2 * c + 1] = a.mesh_pts[2 * i + 1];
        val[3 * t + c] = a.vo2_tri[t];
      }
    }
    this._gpu.vo2pos = this._buffer(pos);
    this._gpu.vo2val = this._buffer(val);
    this._gpu.vo2n = nt * 3;
    const r = this.data.head.vo2_range || [0, 1];
    this._vo2Range = [r[0], r[1] > r[0] ? r[1] : r[0] + 1];
    return true;
  }

  _edgeBuffer() {
    if (this._gpu.edges) return true;
    const a = this.data.arrays;
    if (!a.mesh_tri || !this.gl) return false;
    const n = a.mesh_pts.length / 2, nt = a.mesh_tri.length / 3;
    const keys = new Float64Array(nt * 3);
    for (let t = 0; t < nt; t++) {
      for (let c = 0; c < 3; c++) {
        const i = a.mesh_tri[3 * t + c], j = a.mesh_tri[3 * t + ((c + 1) % 3)];
        keys[3 * t + c] = i < j ? i * n + j : j * n + i;
      }
    }
    keys.sort();
    const idx = [];
    let last = -1;
    for (const k of keys) { if (k !== last) { idx.push(Math.floor(k / n), k % n); last = k; } }
    this._gpu.edges = this._buffer(new Uint32Array(idx), this.gl.ELEMENT_ARRAY_BUFFER);
    this._gpu.edgeN = idx.length;
    return true;
  }

  // ---- colour range ---------------------------------------------------------------------------
  setRange(range) {                       // PO2 range [lo, hi] mmHg, or null for automatic
    this.fixedRange = range ? [Number(range[0]), Number(range[1])] : null;
    this._showRangeForm();
    this._afterChange();
  }

  setColormap(name) {
    this.colormap = name;
    this.el.cmap.value = name;
    this._lutDirty = true;
    this._afterChange();
  }

  _showRangeForm() {
    const fixed = !!this.fixedRange;
    this._rangeRadios.forEach((rb) => { rb.checked = rb.value === (fixed ? "fixed" : "auto"); });
    this.el.fixed.hidden = !fixed;
    const r = this.fixedRange || this._niceAuto();
    if (r) { this.el.lo.value = String(+r[0].toFixed(2)); this.el.hi.value = String(+r[1].toFixed(2)); }
  }

  _niceAuto() {
    const r = this._autoRange;
    if (!r) return null;
    const a = Math.floor(Math.max(0, r[0])), b = Math.ceil(r[1]);
    return [a, b > a ? b : a + 1];
  }

  _rangeFromForm() {
    const fixed = this._rangeRadios.find((r) => r.checked)?.value === "fixed";
    if (!fixed) { this.fixedRange = null; this.el.fixed.hidden = true; }
    else {
      this.el.fixed.hidden = false;
      const lo = Number(this.el.lo.value), hi = Number(this.el.hi.value);
      if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo >= 0 && hi > lo)) {
        this.el.cover.hidden = false;
        this.el.cover.textContent = "The fixed range needs 0 ≤ lowest < highest.";
        return;
      }
      this.fixedRange = [lo, hi];
    }
    if (this.opts.onRange) this.opts.onRange(this.fixedRange);
    this._afterChange();
  }

  _currentRange() {
    const m = this.layers.map;
    if (m === "po2") {
      if (this.fixedRange) return this.fixedRange;
      const r = this._autoRange || [0, 1];
      return [Math.max(0, r[0]), r[1] > r[0] ? r[1] : r[0] + 1];
    }
    if (m === "vo2") return this._vo2Range || [0, 1];
    return null;
  }

  _afterChange() {
    const m = this.layers.map;
    if (m === "vo2") this._vo2Buffers();
    const r = this._currentRange();
    const mapOn = r && (m === "po2" || m === "vo2") && this.data;
    this.el.cbar.hidden = !mapOn;
    this.el.range.hidden = m !== "po2";
    if (mapOn) {
      this.el.grad.style.background = lutGradient(this.colormap);
      const d = m === "po2" ? 1 : 2;
      this.el.clo.textContent = fmtNum(r[0], d);
      this.el.chi.textContent = fmtNum(r[1], d);
      this.el.cunit.textContent = m === "po2" ? "PO₂ (mmHg)" : "O₂ consumption (ml O₂/100 ml/min)";
    }
    // legend for fibre types
    const types = m === "types" || m === "none";
    this.el.legend.hidden = !(m === "types" && this.data && this.data.head.n_fibres);
    if (!this.el.legend.hidden) {
      const present = new Set(this._fibres.map((f) => f.type));
      this.el.legend.innerHTML = [1, 21, 22, 0].filter((t) => present.has(t))
        .map((t) => `<span><i style="background:rgb(${TYPE_COLOURS[t].join(",")})"></i>Type ${TYPE_NAMES[t]}</span>`).join("");
    }
    // how much of the sample lies outside a fixed PO2 range (as the desktop's status bar)
    this.el.cover.hidden = true;
    if (m === "po2" && this.fixedRange && this.data && this.data.arrays.po2) {
      const cov = coverage(this.data.arrays.po2, this.fixedRange[0], this.fixedRange[1]);
      if (cov && cov.outside > 0) {
        this.el.cover.hidden = false;
        const parts = [];
        if (cov.below) parts.push(`${(100 * cov.below).toFixed(1)} % below`);
        if (cov.above) parts.push(`${(100 * cov.above).toFixed(1)} % above`);
        this.el.cover.textContent = `Colour scale fixed at ${this.fixedRange[0]}–${this.fixedRange[1]} mmHg: ${(100 * cov.outside).toFixed(1)} % of this sample lies outside it (${parts.join(", ")}); those parts are drawn in the end colours.`;
      }
    }
    void types;
    this.render();
  }

  // ---- view transform -----------------------------------------------------------------------
  _size() {
    const r = this.el.stage.getBoundingClientRect();
    return { W: Math.max(1, r.width), H: Math.max(1, r.height) };
  }

  fit(redraw = true) {
    if (!this.data) return;
    const [w, h] = this.data.head.size_um;
    const { W, H } = this._size();
    const s = 0.96 * Math.min(W / w, H / h);
    this.view = { cx: w / 2, cy: h / 2, s };
    if (redraw) this.render();
  }

  zoom(f, px = null, py = null) {
    if (!this.view) return;
    const { W, H } = this._size();
    const v = this.view;
    const ax = px === null ? W / 2 : px, ay = py === null ? H / 2 : py;
    const wx = v.cx + (ax - W / 2) / v.s, wy = v.cy - (ay - H / 2) / v.s;
    const s = Math.min(Math.max(v.s * f, this._minScale() / 4), this._minScale() * 400);
    this.view = { s, cx: wx - (ax - W / 2) / s, cy: wy + (ay - H / 2) / s };
    this.render();
  }

  _minScale() {
    if (!this.data) return 1;
    const [w, h] = this.data.head.size_um;
    const { W, H } = this._size();
    return Math.min(W / w, H / h);
  }

  toScreen(x, y) {
    const { W, H } = this._size();
    const v = this.view;
    return [W / 2 + (x - v.cx) * v.s, H / 2 - (y - v.cy) * v.s];
  }

  toWorld(px, py) {
    const { W, H } = this._size();
    const v = this.view;
    return [v.cx + (px - W / 2) / v.s, v.cy - (py - H / 2) / v.s];
  }

  _resize() {
    const { W, H } = this._size();
    const dpr = window.devicePixelRatio || 1;
    for (const c of [this.el.gl, this.el.ov]) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    if (this.data && this.view) {
      // keep the zoom relative to the fitted size
      if (!this._lastW) this.fit(false);
      else this.view.s *= W / this._lastW;
    }
    this._lastW = W;
    this.render();
  }

  _wireMouse() {
    const st = this.el.stage;
    let drag = null;
    st.addEventListener("wheel", (e) => {
      if (!this.view) return;
      e.preventDefault();
      const r = st.getBoundingClientRect();
      this.zoom(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018)), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    st.addEventListener("pointerdown", (e) => {
      if (!this.view || e.button !== 0) return;
      st.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, cx: this.view.cx, cy: this.view.cy };
      st.classList.add("dragging");
    });
    st.addEventListener("pointermove", (e) => {
      const r = st.getBoundingClientRect();
      if (drag) {
        this.view.cx = drag.cx - (e.clientX - drag.x) / this.view.s;
        this.view.cy = drag.cy + (e.clientY - drag.y) / this.view.s;
        this.render();
      }
      this._hover(e.clientX - r.left, e.clientY - r.top);
    });
    const end = () => { drag = null; st.classList.remove("dragging"); };
    st.addEventListener("pointerup", end);
    st.addEventListener("pointercancel", end);
    st.addEventListener("pointerleave", () => { this.el.readout.textContent = ""; });
    st.addEventListener("dblclick", () => this.fit());
    st.addEventListener("keydown", (e) => {
      if (e.key === "+" || e.key === "=") this.zoom(1.5);
      else if (e.key === "-" || e.key === "_") this.zoom(1 / 1.5);
      else if (e.key === "0") this.fit();
      else return;
      e.preventDefault();
    });
  }

  // ---- hover --------------------------------------------------------------------------------
  _units(x, y) {
    if (this.units === "ndim") {
      const { ld, ss } = this.data.head;
      return `(${(x / ld - ss[0]).toFixed(3)}, ${(y / ld - ss[1]).toFixed(3)})`;
    }
    return `(${x.toFixed(1)}, ${y.toFixed(1)}) µm`;
  }

  _buildGrid() {
    const a = this.data.arrays;
    const [w, h] = this.data.head.size_um;
    const nt = a.mesh_tri.length / 3;
    const cell = Math.sqrt((w * h) / nt) * 3;
    const nx = Math.max(1, Math.ceil(w / cell)), ny = Math.max(1, Math.ceil(h / cell));
    const P = a.mesh_pts, T = a.mesh_tri;
    const cellOf = (x, y) => Math.min(ny - 1, Math.max(0, Math.floor(y / cell))) * nx + Math.min(nx - 1, Math.max(0, Math.floor(x / cell)));
    // each triangle in every cell its bounding box touches (triangles are small: usually one)
    const ranges = new Int32Array(nt * 4);
    const count = new Int32Array(nx * ny + 1);
    for (let t = 0; t < nt; t++) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let c = 0; c < 3; c++) { const i = T[3 * t + c]; x0 = Math.min(x0, P[2 * i]); x1 = Math.max(x1, P[2 * i]); y0 = Math.min(y0, P[2 * i + 1]); y1 = Math.max(y1, P[2 * i + 1]); }
      const i0 = Math.max(0, Math.floor(x0 / cell)), i1 = Math.min(nx - 1, Math.floor(x1 / cell));
      const j0 = Math.max(0, Math.floor(y0 / cell)), j1 = Math.min(ny - 1, Math.floor(y1 / cell));
      ranges.set([i0, i1, j0, j1], 4 * t);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) count[j * nx + i + 1]++;
    }
    for (let k = 1; k < count.length; k++) count[k] += count[k - 1];
    const fill = count.slice(0, -1);
    const items = new Int32Array(count[count.length - 1]);
    for (let t = 0; t < nt; t++) {
      const [i0, i1, j0, j1] = ranges.subarray(4 * t, 4 * t + 4);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) items[fill[j * nx + i]++] = t;
    }
    this._grid = { cell, nx, ny, start: count, items, cellOf };
  }

  // triangle containing (x, y) and its barycentric weights, or null
  locate(x, y) {
    const a = this.data && this.data.arrays;
    if (!a || !a.mesh_tri) return null;
    if (!this._grid) this._buildGrid();
    const g = this._grid, P = a.mesh_pts, T = a.mesh_tri;
    const k = g.cellOf(x, y);
    for (let p = g.start[k]; p < g.start[k + 1]; p++) {
      const t = g.items[p];
      const i = T[3 * t], j = T[3 * t + 1], l = T[3 * t + 2];
      const x1 = P[2 * i], y1 = P[2 * i + 1], x2 = P[2 * j], y2 = P[2 * j + 1], x3 = P[2 * l], y3 = P[2 * l + 1];
      const d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
      if (d === 0) continue;
      const w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d;
      const w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d;
      const w3 = 1 - w1 - w2;
      const e = -1e-9;
      if (w1 >= e && w2 >= e && w3 >= e) return { t, nodes: [i, j, l], w: [w1, w2, w3] };
    }
    return null;
  }

  fibreAt(x, y) {
    let best = null;
    for (const f of this._fibres || []) {
      const [x0, x1, y0, y1] = f.box;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      let inside = false;
      const p = f.pts, n = p.length / 2;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = p[2 * i], yi = p[2 * i + 1], xj = p[2 * j], yj = p[2 * j + 1];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
      if (inside && (!best || f.area < best.area)) best = f;
    }
    return best;
  }

  _hover(px, py) {
    if (!this.data || !this.view) return;
    const [x, y] = this.toWorld(px, py);
    const [w, h] = this.data.head.size_um;
    if (x < 0 || y < 0 || x > w || y > h) { this.el.readout.textContent = ""; return; }
    const parts = [this._units(x, y)];
    const a = this.data.arrays;
    // capillary under the cursor
    const rc = Math.max(this.data.head.rcap_um || 2, 4 / this.view.s);
    let cap = -1;
    if (a.caps) for (let i = 0; i < a.caps.length / 2; i++) if (Math.hypot(a.caps[2 * i] - x, a.caps[2 * i + 1] - y) <= rc) { cap = i; break; }
    if (cap >= 0) parts.push(`capillary ${cap + 1}`);
    else {
      const loc = this.locate(x, y);
      if (loc && a.po2) {
        const v = loc.w[0] * a.po2[loc.nodes[0]] + loc.w[1] * a.po2[loc.nodes[1]] + loc.w[2] * a.po2[loc.nodes[2]];
        parts.push(`PO₂ ${Math.max(0, v).toFixed(2)} mmHg`);
      }
      if (loc && a.vo2_tri) parts.push(`O₂ consumption ${fmtNum(a.vo2_tri[loc.t], 3)} ml/100 ml/min`);
      const f = this.fibreAt(x, y);
      parts.push(f ? `fibre ${f.owner + 1}, type ${TYPE_NAMES[f.type] ?? "?"}` : (this.data.head.n_fibres ? "interstitial space" : ""));
    }
    this.el.readout.textContent = parts.filter(Boolean).join(" · ");
  }

  // ---- drawing ------------------------------------------------------------------------------
  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  _draw() {
    if (!this.data || !this.view) return;
    this._drawGL();
    this._drawOverlay(this.el.ov.getContext("2d"), window.devicePixelRatio || 1);
  }

  _drawGL() {
    const gl = this.gl;
    if (!gl) return;
    const c = this.el.gl;
    gl.viewport(0, 0, c.width, c.height);
    const m = this.layers.map;
    const light = m === "types" || m === "none" || !this._gpu.pos;
    const bg = light ? [0.96, 0.97, 0.98] : [0.08, 0.09, 0.13];
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this._gpu.pos) return;
    const { W, H } = this._size();
    const v = this.view;
    gl.useProgram(this.prog);
    gl.uniform2f(this.loc.center, v.cx, v.cy);
    gl.uniform2f(this.loc.scale, (2 * v.s) / W, (2 * v.s) / H);
    this._uploadLUT();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.loc.lut, 0);
    const r = this._currentRange() || [0, 1];
    gl.uniform2f(this.loc.range, r[0], r[1]);
    const bindPos = (buf) => { gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0); };
    const bindVal = (buf) => {
      if (buf) { gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(this.loc.val); gl.vertexAttribPointer(this.loc.val, 1, gl.FLOAT, false, 0, 0); }
      else { gl.disableVertexAttribArray(this.loc.val); gl.vertexAttrib1f(this.loc.val, 0); }
    };
    if (m === "po2" && this._gpu.po2) {
      gl.uniform4f(this.loc.solid, 0, 0, 0, 0);
      bindPos(this._gpu.pos); bindVal(this._gpu.po2);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._gpu.tri);
      gl.drawElements(gl.TRIANGLES, this.data.arrays.mesh_tri.length, gl.UNSIGNED_INT, 0);
    } else if (m === "vo2" && this._gpu.vo2pos) {
      gl.uniform4f(this.loc.solid, 0, 0, 0, 0);
      bindPos(this._gpu.vo2pos); bindVal(this._gpu.vo2val);
      gl.drawArrays(gl.TRIANGLES, 0, this._gpu.vo2n);
    }
    if (this.layers.mesh && this._edgeBuffer()) {
      const dark = light;
      gl.uniform4f(this.loc.solid, dark ? 0.2 : 1, dark ? 0.22 : 1, dark ? 0.3 : 1, dark ? 0.35 : 0.22);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      bindPos(this._gpu.pos); bindVal(null);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._gpu.edges);
      gl.drawElements(gl.LINES, this._gpu.edgeN, gl.UNSIGNED_INT, 0);
      gl.disable(gl.BLEND);
    }
  }

  _path(ctx, pts, closed) {
    const n = pts.length / 2;
    if (n < 2) return;
    let [x, y] = this.toScreen(pts[0], pts[1]);
    ctx.moveTo(x, y);
    for (let i = 1; i < n; i++) { [x, y] = this.toScreen(pts[2 * i], pts[2 * i + 1]); ctx.lineTo(x, y); }
    if (closed) ctx.closePath();
  }

  _flatPaths(ctx, pts, off, closed) {
    for (let k = 0; k + 1 < off.length; k++) this._path(ctx, pts.subarray(2 * off[k], 2 * off[k + 1]), closed);
  }

  _drawOverlay(ctx, dpr) {
    const { W, H } = this._size();
    const a = this.data.arrays, head = this.data.head;
    const m = this.layers.map;
    const onMap = (m === "po2" && a.po2) || (m === "vo2" && a.vo2_tri);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const [w, h] = head.size_um;
    // tissue frame
    const [fx0, fy0] = this.toScreen(0, h), [fx1, fy1] = this.toScreen(w, 0);
    if (!onMap) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
    }
    ctx.strokeStyle = onMap ? "rgba(255,255,255,0.35)" : "#9aa3b2";
    ctx.lineWidth = 1;
    ctx.strokeRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
    // fibre-type fill
    if (m === "types" && this._fibres.length) {
      for (const f of this._fibres) {
        const c = TYPE_COLOURS[f.type] || TYPE_COLOURS[0];
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.45)`;
        ctx.beginPath(); this._path(ctx, f.pts, true); ctx.fill();
      }
    }
    // capillary domains (red, as the Python and MATLAB figures)
    if (this.layers.voronoi && a.vor_pts) {
      ctx.strokeStyle = "rgba(230,40,40,0.95)";
      ctx.lineWidth = 1.1;
      ctx.beginPath(); this._flatPaths(ctx, a.vor_pts, a.vor_off, true); ctx.stroke();
    }
    // fibre outlines
    if (this.layers.fibres && this._fibres.length) {
      ctx.strokeStyle = onMap ? "rgba(12,14,22,0.9)" : "#2a2f3c";
      ctx.lineWidth = onMap ? 0.9 : 1;
      ctx.beginPath();
      for (const f of this._fibres) this._path(ctx, f.pts, true);
      ctx.stroke();
    }
    // flux lines
    if (this.layers.flux && a.flux_pts) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = onMap ? "rgba(255,255,255,0.9)" : "rgba(20,90,200,0.9)";
      ctx.beginPath(); this._flatPaths(ctx, a.flux_pts, a.flux_off, false); ctx.stroke();
    }
    // capillaries: true radius, at least 2.5 px
    if (this.layers.capillaries && a.caps) {
      const r = Math.max(2.5, (head.rcap_um || 1.8) * this.view.s);
      ctx.fillStyle = onMap ? "#ffffff" : "#d61f1f";
      ctx.strokeStyle = onMap ? "#0c0e16" : "#6b0000";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < a.caps.length / 2; i++) {
        const [x, y] = this.toScreen(a.caps[2 * i], a.caps[2 * i + 1]);
        if (x < -r || y < -r || x > W + r || y > H + r) continue;
        ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 2 * Math.PI);
      }
      ctx.fill(); ctx.stroke();
    }
    // ROI
    if (this.layers.roi && head.roi_um) {
      const [x0, x1, y0, y1] = head.roi_um;
      const [sx0, sy0] = this.toScreen(x0, y1), [sx1, sy1] = this.toScreen(x1, y0);
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = onMap ? "#ffffff" : "#008ba3";
      ctx.lineWidth = 1.6;
      ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.setLineDash([]);
    }
    this._scaleBar(ctx, W, H, onMap);
  }

  _scaleBar(ctx, W, H, onMap) {
    const target = 110 / this.view.s;                    // µm in ~110 px
    let len, label;
    if (this.units === "ndim") {
      const ld = this.data.head.ld;
      const nd = niceStep(target / ld);
      len = nd * ld;
      label = `${+nd.toPrecision(3)} model units`;
    } else {
      len = niceStep(target);
      label = `${+len.toPrecision(3)} µm`;
    }
    const px = len * this.view.s;
    const x = 14, y = H - 16;
    ctx.font = "12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = onMap ? "rgba(10,12,20,0.6)" : "rgba(255,255,255,0.8)";
    ctx.fillRect(x - 6, y - 20, Math.max(px, tw) + 12, 28);
    ctx.fillStyle = onMap ? "#ffffff" : "#1b2030";
    ctx.fillRect(x, y, px, 3);
    ctx.fillText(label, x, y - 5);
  }

  // ---- export -------------------------------------------------------------------------------
  savePNG(name = null) {
    if (!this.data) return;
    this._draw();
    const dpr = window.devicePixelRatio || 1;
    const { W, H } = this._size();
    const r = this._currentRange();
    const mapOn = r && ((this.layers.map === "po2" && this.data.arrays.po2) || (this.layers.map === "vo2" && this.data.arrays.vo2_tri));
    const extra = mapOn ? 46 : 0;
    const c = document.createElement("canvas");
    c.width = Math.round(W * dpr);
    c.height = Math.round((H + extra) * dpr);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, c.width, c.height);
    if (this.gl) ctx.drawImage(this.el.gl, 0, 0);
    ctx.drawImage(this.el.ov, 0, 0);
    if (mapOn) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const x0 = 16, y0 = H + 10, bw = Math.min(360, W - 200);
      const cl = lut(this.colormap);
      for (let i = 0; i < bw; i++) {
        const k = Math.round((i / (bw - 1)) * 255) * 3;
        ctx.fillStyle = `rgb(${cl[k]},${cl[k + 1]},${cl[k + 2]})`;
        ctx.fillRect(x0 + i, y0, 1.5, 12);
      }
      ctx.fillStyle = "#1b2030";
      ctx.font = "12px system-ui, sans-serif";
      const d = this.layers.map === "po2" ? 1 : 2;
      ctx.fillText(fmtNum(r[0], d), x0, y0 + 27);
      const hi = fmtNum(r[1], d);
      ctx.fillText(hi, x0 + bw - ctx.measureText(hi).width, y0 + 27);
      ctx.fillText(this.layers.map === "po2" ? "PO₂ (mmHg)" : "O₂ consumption (ml O₂/100 ml/min)", x0 + bw + 14, y0 + 11);
    }
    c.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name || `${this.data.head.name}_view_${this.layers.map}.png`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    }, "image/png");
  }
}
