// OTM Web corrections editor: the desktop's "Correct data" step (otm_desktop/widgets/tissue_editor.py).
//
// Draws the data as a run will use them (centred pixel frame, y up: the frame of otm_core.edits)
// and turns every click into an edit, {kind, ...} as Edit.to_dict, passed to onEdit. It never changes
// the data itself: the page sends the edit list to Python (otm_web.prepare) and calls setData with
// the corrected data. Tools as on the desktop:
//
//   pan               drag to pan (the wheel zooms with every tool)
//   add_capillary     click to add
//   remove_capillary  click a capillary
//   move_capillary    drag a capillary
//   fiber_type        click a fibre to give it the chosen type
//   delete_fiber      click a fibre
//   draw_fiber        click the corners of a new fibre; double-click or Enter closes it
//   redraw_fiber      click a fibre, then its new corners; double-click or Enter closes it
//
// Esc cancels an outline being drawn.

import { TYPE_COLOURS, TYPE_NAMES } from "./viewer.js";

export const TOOLS = [
  ["pan", "Pan / zoom", "Drag to move the view; the mouse wheel zooms."],
  ["add_capillary", "Add capillary", "Click where a capillary is missing."],
  ["remove_capillary", "Remove capillary", "Click a capillary that is not one."],
  ["move_capillary", "Move capillary", "Drag a capillary to where it belongs."],
  ["fiber_type", "Set fibre type", "Click fibres to give them the type chosen below."],
  ["delete_fiber", "Delete fibre", "Click a fibre that is not one."],
  ["draw_fiber", "Draw fibre", "Click the corners of a missing fibre; double-click (or Enter) to close it. Esc cancels."],
  ["redraw_fiber", "Redraw outline", "Click a fibre, then the corners of its correct outline; double-click (or Enter) to finish."],
];

const CAP_R = 4;          // capillary dot radius, screen px
const PICK = 9;           // pick distance, screen px

export class CorrectionsEditor {
  constructor(stage, { onEdit = () => {}, onHover = () => {}, onTool = () => {} } = {}) {
    this.stage = stage;
    this.canvas = document.createElement("canvas");
    stage.append(this.canvas);
    this.onEdit = onEdit;
    this.onHover = onHover;
    this.onTool = onTool;
    this.tool = "pan";
    this.fiberType = 1;
    this.data = null;
    this.view = null;
    this.outline = [];
    this.redrawTarget = null;       // fibre id being redrawn
    this.dragCap = null;            // {i, x, y} capillary being moved
    this._raf = 0;
    this._wire();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(stage);
  }

  destroy() { this._ro.disconnect(); cancelAnimationFrame(this._raf); this.canvas.remove(); }

  // ---- data ---------------------------------------------------------------------------------
  /** payload from otm_web.prepare */
  setData(p) {
    const first = !this.data;
    this.data = p;
    this.fibres = p.fibres.map((r) => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, area = 0;
      for (let i = 0; i < r.length; i++) {
        const [x, y] = r[i];
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        const [xn, yn] = r[(i + 1) % r.length];
        area += x * yn - xn * y;
      }
      return { r, box: [x0, x1, y0, y1], area: Math.abs(area) / 2 };
    });
    this.flagF = new Map(p.flags.fibres.map(([i, label]) => [i, label]));
    if (first || !this.view) this.fit(false);
    if (this.redrawTarget !== null && !p.ids.includes(this.redrawTarget)) this.cancelOutline();
    this.render();
  }

  // ---- tools --------------------------------------------------------------------------------
  setTool(t) {
    if (!TOOLS.some(([k]) => k === t)) throw new Error(`unknown tool ${t}`);
    this.cancelOutline();
    this.tool = t;
    this.canvas.style.cursor = t === "pan" ? "grab" : "crosshair";
    this.onTool(t);
  }

  setFiberType(code) { this.fiberType = Number(code); }

  cancelOutline() {
    this.outline = [];
    this.redrawTarget = null;
    this.render();
  }

  finishOutline() {
    const pts = this.outline.map(([x, y]) => [Math.round(x * 100) / 100, Math.round(y * 100) / 100]);
    const target = this.redrawTarget;
    this.outline = [];
    this.redrawTarget = null;
    this.render();
    if (pts.length < 3) return null;
    if (this.tool === "redraw_fiber" && target !== null) return this._emit({ kind: "replace_fiber", fiber: target, ring: pts });
    if (this.tool === "draw_fiber") return this._emit({ kind: "add_fiber", ring: pts, fiber_type: this.fiberType });
    return null;
  }

  _emit(e) { this.onEdit(e); return e; }

  pickTolerance() { return PICK / (this.view ? this.view.s : 1); }

  nearestCapillary(x, y, tol) {
    const c = this.data ? this.data.caps : [];
    let best = -1, bd = tol;
    for (let i = 0; i < c.length; i++) {
      const d = Math.hypot(c[i][0] - x, c[i][1] - y);
      if (d <= bd) { bd = d; best = i; }
    }
    return best >= 0 ? best : null;
  }

  fiberAt(x, y) {                     // the smallest fibre containing (x, y), as otm_core.edits.fiber_at
    let best = null, ba = Infinity;
    (this.fibres || []).forEach((f, i) => {
      const [x0, x1, y0, y1] = f.box;
      if (x < x0 || x > x1 || y < y0 || y > y1) return;
      let inside = false;
      const r = f.r;
      for (let a = 0, b = r.length - 1; a < r.length; b = a++) {
        const [xa, ya] = r[a], [xb, yb] = r[b];
        if ((ya > y) !== (yb > y) && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) inside = !inside;
      }
      if (inside && f.area < ba) { best = i; ba = f.area; }
    });
    return best;
  }

  clickAt(x, y) {
    const t = this.tool;
    const r2 = (v) => Math.round(v * 100) / 100;
    if (t === "add_capillary") return this._emit({ kind: "add_capillary", x: r2(x), y: r2(y) });
    if (t === "remove_capillary") {
      const i = this.nearestCapillary(x, y, this.pickTolerance());
      if (i !== null) return this._emit({ kind: "remove_capillary", x: this.data.caps[i][0], y: this.data.caps[i][1] });
    } else if (t === "fiber_type" || t === "delete_fiber") {
      const i = this.fiberAt(x, y);
      if (i !== null) {
        const fid = this.data.ids[i];
        if (t === "delete_fiber") return this._emit({ kind: "delete_fiber", fiber: fid });
        if (this.data.types[i] !== this.fiberType) return this._emit({ kind: "set_fiber_type", fiber: fid, fiber_type: this.fiberType });
      }
    } else if (t === "draw_fiber") {
      this.outline.push([x, y]);
      this.render();
    } else if (t === "redraw_fiber") {
      if (this.redrawTarget === null) {
        const i = this.fiberAt(x, y);
        if (i !== null) {
          this.redrawTarget = this.data.ids[i];
          this.onHover(`Redrawing fibre ${i + 1}: click its corners, double-click to finish`);
          this.render();
        }
      } else {
        this.outline.push([x, y]);
        this.render();
      }
    }
    return null;
  }

  // ---- view ---------------------------------------------------------------------------------
  _size() { const r = this.stage.getBoundingClientRect(); return { W: Math.max(1, r.width), H: Math.max(1, r.height) }; }

  fit(redraw = true) {
    if (!this.data) return;
    const [w, h] = this.data.frame;
    const { W, H } = this._size();
    this.view = { cx: 0, cy: 0, s: 0.94 * Math.min(W / w, H / h) };
    if (redraw) this.render();
  }

  zoom(f, px = null, py = null) {
    if (!this.view) return;
    const { W, H } = this._size();
    const v = this.view;
    const ax = px ?? W / 2, ay = py ?? H / 2;
    const [wx, wy] = this.toWorld(ax, ay);
    const s = Math.min(Math.max(v.s * f, 0.02), 60);
    this.view = { s, cx: wx - (ax - W / 2) / s, cy: wy + (ay - H / 2) / s };
    this.render();
  }

  toScreen(x, y) { const { W, H } = this._size(); const v = this.view; return [W / 2 + (x - v.cx) * v.s, H / 2 - (y - v.cy) * v.s]; }
  toWorld(px, py) { const { W, H } = this._size(); const v = this.view; return [v.cx + (px - W / 2) / v.s, v.cy - (py - H / 2) / v.s]; }

  _resize() {
    const { W, H } = this._size();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    this.canvas.style.width = W + "px";
    this.canvas.style.height = H + "px";
    if (this.data && (!this._lastW || !this.view)) this.fit(false);
    else if (this.view && this._lastW) this.view.s *= W / this._lastW;
    this._lastW = W;
    this.render();
  }

  _wire() {
    const c = this.canvas;
    c.tabIndex = 0;
    let pan = null;
    const local = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [px, py] = local(e);
      this.zoom(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018)), px, py);
    }, { passive: false });
    c.addEventListener("pointerdown", (e) => {
      if (!this.view) return;
      c.focus();
      const [px, py] = local(e);
      const [x, y] = this.toWorld(px, py);
      if (e.button === 1 || (e.button === 0 && this.tool === "pan")) {
        c.setPointerCapture(e.pointerId);
        pan = { x: e.clientX, y: e.clientY, cx: this.view.cx, cy: this.view.cy };
        c.style.cursor = "grabbing";
        return;
      }
      if (e.button !== 0) return;
      if (this.tool === "move_capillary") {
        const i = this.nearestCapillary(x, y, this.pickTolerance());
        if (i !== null) { c.setPointerCapture(e.pointerId); this.dragCap = { i, x, y }; }
        return;
      }
      if (e.detail >= 2 && (this.tool === "draw_fiber" || this.tool === "redraw_fiber")) return;   // the dblclick closes it
      this.clickAt(x, y);
    });
    c.addEventListener("pointermove", (e) => {
      const [px, py] = local(e);
      if (pan) {
        this.view.cx = pan.cx - (e.clientX - pan.x) / this.view.s;
        this.view.cy = pan.cy + (e.clientY - pan.y) / this.view.s;
        this.render();
        return;
      }
      if (!this.view || !this.data) return;
      const [x, y] = this.toWorld(px, py);
      if (this.dragCap) { this.dragCap.x = x; this.dragCap.y = y; this.render(); }
      this._cursor = [x, y];
      if (this.outline.length) this.render();
      this._hover(x, y);
    });
    c.addEventListener("pointerup", (e) => {
      if (pan) { pan = null; c.style.cursor = this.tool === "pan" ? "grab" : "crosshair"; return; }
      if (this.dragCap) {
        const d = this.dragCap;
        this.dragCap = null;
        const [x0, y0] = this.data.caps[d.i];
        const [px, py] = local(e);
        const [x1, y1] = this.toWorld(px, py);
        if (Math.hypot(x1 - x0, y1 - y0) * this.view.s > 1)
          this._emit({ kind: "move_capillary", x: x0, y: y0, x2: Math.round(x1 * 100) / 100, y2: Math.round(y1 * 100) / 100 });
        this.render();
      }
    });
    c.addEventListener("dblclick", (e) => {
      e.preventDefault();
      if ((this.tool === "draw_fiber" || this.tool === "redraw_fiber") && this.outline.length) {
        // the second click of the double-click added no corner (see pointerdown)
        this.finishOutline();
      } else if (this.tool === "pan") this.fit();
    });
    c.addEventListener("keydown", (e) => {
      if ((e.key === "Enter") && this.outline.length) { e.preventDefault(); this.finishOutline(); }
      else if (e.key === "Escape") {
        if (this.outline.length || this.redrawTarget !== null) { e.preventDefault(); e.stopPropagation(); }   // not the dialog
        this.cancelOutline();
      }
      else if (e.key === "Backspace" && this.outline.length) { e.preventDefault(); this.outline.pop(); this.render(); }
      else if (e.key === "+" || e.key === "=") this.zoom(1.4);
      else if (e.key === "-") this.zoom(1 / 1.4);
      else if (e.key === "0") this.fit();
    });
    c.addEventListener("pointerleave", () => { this._cursor = null; this.render(); });
  }

  _hover(x, y) {
    const i = this.nearestCapillary(x, y, this.pickTolerance());
    if (i !== null) return this.onHover(`Capillary at (${this.data.caps[i][0].toFixed(0)}, ${this.data.caps[i][1].toFixed(0)}) px`);
    const f = this.fiberAt(x, y);
    if (f !== null) {
      const flag = this.flagF.get(f);
      return this.onHover(`Fibre ${f + 1}, type ${TYPE_NAMES[this.data.types[f]] ?? "?"}` + (flag ? ` · flagged: ${flag}` : ""));
    }
    this.onHover(`(${x.toFixed(0)}, ${y.toFixed(0)}) px`);
  }

  // ---- drawing ------------------------------------------------------------------------------
  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  _ring(ctx, r, closed = true) {
    if (!r.length) return;
    let [x, y] = this.toScreen(r[0][0], r[0][1]);
    ctx.moveTo(x, y);
    for (let i = 1; i < r.length; i++) { [x, y] = this.toScreen(r[i][0], r[i][1]); ctx.lineTo(x, y); }
    if (closed) ctx.closePath();
  }

  _draw() {
    const ctx = this.canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const { W, H } = this._size();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#141622";
    ctx.fillRect(0, 0, W, H);
    if (!this.data || !this.view) return;
    const [w, h] = this.data.frame;
    const [fx0, fy0] = this.toScreen(-w / 2, h / 2), [fx1, fy1] = this.toScreen(w / 2, -h / 2);
    ctx.fillStyle = "#1E2235";
    ctx.fillRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
    ctx.strokeStyle = "#3E4668";
    ctx.lineWidth = 1;
    ctx.strokeRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
    // fibres by type
    this.fibres.forEach((f, i) => {
      const c = TYPE_COLOURS[this.data.types[i]] || TYPE_COLOURS[0];
      ctx.beginPath();
      this._ring(ctx, f.r);
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.55)`;
      ctx.fill();
      const target = this.redrawTarget !== null && this.data.ids[i] === this.redrawTarget;
      ctx.strokeStyle = target ? "#00E5FF" : "#94A3B8";
      ctx.lineWidth = target ? 2.2 : 0.8;
      ctx.stroke();
    });
    // data-check marks: flagged fibres dashed and labelled, flagged capillaries ringed
    ctx.strokeStyle = "#D9806F";
    ctx.fillStyle = "#D9806F";
    ctx.setLineDash([7, 4]);
    ctx.lineWidth = 2.5;
    ctx.font = "bold 13px system-ui, sans-serif";
    for (const [i, label] of this.flagF) {
      const f = this.fibres[i];
      if (!f) continue;
      ctx.beginPath(); this._ring(ctx, f.r); ctx.stroke();
      const top = f.r.reduce((a, b) => (b[1] > a[1] ? b : a));
      const [tx, ty] = this.toScreen(top[0], top[1]);
      ctx.fillText(label, tx + 4, Math.max(14, ty - 4));
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    for (const [x, y] of this.data.flags.caps) {
      const [sx, sy] = this.toScreen(x, y);
      ctx.beginPath(); ctx.arc(sx, sy, 2.2 * CAP_R, 0, 2 * Math.PI); ctx.stroke();
    }
    // capillaries (fixed screen size, as on the desktop)
    ctx.fillStyle = "#FF5252";
    ctx.strokeStyle = "#E2E8F0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    this.data.caps.forEach(([x, y], i) => {
      if (this.dragCap && this.dragCap.i === i) [x, y] = [this.dragCap.x, this.dragCap.y];
      const [sx, sy] = this.toScreen(x, y);
      if (sx < -CAP_R || sy < -CAP_R || sx > W + CAP_R || sy > H + CAP_R) return;
      ctx.moveTo(sx + CAP_R, sy);
      ctx.arc(sx, sy, CAP_R, 0, 2 * Math.PI);
    });
    ctx.fill();
    ctx.stroke();
    // outline being drawn
    if (this.outline.length) {
      const pts = this._cursor ? [...this.outline, this._cursor] : this.outline;
      ctx.strokeStyle = "#00E5FF";
      ctx.lineWidth = 2;
      ctx.beginPath(); this._ring(ctx, pts, false); ctx.stroke();
      ctx.fillStyle = "#00E5FF";
      for (const [x, y] of this.outline) { const [sx, sy] = this.toScreen(x, y); ctx.fillRect(sx - 3, sy - 3, 6, 6); }
    }
  }
}
