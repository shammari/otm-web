"""Python side of OTM Web: runs inside Pyodide, in the page's background worker.

The worker (``worker.js``) writes the user's files to ``/data`` and calls these
functions; everything else is :mod:`otm_core`, exactly as on the desktop. All
results go back to the page as JSON strings.

Folders in the in-browser file system (memory only, gone when the tab closes)::

    /data/<sample>.mat    the uploaded Dtect export (+ <sample>.otm with saved corrections)
    /out/<sample>/INDICES and /out/<sample>/PO2    what run_file writes, as the desktop app
    /out/OTM_batch_summary.xlsx/.csv, /out/OTM_batch_settings.json
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import shutil
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional

import numpy as np

DATA = "/data"
OUT = "/out"
ZIP_PATH = "/tmp/OTM_results.zip"

_ROWS: Dict[str, Dict[str, Any]] = {}        # sample -> batch summary row of its last run

for _d in (DATA, OUT):
    os.makedirs(_d, exist_ok=True)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _clean(v: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None, tuples to lists."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def _dumps(v: Any) -> str:
    return json.dumps(_clean(v))


def _data_path(name: str) -> str:
    return os.path.join(DATA, name + ".mat")


# ----------------------------------------------------------------------------------------------
# start-up information for the settings form
# ----------------------------------------------------------------------------------------------

def info() -> str:
    """Defaults and choices for the settings form (single source: otm_core)."""
    import otm_core
    from otm_core.fem import solve as fem_solve
    from otm_core.fem.mesh import mesh_backend
    from otm_core.figures import FORMATS, WIDTH_LABELS
    from otm_core.pipeline import EXERCISE_LEVELS, STEPS, RunSettings

    TP = fem_solve.TransportParameters
    exercise = {t: {lv: _switch_info(getattr(fem_solve.ModelSwitches, t)(lv)) for lv in EXERCISE_LEVELS}
                for t in ("skeletal", "cardiac")}
    names = [f.name for f in dataclasses.fields(TP)]
    fields = [{"name": n, "label": d[0], "symbol": d[1], "unit": d[2]} for n, d in zip(names, TP.DESCRIPTIONS)]
    defaults = {t: dict(zip(names, v)) for t, v in TP.DEFAULTS.items()}
    import sys
    return _dumps({
        "otm_core": getattr(otm_core, "__version__", "?"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mesher": mesh_backend(),
        "parameters": {"fields": fields, "defaults": defaults},
        "settings": RunSettings().to_dict(),
        "exercise_levels": list(EXERCISE_LEVELS),
        "exercise": exercise,
        "steps": list(STEPS),
        "widths": WIDTH_LABELS,
        "formats": list(FORMATS),
    })


def _switch_info(sw: Any) -> Dict[str, Any]:
    """What an exercise level does (otm_core.fem.ModelSwitches): the multipliers the solver applies."""
    return {"michaelis_menten": bool(sw.michaelis_menten), "myoglobin": bool(sw.myoglobin),
            "demand": float(sw.exercise_level), "permeability": float(sw.permeability_factor)}


# ----------------------------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------------------------

def _opts(opts_json: Optional[str]) -> Dict[str, Any]:
    o = json.loads(opts_json) if opts_json else {}
    return {"tissue": o.get("tissue") or "skeletal", "use_fibre_types": o.get("use_fibre_types", True) is not False,
            "data_check": o.get("data_check") or "fix"}


def inspect(name: str, opts_json: str = "") -> str:
    """What the desktop's Load data step shows: image size, capillaries, fibres,
    fibre types and the data check; plus the saved corrections in ``<name>.otm``
    if one was added. ``opts_json``: tissue and use_fibre_types (the check depends on them)."""
    from otm_core import qc
    from otm_core.io import is_hdf5_mat, load_dtect_mat

    o = _opts(opts_json)
    path = _data_path(name)
    t0 = time.perf_counter()
    try:
        d = load_dtect_mat(path)
    except Exception as exc:                        # shown on the file's row; the file is left out
        return _dumps({"ok": False, "error": _plain_error(exc)})
    types = None
    if d.fiber_types is not None:
        codes = np.asarray(d.fiber_types).astype(int).ravel()
        types = {lab: int(np.sum(codes == c)) for lab, c in (("I", 1), ("IIa", 21), ("IIb", 22), ("unknown", 0))}
    rows, cols = (int(v) for v in d.image_size[:2])
    try:
        check = _qc_public(qc.check(d, tissue=o["tissue"], use_fibre_types=o["use_fibre_types"]))
    except Exception as exc:                        # the check itself must never hide the file
        check = {"status": "warning", "headline": f"The data check could not run: {_plain_error(exc)}",
                 "issues": [], "fixed": [], "actions": [], "blocking": False, "can_keep": True}
    out = {"ok": True, "image_rows": rows, "image_cols": cols, "n_capillaries": int(len(d.capillaries_px)),
           "n_fibres": int(len(d.fibers_px)), "fibre_types": types, "notes": list(d.notes),
           "format": "MATLAB v7.3" if is_hdf5_mat(path) else "MATLAB v5", "check": check,
           "seconds": time.perf_counter() - t0}
    del d
    gc.collect()
    return _dumps(out)


# ----------------------------------------------------------------------------------------------
# data check and corrections (the editor)
# ----------------------------------------------------------------------------------------------

_ISSUE_KEYS = ("code", "severity", "title", "message", "why", "advice", "overridable", "fix")


def _qc_public(Q: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the page shows of an ``otm_core.qc`` result."""
    from otm_core import qc

    if not Q or "issues" not in Q:
        return None
    blocking, can_keep = qc.is_blocking(Q)

    def issue(i: Dict[str, Any]) -> Dict[str, Any]:
        d = {k: i.get(k) for k in _ISSUE_KEYS}
        d["n"] = max(1, len(i.get("items") or []))
        d["items"] = list(i.get("items") or [])[:30]
        return d

    return {"status": Q["status"], "headline": qc.headline(Q), "issues": [issue(i) for i in Q["issues"]],
            "fixed": [issue(i) for i in Q.get("fixed_issues") or []],
            "actions": [a["text"] for a in Q.get("actions") or []], "blocking": blocking, "can_keep": can_keep,
            "overridden": bool(Q.get("overridden"))}


def _prepared(name: str, o: Dict[str, Any]):
    """The data a run of this file starts from, as ``run_file`` makes it: loaded,
    checked, and the data-check policy applied. Returns (data, check, note);
    ``note`` says what the policy did or why the file would not be analysed."""
    from otm_core import qc
    from otm_core.io import load_dtect_mat

    d = load_dtect_mat(_data_path(name))
    Q = qc.check(d, tissue=o["tissue"], use_fibre_types=o["use_fibre_types"])
    note = ""
    if Q["status"] == "fatal":
        return d, Q, "This file cannot be analysed (see the data check)."
    blocking, can_keep = qc.is_blocking(Q)
    if blocking:
        if o["data_check"] == "skip":
            note = "With the setting 'Skip the file', this file will not be analysed."
        elif o["data_check"] == "keep" and can_keep:
            Q = qc.override(Q, "kept (setting)")
            note = "The flagged items are kept (setting 'Keep the flagged items')."
        else:
            d, act = qc.apply_fixes(d, Q)
            Q = qc.check(d, tissue=o["tissue"], use_fibre_types=o["use_fibre_types"], previous=Q, action=act)
            note = act["text"] + "."
    return d, Q, note


def _r2(a: Any) -> List[Any]:
    return np.round(np.asarray(a, float), 2).tolist()


def prepare(name: str, opts_json: str, edits_json: str) -> str:
    """Everything the corrections editor draws, with ``edits`` applied: the image
    frame, capillaries and fibre outlines (centred pixel frame, y up, as
    ``otm_core.edits``), fibre types and ids, what the data check flags, and
    one line per correction."""
    from otm_core import qc
    from otm_core.edits import apply_edits, edits_from_list
    from otm_core.geometry import dtect_to_centred, ingest_dtect

    o = _opts(opts_json)
    try:
        d, Q, note = _prepared(name, o)
        edits = edits_from_list(json.loads(edits_json or "[]"))
    except Exception as exc:
        return _dumps({"ok": False, "error": _plain_error(exc)})
    skeletal = o["tissue"] == "skeletal"
    fibres = d.fibers_px if skeletal else []
    types = d.fiber_types if (skeletal and o["use_fibre_types"]) else None
    ing = ingest_dtect(d.capillaries_px, fibres, d.image_size, types)
    er = apply_edits(ing, edits)
    g = er.ingest
    ids = [int(v) for v in er.fiber_ids]
    # what the check flags: fibres by their position in the data (= ingest id), capillaries by position
    labels_f: Dict[int, List[str]] = {}
    caps_flag: List[List[Any]] = []
    if Q and Q.get("issues"):
        caps_raw = np.asarray(d.capillaries_px, float).reshape(-1, 2)
        for it in Q["issues"]:
            if it["severity"] not in ("error", "fatal", "warning"):
                continue
            for k in it["fibres"]:
                labels_f.setdefault(int(k), []).append(it["code"])
            if it["capillaries"] and len(caps_raw):
                sel = caps_raw[[c for c in it["capillaries"] if c < len(caps_raw)]]
                cc, _ = dtect_to_centred(sel, [], g.frame)
                caps_flag += [[*_r2(p), it["code"]] for p in cc]
    flag_f = [[i, " ".join(sorted(set(labels_f[fid])))] for i, fid in enumerate(ids) if fid in labels_f]
    out = {"ok": True, "frame": [g.frame.width, g.frame.height],
           "caps": _r2(np.asarray(g.capillaries_pxc, float).reshape(-1, 2)),
           "fibres": [_r2(np.asarray(f, float).reshape(-1, 2)) for f in g.fibers_pxc],
           "types": [int(t) for t in np.asarray(g.fiber_types, int)] if len(g.fibers_pxc) else [],
           "ids": ids, "n_original_fibres": len(ing.fibers_pxc),
           "edits_text": [e.describe() for e in edits], "warnings": list(er.warnings),
           "applied": [bool(a) for a in er.applied],
           "check": _qc_public(Q), "policy_note": note, "skip": bool(note and "not be analysed" in note),
           "flags": {"fibres": flag_f, "caps": caps_flag}}
    return _dumps(out)


def session_from_edits(name: str, opts_json: str, edits_json: str) -> str:
    """A desktop session (``.otm``) with the data, the data check and the
    corrections, but no results: correct here, analyse on the desktop.
    Returns the file's path in the in-browser file system."""
    from otm_core.edits import edits_from_list
    from otm_core.session import Session, save_session

    o = _opts(opts_json)
    d, Q, _note = _prepared(name, o)
    s = Session(raw=d, source_path=_data_path(name), tissue=o["tissue"], use_fibre_types=o["use_fibre_types"],
                edits=edits_from_list(json.loads(edits_json or "[]")), data_check=Q)
    path = f"/tmp/{name}.otm"
    save_session(path, s)
    return path


def corrections(name: str) -> str:
    """Saved corrections in ``/data/<name>.otm`` (a desktop session): count and any problem."""
    from otm_core.pipeline import find_saved_edits

    try:
        edits, meta = find_saved_edits(_data_path(name))
    except Exception as exc:
        return _dumps({"ok": False, "error": _plain_error(exc)})
    if meta is None:
        return _dumps({"ok": False, "error": "not an OTM session file"})
    return _dumps({"ok": True, "n_edits": len(edits), "edits": [e.to_dict() for e in edits]})


def drop(name: str) -> None:
    for ext in (".mat", ".otm"):
        p = os.path.join(DATA, name + ext)
        if os.path.exists(p):
            os.remove(p)


# ----------------------------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------------------------

def check_settings(settings_json: str) -> str:
    """None (as JSON null) when the settings are valid, else the reason."""
    from otm_core.pipeline import RunSettings

    try:
        RunSettings.from_dict(json.loads(settings_json))
    except Exception as exc:
        return _dumps(_plain_error(exc))
    return _dumps(None)


def parse_parameters(text: str) -> str:
    """A legacy ``.dat`` parameter file (10 numbers) -> {field: value}."""
    from otm_core.fem.solve import TransportParameters

    try:
        vals = [float(t) for t in text.replace(",", " ").split()]
        p = TransportParameters.from_values(vals)
    except Exception as exc:
        return _dumps({"ok": False, "error": _plain_error(exc)})
    return _dumps({"ok": True, "values": dataclasses.asdict(p)})


def parse_dimensions(text: str) -> str:
    """A dimensions CSV (as the batch runner's ``--dims``) -> {sample stem: changes}."""
    from otm_core.pipeline import read_dimensions_table

    tmp = "/tmp/dims.csv"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    try:
        table = read_dimensions_table(tmp)
    except Exception as exc:
        return _dumps({"ok": False, "error": _plain_error(exc)})
    return _dumps({"ok": True, "table": table})


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------

VIEW_DIR = "/tmp/views"
WEB_KEYS = ("save_session", "figures_later")  # page-only settings, not part of RunSettings
_PENDING: Dict[str, Any] = {}                 # sample -> its RunResult, until its figures are written


def run(name: str, settings_json: str, progress: Callable[[float, str], Any], edits_json: str = "null") -> str:
    """Run one sample with ``run_file`` (as the desktop batch runner) and return
    what the results page shows. The big arrays are dropped before returning;
    what the tissue view draws is kept in ``/tmp/views/<name>.bin`` (:func:`view_path`).

    ``edits_json``: corrections made in the page's editor (a list, Edit.to_dict
    items), used instead of the ones in ``<name>.otm``; ``null``: use the .otm."""
    from otm_core.edits import edits_from_list
    from otm_core.fem import mo2_statistics, po2_statistics
    from otm_core.pipeline import RunSettings, find_saved_edits, run_file
    from otm_core.tables import hypoxia_text, hypoxic_compartments, indices_rows, mo2_rows, po2_rows

    raw = json.loads(settings_json)
    save = bool(raw.get("save_session", False))
    s = RunSettings.from_dict({k: v for k, v in raw.items() if k not in WEB_KEYS})
    # figures_later: the tables now, the figures with figures(name) afterwards (results are shown sooner)
    later = bool(raw.get("figures_later", False)) and s.figures and s.export
    _PENDING.pop(name, None)
    if later:
        s = s.with_changes(figures=False)
    path = _data_path(name)
    shutil.rmtree(os.path.join(OUT, name), ignore_errors=True)
    _drop_view(name)
    page_edits = json.loads(edits_json) if edits_json else None
    if page_edits is not None:
        edits, src = edits_from_list(page_edits), None
    else:
        edits, src = find_saved_edits(path)
    t0 = time.perf_counter()
    r = run_file(path, s, out_dir=OUT, base_name=name, progress=lambda f, m: progress(float(f), str(m)),
                 save_session=save, edits=edits, edits_source=src)
    out: Dict[str, Any] = {"name": name, "ok": r.ok, "seconds": time.perf_counter() - t0,
                           "errors": [{"step": k, "message": _plain_message(k, v)} for k, v in r.errors.items()],
                           "timings": r.timings, "corrections": len(r.edits),
                           "corrections_from": "page" if page_edits is not None else ("otm" if src else None),
                           "corrections_skipped": list(r.edit_warnings),
                           "check": _qc_public(r.data_check)}
    try:
        out["view"] = write_view(name, r)
    except Exception as exc:                       # the tables and files are still there
        out["view"] = None
        out["view_error"] = _plain_error(exc)
    out["switches"] = {"tissue": s.tissue, "exercise": s.exercise, **_switch_info(s.switches())}
    g = r.geometry
    if g is not None:
        ls = g.length_scale
        out["size_um"] = [ls.x_um, ls.y_um]
        out["roi_um"] = [g.roi.x_min_um, g.roi.x_max_um, g.roi.y_min_um, g.roi.y_max_um]
    if r.morphometrics is not None:
        out["indices"] = indices_rows(r.morphometrics)
    if r.solution is not None:
        pr = po2_rows(po2_statistics(r.solution))
        out["po2"] = pr
        out["po2_hypoxic_rows"] = [i for i, _n, _p in hypoxic_compartments(pr)]
        out["hypoxia"] = hypoxia_text(pr)
        out["mo2"] = mo2_rows(mo2_statistics(r.solution))
        out["solve"] = r.solution.summary()
    if r.mesh is not None:
        out["mesh"] = {"nodes": r.mesh.n_nodes, "triangles": r.mesh.n_triangles,
                       "mesher": (getattr(r.mesh, "info", None) or {}).get("source", "gmsh")}
    if r.demand is not None:
        out["demand"] = {**r.demand.summary(), "text": r.demand.text(), "enabled": r.demand.settings.enabled,
                          "warning": r.demand.warning}
    if r.flux_lines is not None:
        sm = r.flux_lines.summary()
        out["flux"] = {"seed_capillaries": sm["n_seed_capillaries"],
                       "capillaries_with_lines": sm["n_capillaries_with_lines"],
                       "moving_lines": sm["n_moving_lines"]}
        cap = float(getattr(r.flux_lines, "max_length", float("nan")))
        if math.isfinite(cap) and g is not None and g.length_scale is not None:
            out["flux"]["cap_um"] = cap * g.length_scale.y_um / 2.0
    out["files"] = sorted(os.path.relpath(f, OUT) for f in r.files)
    out["figures_pending"] = bool(later and r.ok)
    if out["figures_pending"]:
        _PENDING[name] = r                        # figures(name) needs the mesh and the solution
    row = r.summary()
    row.pop("folder", None)
    _ROWS[name] = row
    out["summary"] = row
    del r
    gc.collect()
    return _dumps(out)


def figures(name: str, progress: Callable[[float, str], Any]) -> str:
    """Write the figures of a run made with ``figures_later`` (the tables are
    written again, unchanged); returns every output file of the sample."""
    from otm_core.pipeline import export_run

    r = _PENDING.pop(name, None)
    if r is None:
        return _dumps({"ok": False, "error": "no run waiting for its figures"})
    t0 = time.perf_counter()
    try:
        export_run(r, OUT, progress=lambda f, m: progress(float(f), "figures: " + str(m)), figures=True)
        ok, err = True, None
    except Exception as exc:                       # the tables are there already
        ok, err = False, _plain_message("export", f"{type(exc).__name__}: {exc}")
    del r
    gc.collect()
    root = os.path.join(OUT, name)
    files = sorted(os.path.relpath(os.path.join(d, f), OUT) for d, _s, fs in os.walk(root) for f in fs)
    return _dumps({"ok": ok, "error": err, "files": files, "seconds": time.perf_counter() - t0})


# ----------------------------------------------------------------------------------------------
# moving a finished sample between workers (samples run in parallel in several workers; the
# first one holds every result for the ZIP)
# ----------------------------------------------------------------------------------------------

def pack(name: str) -> str:
    """A ZIP (stored, not compressed: fast) of the sample's outputs, view and summary row."""
    path = f"/tmp/pack_{name}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("row.json", json.dumps(_clean(_ROWS.get(name))))
        root = os.path.join(OUT, name)
        for d, _s, fs in os.walk(root):
            for f in sorted(fs):
                full = os.path.join(d, f)
                z.write(full, "out/" + os.path.relpath(full, OUT))
        if os.path.exists(view_path(name)):
            z.write(view_path(name), "view.bin")
    return path


def unpack(name: str, path: str) -> None:
    """Take over a sample packed by another worker."""
    shutil.rmtree(os.path.join(OUT, name), ignore_errors=True)
    _drop_view(name)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.filename == "row.json":
                row = json.loads(z.read(info))
                if row is not None:
                    _ROWS[name] = row
            elif info.filename == "view.bin":
                os.makedirs(VIEW_DIR, exist_ok=True)
                with open(view_path(name), "wb") as fh:
                    fh.write(z.read(info))
            elif info.filename.startswith("out/"):
                dest = os.path.join(OUT, info.filename[4:])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(z.read(info))
    os.remove(path)


def forget(name: str) -> None:
    """Remove a sample's results (its row in the summary, its output folder and its view)."""
    _ROWS.pop(name, None)
    _PENDING.pop(name, None)
    shutil.rmtree(os.path.join(OUT, name), ignore_errors=True)
    _drop_view(name)


# ----------------------------------------------------------------------------------------------
# tissue view: one binary file per sample
# ----------------------------------------------------------------------------------------------
#
#   bytes 0-3   "OTMV"     4-7   header length H (uint32, little endian)
#   8 .. 8+H    JSON header: sizes and, per array, {name: [dtype, offset, length]}
#   then the arrays, each starting at a multiple of 4 bytes
#
# Coordinates are µm in the frame of the desktop plots (origin bottom-left, y up):
# um = ld * (ndim + ss). Rings and polylines are stored flat with an offsets array
# (start of each item in points, plus the total at the end).

def view_path(name: str) -> str:
    return os.path.join(VIEW_DIR, name + ".bin")


def _drop_view(name: str) -> None:
    p = view_path(name)
    if os.path.exists(p):
        os.remove(p)


def _rings_of(f: Any) -> List[np.ndarray]:
    """Outline rings of a fibre: (N, 2) ring, list of rings, or Shapely geometry."""
    if hasattr(f, "geom_type"):
        from shapely import get_parts

        out = []
        for p in get_parts(f):
            if getattr(p, "is_empty", True) or not hasattr(p, "exterior"):
                continue
            out.append(np.asarray(p.exterior.coords)[:, :2])
        return out
    if isinstance(f, (list, tuple)) and f and np.ndim(f[0]) == 2:
        return [np.asarray(r, float) for r in f]
    a = np.asarray(f, float)
    return [a] if a.ndim == 2 and len(a) >= 3 else []


def _flat(lines: List[np.ndarray]):
    lines = [np.asarray(l, float).reshape(-1, 2) for l in lines]
    lines = [l[np.isfinite(l).all(axis=1)] for l in lines]
    lines = [l for l in lines if len(l) >= 2]
    off = np.zeros(len(lines) + 1, np.uint32)
    if lines:
        off[1:] = np.cumsum([len(l) for l in lines])
        pts = np.vstack(lines)
    else:
        pts = np.zeros((0, 2))
    return pts.astype(np.float32), off, lines


def write_view(name: str, r: Any) -> Optional[Dict[str, Any]]:
    """Write what the tissue view draws for run result ``r``; returns the header."""
    g = r.geometry
    if g is None or g.length_scale is None:
        return None
    ls, frame = g.length_scale, g.frame
    ss = np.asarray(frame.ss, float)
    to_um = lambda xy: ls.ld * (np.asarray(xy, float).reshape(-1, 2) + ss)       # noqa: E731
    arrays: Dict[str, np.ndarray] = {}
    head: Dict[str, Any] = {"name": name, "ld": float(ls.ld), "ss": ss.tolist(), "size_um": [ls.x_um, ls.y_um]}
    if g.roi is not None:
        head["roi_um"] = [g.roi.x_min_um, g.roi.x_max_um, g.roi.y_min_um, g.roi.y_max_um]
    # capillaries (the model's, after de-duplication and corrections)
    arrays["caps"] = to_um(g.capillaries).astype(np.float32)
    try:
        head["rcap_um"] = float(r.settings.transport_parameters().r_um)
    except Exception:
        head["rcap_um"] = None
    # fibres (retouched outlines), one or more rings each
    rings, owner = [], []
    for k, f in enumerate(g.fibers):
        for ring in _rings_of(f):
            rings.append(to_um(ring))
            owner.append(k)
    pts, off, kept = _flat(rings)
    arrays["fib_pts"], arrays["fib_off"] = pts, off
    own = np.asarray(owner, np.int32)
    types = np.asarray(g.fiber_types, np.int32).ravel()
    arrays["fib_owner"] = own
    arrays["fib_type"] = types if types.size == len(g.fibers) else np.zeros(len(g.fibers), np.int32)
    head["n_fibres"] = len(g.fibers)
    dm = getattr(r, "demand", None)
    if dm is not None and len(dm.lambda_fibre) == len(g.fibers) and len(g.fibers):
        arrays["fib_lambda"] = np.asarray(dm.lambda_fibre, np.float32)      # oxygen demand Λ per fibre
        head["lambda_range"] = [float(np.min(dm.lambda_fibre)), float(np.max(dm.lambda_fibre))]
        head["lambda_tissue"], head["lambda_ratio"] = float(dm.lambda_tissue), float(dm.ratio)
    # capillary domains (Voronoi), clipped to the tissue
    vor = None
    if r.morphometrics is not None:
        vor = getattr(r.morphometrics.domains, "voronoi", None)
    if vor is not None:
        from shapely.geometry import box

        x0, y0 = to_um([[-ss[0], -ss[1]]])[0]
        clip = box(0.0, 0.0, ls.x_um, ls.y_um)
        cells = []
        for c in vor.cells:
            if c is None or c.is_empty:
                continue
            ring = to_um(np.asarray(c.exterior.coords)[:, :2])
            from shapely.geometry import Polygon

            pc = Polygon(ring).intersection(clip)
            for p in getattr(pc, "geoms", [pc]):
                if hasattr(p, "exterior") and not p.is_empty:
                    cells.append(np.asarray(p.exterior.coords)[:, :2])
        arrays["vor_pts"], arrays["vor_off"], _ = _flat(cells)
    # mesh and PO2
    if r.mesh is not None:
        m = r.mesh.mesh if hasattr(r.mesh, "mesh") else r.mesh
        arrays["mesh_pts"] = to_um(m.points).astype(np.float32)
        arrays["mesh_tri"] = np.asarray(m.triangles, np.uint32).ravel()
        head["n_nodes"], head["n_triangles"] = int(m.n_nodes), int(m.n_triangles)
        if r.solution is not None:
            po2 = float(r.solution.params.raw.Pcap) * np.clip(np.asarray(r.solution.u, float), 0, None)
            arrays["po2"] = po2.astype(np.float32)          # as the PO2 map figure (negative u clipped)
            head["po2_range"] = [float(np.nanmin(po2)), float(np.nanmax(po2))]
            head["pcap"] = float(r.solution.params.raw.Pcap)
            try:
                from otm_core.export import _uptake_field

                v = np.asarray(_uptake_field(r.solution, r.settings.tissue)[0], float)
                arrays["vo2_tri"] = v.astype(np.float32)    # per triangle, ml O2/100 ml/min, as the VO2 map
                head["vo2_range"] = [float(np.nanmin(v)), float(np.nanmax(v))]
            except Exception:
                pass
    # flux lines (exact paths where stored)
    fl = r.flux_lines
    if fl is not None:
        lines = []
        paths = getattr(fl, "paths", None)
        streams = np.asarray(fl.streams)
        for i in range(streams.shape[0]):
            for j in range(streams.shape[1]):
                p = paths[i][j] if paths else streams[i, j]
                lines.append(to_um(p))
        arrays["flux_pts"], arrays["flux_off"], _ = _flat(lines)
    return _write_bin(view_path(name), head, arrays)


def _write_bin(path: str, head: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table, chunks, pos = {}, [], 0
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        dt = {np.dtype(np.float32): "f32", np.dtype(np.uint32): "u32", np.dtype(np.int32): "i32"}[a.dtype]
        b = a.tobytes()
        table[k] = [dt, pos, int(a.size)]
        chunks.append(b)
        pos += len(b)
        pad = (-pos) % 4
        if pad:
            chunks.append(b"\0" * pad)
            pos += pad
    head = dict(head, arrays=table)
    hb = json.dumps(_clean(head)).encode("utf-8")
    hb += b" " * ((-len(hb)) % 4)
    with open(path, "wb") as fh:
        fh.write(b"OTMV")
        fh.write(np.uint32(len(hb)).tobytes())
        fh.write(hb)
        for c in chunks:
            fh.write(c)
    return {k: v for k, v in head.items() if k != "arrays"}


def make_zip(names_json: str, settings_json: str) -> str:
    """ZIP of the batch: per sample its INDICES and PO2 folders, plus the batch
    summary (.xlsx and .csv) and the settings JSON. Returns the ZIP's path."""
    from otm_core.pipeline import SETTINGS_FILE, SUMMARY_FILE, write_summary

    names = [n for n in json.loads(names_json) if n in _ROWS]
    stem = os.path.join(OUT, SUMMARY_FILE)
    for ext in (".csv", ".xlsx"):
        if os.path.exists(stem + ext):
            os.remove(stem + ext)
    extra = write_summary([_ROWS[n] for n in names], stem) if names else []
    settings_path = os.path.join(OUT, SETTINGS_FILE)
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(json.loads(settings_json), fh, indent=2)
    extra.append(settings_path)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in extra:
            z.write(f, os.path.basename(f))
        for n in names:
            root = os.path.join(OUT, n)
            for dirpath, _dirs, files in os.walk(root):
                for f in sorted(files):
                    full = os.path.join(dirpath, f)
                    z.write(full, os.path.relpath(full, OUT))
    return ZIP_PATH


# ----------------------------------------------------------------------------------------------
# errors in plain words
# ----------------------------------------------------------------------------------------------

_STEP_WORDS = {"load": "Reading the file", "geometry": "Setting the tissue size and region",
               "indices": "Supply indices", "mesh": "Meshing", "po2": "PO₂ solution", "flux": "Flux lines",
               "export": "Writing the results", "session": "Saving the session"}


def _plain_error(exc: BaseException) -> str:
    msg = str(exc).strip() or type(exc).__name__
    if isinstance(exc, MemoryError):
        return "the browser ran out of memory; close other tabs or run fewer files at once"
    if "Unable to open file" in msg or "not a MATLAB" in msg.lower() or "Unknown mat file type" in msg:
        return "not a MATLAB .mat file that OTM can read"
    return msg


def _plain_message(step: str, message: str) -> str:
    text = message.split(": ", 1)[1] if ": " in message else message
    if "MemoryError" in message:
        text = "the browser ran out of memory; close other tabs or run fewer files at once"
    return f"{_STEP_WORDS.get(step, step)} failed: {text}"
