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
        "steps": list(STEPS),
        "widths": WIDTH_LABELS,
        "formats": list(FORMATS),
    })


# ----------------------------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------------------------

def inspect(name: str) -> str:
    """What the desktop's Load data step shows: image size, capillaries, fibres,
    fibre types; plus the saved corrections in ``<name>.otm`` if one was added."""
    from otm_core.io import is_hdf5_mat, load_dtect_mat

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
    out = {"ok": True, "image_rows": rows, "image_cols": cols, "n_capillaries": int(len(d.capillaries_px)),
           "n_fibres": int(len(d.fibers_px)), "fibre_types": types, "notes": list(d.notes),
           "format": "MATLAB v7.3" if is_hdf5_mat(path) else "MATLAB v5", "seconds": time.perf_counter() - t0}
    del d
    gc.collect()
    return _dumps(out)


def corrections(name: str) -> str:
    """Saved corrections in ``/data/<name>.otm`` (a desktop session): count and any problem."""
    from otm_core.pipeline import find_saved_edits

    try:
        edits, meta = find_saved_edits(_data_path(name))
    except Exception as exc:
        return _dumps({"ok": False, "error": _plain_error(exc)})
    if meta is None:
        return _dumps({"ok": False, "error": "not an OTM session file"})
    return _dumps({"ok": True, "n_edits": len(edits)})


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

def run(name: str, settings_json: str, progress: Callable[[float, str], Any]) -> str:
    """Run one sample with ``run_file`` (as the desktop batch runner) and return
    what the results page shows. The big arrays are dropped before returning."""
    from otm_core.fem import mo2_statistics, po2_statistics
    from otm_core.pipeline import RunSettings, find_saved_edits, run_file
    from otm_core.tables import hypoxia_text, hypoxic_compartments, indices_rows, mo2_rows, po2_rows

    s = RunSettings.from_dict(json.loads(settings_json))
    path = _data_path(name)
    shutil.rmtree(os.path.join(OUT, name), ignore_errors=True)
    edits, src = find_saved_edits(path)
    t0 = time.perf_counter()
    r = run_file(path, s, out_dir=OUT, base_name=name, progress=lambda f, m: progress(float(f), str(m)),
                 edits=edits, edits_source=src)
    out: Dict[str, Any] = {"name": name, "ok": r.ok, "seconds": time.perf_counter() - t0,
                           "errors": [{"step": k, "message": _plain_message(k, v)} for k, v in r.errors.items()],
                           "timings": r.timings, "corrections": len(r.edits),
                           "corrections_skipped": list(r.edit_warnings)}
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
    if r.flux_lines is not None:
        sm = r.flux_lines.summary()
        out["flux"] = {"seed_capillaries": sm["n_seed_capillaries"],
                       "capillaries_with_lines": sm["n_capillaries_with_lines"],
                       "moving_lines": sm["n_moving_lines"]}
    out["files"] = sorted(os.path.relpath(f, OUT) for f in r.files)
    row = r.summary()
    row.pop("folder", None)
    _ROWS[name] = row
    out["summary"] = row
    del r
    gc.collect()
    return _dumps(out)


def forget(name: str) -> None:
    """Remove a sample's results (its row in the summary and its output folder)."""
    _ROWS.pop(name, None)
    shutil.rmtree(os.path.join(OUT, name), ignore_errors=True)


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
