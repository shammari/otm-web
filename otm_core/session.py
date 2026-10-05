"""Save and restore a whole OTM session (``.otm`` file).

A session is everything needed to come back to an analysis without recomputing
the expensive steps: the raw Dtect data, every setting chosen in the wizard,
the tissue geometry, and the mesh, PO2 solution and flux lines when they exist.

File format
-----------
A ZIP archive (open it with any unzip tool) with two members:

``session.json``
    Format name and version, when and with which program it was saved, the
    source file (path, size, modification time), tissue type, retouch settings,
    tissue size, ROI, transport parameters, model switches, mesh and flux
    settings, the random seed of the index pairings, view preferences, and which
    results are stored. Plain JSON: readable and diff-able.
``arrays.npz``
    NumPy arrays: raw Dtect data, geometry, mesh, model selection, nodal
    PO2/Pcap and flux lines. Loaded with ``allow_pickle=False``: opening a
    session someone sent you cannot run code.

What is *not* stored, and why
    * Supply indices: recomputed on loading (about a second; the random NN
      pairings use the saved seed, so the numbers are identical).
    * Ingest and retouch preview: recomputed from the stored raw data.
    * PDE coefficients and derived parameters: recomputed from the stored
      parameters, switches and mesh (cheap and exactly reproducible).

Results are only stored when they belong to the saved inputs (a solution needs
its mesh; flux lines need the solution), and :func:`load_session` checks the
array shapes so a damaged file is refused instead of showing a wrong map.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .models import ROI, Geometry, ImageFrame, LengthScale, RetouchSettings
from .progress import ProgressCallback, report

__all__ = ["Session", "save_session", "load_session", "read_session_info", "SessionError", "FORMAT", "VERSION",
           "SUFFIX", "exercise_level_name"]

FORMAT = "otm-session"
VERSION = 1
SUFFIX = ".otm"


class SessionError(ValueError):
    """The file is not an OTM session, is damaged, or was written by a newer version."""


@dataclass
class Session:
    """In-memory session. Only ``raw`` is required; every result is optional."""

    raw: Any                                          # otm_core.io.DtectData
    source_path: str = ""
    tissue: str = "skeletal"
    use_fibre_types: bool = True
    retouch_settings: RetouchSettings = field(default_factory=RetouchSettings)
    length_scale: Optional[LengthScale] = None
    roi: Optional[ROI] = None
    parameters: Any = None                            # TransportParameters
    switches: Any = None                              # ModelSwitches
    mesh_settings: Any = None                         # MeshSettings
    flux_settings: Any = None                         # FluxSettings
    index_seed: Optional[int] = 20221
    geometry: Optional[Geometry] = None
    has_indices: bool = False                         # recompute the supply indices on loading
    mesh: Any = None                                  # TissueMesh
    model: Any = None                                 # ModelGeometry
    solution: Any = None                              # PO2Solution
    flux_lines: Any = None                            # FluxLines
    view: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    edits: List[Any] = field(default_factory=list)    # otm_core.edits.Edit, applied to the ingest of ``raw``
    # filled in by load_session
    morphometrics: Any = None
    ingest: Any = None                                # with the edits applied
    original_ingest: Any = None                       # without them
    fiber_ids: Any = None                             # stable fibre ids (otm_core.edits)
    retouch: Any = None
    info: Dict[str, Any] = field(default_factory=dict)  # session.json as read (saved_at, warnings ...)

    @property
    def warnings(self) -> List[str]:
        return list(self.info.get("warnings", []))


# ==========================================================================
# packing helpers
# ==========================================================================


def _pack_rings(prefix: str, items: Sequence[Any], arrays: Dict[str, np.ndarray]) -> None:
    """Fibre-like objects -> flat arrays. Kinds: 0 ring array, 1 list of rings,
    2 Shapely geometry (WKB bytes)."""
    kinds, n_rings, ring_len, coords, wkb_len, wkb = [], [], [], [], [], []
    for f in items:
        if hasattr(f, "geom_type"):
            import shapely

            b = shapely.to_wkb(f)
            kinds.append(2)
            n_rings.append(0)
            wkb_len.append(len(b))
            wkb.append(np.frombuffer(b, dtype=np.uint8))
            continue
        rings = [f] if isinstance(f, np.ndarray) else list(f)
        kinds.append(0 if isinstance(f, np.ndarray) else 1)
        n_rings.append(len(rings))
        for r in rings:
            r = np.asarray(r, float).reshape(-1, 2)
            ring_len.append(len(r))
            coords.append(r)
    arrays[f"{prefix}_kind"] = np.asarray(kinds, np.int8)
    arrays[f"{prefix}_n_rings"] = np.asarray(n_rings, np.int64)
    arrays[f"{prefix}_ring_len"] = np.asarray(ring_len, np.int64)
    arrays[f"{prefix}_coords"] = np.concatenate(coords) if coords else np.zeros((0, 2))
    arrays[f"{prefix}_wkb_len"] = np.asarray(wkb_len, np.int64)
    arrays[f"{prefix}_wkb"] = np.concatenate(wkb) if wkb else np.zeros(0, np.uint8)


def _unpack_rings(prefix: str, a: Mapping) -> List[Any]:
    kinds, n_rings, ring_len = a[f"{prefix}_kind"], a[f"{prefix}_n_rings"], a[f"{prefix}_ring_len"]
    coords, wkb_len, wkb = a[f"{prefix}_coords"], a[f"{prefix}_wkb_len"], a[f"{prefix}_wkb"]
    if int(ring_len.sum()) != len(coords) or int(wkb_len.sum()) != len(wkb) or int(n_rings.sum()) != len(ring_len):
        raise SessionError(f"damaged outline data ({prefix})")
    out, r, c, w, wo = [], 0, 0, 0, 0
    for k, n in zip(kinds, n_rings):
        if k == 2:
            import shapely

            ln = int(wkb_len[w])
            out.append(shapely.from_wkb(wkb[wo:wo + ln].tobytes()))
            w, wo = w + 1, wo + ln
            continue
        rings = []
        for _ in range(int(n)):
            ln = int(ring_len[r])
            rings.append(coords[c:c + ln].copy())
            c += ln
            r += 1
        out.append(rings[0] if k == 0 and len(rings) == 1 else rings)
    return out


def _dc(obj: Any) -> Optional[Dict[str, Any]]:
    return None if obj is None else dataclasses.asdict(obj)


def _from_dc(typ: Any, d: Optional[Mapping[str, Any]]) -> Any:
    if d is None:
        return None
    names = {f.name for f in dataclasses.fields(typ)}
    return typ(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else f"<array {v.shape}>"
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def exercise_level_name(switches: Any, tissue: str = "skeletal") -> Optional[str]:
    """"resting" ... "high" for the standard switch sets, None for a custom one."""
    from .fem import ModelSwitches

    for level in ("resting", "low", "moderate", "high"):
        ref = ModelSwitches.cardiac(level) if tissue == "cardiac" else ModelSwitches.skeletal(
            level, switches.non_uniform, switches.differential_extraction)
        if (ref.michaelis_menten, ref.myoglobin, float(ref.exercise_level)) == \
                (switches.michaelis_menten, switches.myoglobin, float(switches.exercise_level)):
            return level
    return None


# ==========================================================================
# save
# ==========================================================================


def _program_version() -> str:
    try:
        from . import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "dev"


def save_session(path: str, s: Session, progress: Optional[ProgressCallback] = None) -> str:
    """Write ``s`` to ``path`` (``.otm`` added if missing). Written to a temporary
    file first and then renamed, so an interrupted save never destroys an older
    session of the same name."""
    if not path.lower().endswith(SUFFIX):
        path += SUFFIX
    if s.raw is None:
        raise ValueError("a session needs the raw data")
    if s.solution is not None and s.mesh is None:
        raise ValueError("a PO2 solution cannot be saved without its mesh")
    if s.flux_lines is not None and s.solution is None:
        raise ValueError("flux lines cannot be saved without the PO2 solution")
    report(progress, 0.05, "Collecting arrays")
    A: Dict[str, np.ndarray] = {}
    raw = s.raw
    A["raw_capillaries_px"] = np.asarray(raw.capillaries_px, float)
    _pack_rings("raw_fibers", list(raw.fibers_px), A)
    if raw.fiber_types is not None:
        A["raw_fiber_types"] = np.asarray(raw.fiber_types, np.int64)
    stored: List[str] = []
    g = s.geometry
    if g is not None:
        A["geo_capillaries"] = np.asarray(g.capillaries, float)
        _pack_rings("geo_fibers", list(g.fibers), A)
        A["geo_fiber_types"] = np.asarray(g.fiber_types, np.int64)
        stored.append("geometry")
    m = s.mesh
    if m is not None:
        for k in ("points", "triangles", "cell_region", "fiber_types", "capillary_facets", "capillary_facet_owner",
                  "box_facets", "capillaries", "rcap", "box"):
            A[f"mesh_{k}"] = np.asarray(getattr(m, k))
        stored.append("mesh")
    md = s.model
    if md is not None:
        for k in ("box", "capillaries", "rcap", "fiber_types", "capillary_index", "fiber_index"):
            A[f"model_{k}"] = np.asarray(getattr(md, k))
        _pack_rings("model_fibers", list(md.fibers), A)
        stored.append("model")
    sol = s.solution
    if sol is not None:
        A["sol_u"] = np.asarray(sol.u, float)
        A["sol_residuals"] = np.asarray(sol.residual_history, float)
        stored.append("solution")
    fl = s.flux_lines
    if fl is not None:
        for k in ("streams", "seeds", "capillary_index", "n_steps", "arc_parameter"):
            A[f"flux_{k}"] = np.asarray(getattr(fl, k))
        A["flux_stop_reason"] = np.asarray(fl.stop_reason, dtype=str)
        if fl.paths:
            lens = [[len(p) for p in row] for row in fl.paths]
            A["flux_path_len"] = np.asarray(lens, np.int64)
            A["flux_path_coords"] = np.concatenate([np.asarray(p, float).reshape(-1, 2) for row in fl.paths
                                                    for p in row]) if any(map(any, lens)) else np.zeros((0, 2))
        stored.append("flux")

    meta: Dict[str, Any] = {
        "format": FORMAT, "version": VERSION, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "program": f"otm {_program_version()}", "notes": s.notes,
        "source": _source_info(s.source_path or getattr(raw, "path", "")),
        "raw": {"image_size": list(map(int, raw.image_size)), "has_fiber_types": raw.fiber_types is not None,
                "n_capillaries": int(len(raw.capillaries_px)), "n_fibres": int(len(raw.fibers_px)),
                "notes": list(getattr(raw, "notes", []))},
        "edits": [e.to_dict() for e in s.edits],
        "tissue": s.tissue, "use_fibre_types": bool(s.use_fibre_types),
        "retouch_settings": _dc(s.retouch_settings),
        "length_scale": None if s.length_scale is None else [s.length_scale.x_um, s.length_scale.y_um],
        "roi": None if s.roi is None else [s.roi.x_min_um, s.roi.x_max_um, s.roi.y_min_um, s.roi.y_max_um],
        "parameters": _dc(s.parameters), "switches": _dc(s.switches),
        "mesh_settings": _dc(s.mesh_settings), "flux_settings": _dc(s.flux_settings),
        "index_seed": s.index_seed, "has_indices": bool(s.has_indices), "view": _json_safe(s.view),
        "stored": stored,
        "geometry": None if g is None else {
            "image_size": list(map(int, g.frame.image_size)),
            "length_scale": None if g.length_scale is None else [g.length_scale.x_um, g.length_scale.y_um],
            "roi": None if g.roi is None else [g.roi.x_min_um, g.roi.x_max_um, g.roi.y_min_um, g.roi.y_max_um]},
        "mesh": None if m is None else {"info": _json_safe(m.info), "n_nodes": m.n_nodes,
                                        "n_triangles": m.n_triangles},
        "model": None if md is None else {"n_unknown_types": int(md.n_unknown_types)},
        "solution": None if sol is None else {"iterations": int(sol.iterations), "converged": bool(sol.converged),
                                              "seconds": float(sol.seconds), "tissue": s.tissue},
        "flux": None if fl is None else {"settings": _dc(fl.settings), "seed_radius": _num(fl.seed_radius),
                                         "max_length": _num(fl.max_length), "has_paths": bool(fl.paths)},
    }
    report(progress, 0.2, "Compressing")
    buf = io.BytesIO()
    np.savez_compressed(buf, **A)
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps(meta, indent=2, allow_nan=True))
        z.writestr(zipfile.ZipInfo("arrays.npz", time.localtime()[:6]), buf.getvalue(),
                   compress_type=zipfile.ZIP_STORED)               # already compressed
    os.replace(tmp, path)
    report(progress, 1.0, "Saved")
    return path


def _num(v: Any) -> Optional[float]:
    v = float(v)
    return v if math.isfinite(v) else None


def _source_info(p: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {"path": os.path.abspath(p) if p else "", "name": os.path.basename(p)}
    if p and os.path.isfile(p):
        st = os.stat(p)
        d.update(size=int(st.st_size), mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)))
    return d


# ==========================================================================
# load
# ==========================================================================


def read_session_info(path: str) -> Dict[str, Any]:
    """Only ``session.json`` (fast; for a file dialog preview or a listing)."""
    try:
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("session.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise SessionError(f"{os.path.basename(path)} is not an OTM session ({exc})") from None
    if meta.get("format") != FORMAT:
        raise SessionError(f"{os.path.basename(path)} is not an OTM session")
    if int(meta.get("version", 0)) > VERSION:
        raise SessionError(f"{os.path.basename(path)} was saved by a newer OTM (session format "
                           f"{meta['version']}; this program reads up to {VERSION})")
    return meta


def load_session(path: str, compute_indices: bool = True, rebuild_preview: bool = True,
                 progress: Optional[ProgressCallback] = None) -> Session:
    """Read a ``.otm`` file. Recomputes the ingest, the retouch preview
    (``rebuild_preview``) and the supply indices (``compute_indices``, when they
    were part of the saved session)."""
    from .fem import MeshSettings, ModelSwitches, TransportParameters, compartment_coefficients
    from .fem.mesh import ModelGeometry, TissueMesh
    from .fem.solve import PO2Solution
    from .flux import FluxLines, FluxSettings
    from .io import DtectData

    meta = read_session_info(path)
    report(progress, 0.05, "Reading arrays")
    try:
        with zipfile.ZipFile(path) as z:
            a = dict(np.load(io.BytesIO(z.read("arrays.npz")), allow_pickle=False))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise SessionError(f"damaged session file ({exc})") from None
    warnings: List[str] = []

    def need(*keys: str) -> None:
        missing = [k for k in keys if k not in a]
        if missing:
            raise SessionError(f"damaged session file (missing {', '.join(missing)})")

    need("raw_capillaries_px")
    src = meta.get("source", {})
    raw = DtectData(capillaries_px=a["raw_capillaries_px"], fibers_px=_unpack_rings("raw_fibers", a),
                    fiber_types=a.get("raw_fiber_types"), image_size=tuple(meta["raw"]["image_size"]),
                    path=src.get("path", ""), notes=list(meta["raw"].get("notes", [])))
    tissue = meta.get("tissue", "skeletal")
    s = Session(raw=raw, source_path=src.get("path", ""), tissue=tissue,
                use_fibre_types=bool(meta.get("use_fibre_types", True)),
                retouch_settings=_from_dc(RetouchSettings, meta.get("retouch_settings")) or RetouchSettings(),
                length_scale=None if meta.get("length_scale") is None else LengthScale(*meta["length_scale"]),
                roi=None if meta.get("roi") is None else ROI(*meta["roi"]),
                parameters=_from_dc(TransportParameters, meta.get("parameters")),
                switches=_from_dc(ModelSwitches, meta.get("switches")),
                mesh_settings=_from_dc(MeshSettings, meta.get("mesh_settings")),
                flux_settings=_from_dc(FluxSettings, meta.get("flux_settings")),
                index_seed=meta.get("index_seed"), has_indices=bool(meta.get("has_indices")),
                view=dict(meta.get("view") or {}), notes=meta.get("notes", ""), info=meta)
    if src.get("path") and os.path.isfile(src["path"]) and src.get("size") is not None \
            and os.path.getsize(src["path"]) != src["size"]:
        warnings.append(f"The data file {src.get('name')} has changed since the session was saved; "
                        "the session uses the data stored in it.")

    stored = set(meta.get("stored", []))
    gm = meta.get("geometry")
    if "geometry" in stored:
        need("geo_capillaries", "geo_fiber_types")
        s.geometry = Geometry(ImageFrame(tuple(gm["image_size"])), a["geo_capillaries"],
                              _unpack_rings("geo_fibers", a), a["geo_fiber_types"],
                              None if gm.get("length_scale") is None else LengthScale(*gm["length_scale"]),
                              None if gm.get("roi") is None else ROI(*gm["roi"]))
    if "mesh" in stored:
        keys = ("points", "triangles", "cell_region", "fiber_types", "capillary_facets", "capillary_facet_owner",
                "box_facets", "capillaries", "rcap", "box")
        need(*(f"mesh_{k}" for k in keys))
        s.mesh = TissueMesh(**{k: a[f"mesh_{k}"] for k in keys}, info=dict((meta.get("mesh") or {}).get("info", {})))
        if s.mesh.triangles.max(initial=-1) >= s.mesh.n_nodes or len(s.mesh.cell_region) != s.mesh.n_triangles:
            raise SessionError("damaged session file (mesh)")
    if "model" in stored:
        keys = ("box", "capillaries", "rcap", "fiber_types", "capillary_index", "fiber_index")
        need(*(f"model_{k}" for k in keys))
        s.model = ModelGeometry(**{k: a[f"model_{k}"] for k in keys}, fibers=_unpack_rings("model_fibers", a),
                                n_unknown_types=int((meta.get("model") or {}).get("n_unknown_types", 0)))
    if "solution" in stored:
        need("sol_u")
        if s.mesh is None or s.parameters is None or s.switches is None or s.geometry is None \
                or s.geometry.length_scale is None:
            raise SessionError("damaged session file (solution without its mesh or inputs)")
        if len(a["sol_u"]) != s.mesh.n_nodes:
            raise SessionError("damaged session file (solution does not match the mesh)")
        report(progress, 0.3, "Restoring the PO2 solution")
        params = s.parameters.derive(len(s.geometry.capillaries), s.geometry.length_scale)
        info = meta.get("solution") or {}
        s.solution = PO2Solution(mesh=s.mesh, u=a["sol_u"], params=params, switches=s.switches,
                                 coefficients=compartment_coefficients(s.mesh, params, s.switches, tissue),
                                 iterations=int(info.get("iterations", 0)),
                                 residual_history=[float(x) for x in a.get("sol_residuals", [])],
                                 converged=bool(info.get("converged", True)), seconds=float(info.get("seconds", 0)))
    if "flux" in stored:
        need("flux_streams", "flux_seeds", "flux_capillary_index", "flux_stop_reason", "flux_n_steps",
             "flux_arc_parameter")
        if s.solution is None:
            raise SessionError("damaged session file (flux lines without a solution)")
        fm = meta.get("flux") or {}
        st = a["flux_streams"]
        paths: List[List[np.ndarray]] = []
        if "flux_path_len" in a:
            lens, coords, k = a["flux_path_len"], a["flux_path_coords"], 0
            if int(lens.sum()) != len(coords):
                raise SessionError("damaged session file (flux paths)")
            for row in lens:
                cur = []
                for n in row:
                    cur.append(coords[k:k + int(n)].copy())
                    k += int(n)
                paths.append(cur)
        nan = float("nan")
        s.flux_lines = FluxLines(streams=st, seeds=a["flux_seeds"], capillary_index=a["flux_capillary_index"],
                                 stop_reason=a["flux_stop_reason"], n_steps=a["flux_n_steps"],
                                 arc_parameter=a["flux_arc_parameter"],
                                 settings=_from_dc(FluxSettings, fm.get("settings")) or FluxSettings(), paths=paths,
                                 seed_radius=nan if fm.get("seed_radius") is None else fm["seed_radius"],
                                 max_length=nan if fm.get("max_length") is None else fm["max_length"])

    # ---- recomputed parts
    from . import ingest_dtect, retouch_fibers

    report(progress, 0.5, "Rebuilding the data view")
    skeletal = tissue == "skeletal"
    types = raw.fiber_types if (skeletal and s.use_fibre_types) else None
    s.original_ingest = ingest_dtect(raw.capillaries_px, raw.fibers_px if skeletal else [], raw.image_size, types)
    from .edits import apply_edits, edits_from_list

    try:
        s.edits = edits_from_list(meta.get("edits") or [])
    except (TypeError, ValueError) as exc:
        raise SessionError(f"damaged session file (corrections: {exc})") from None
    er = apply_edits(s.original_ingest, s.edits)
    s.ingest, s.fiber_ids = er.ingest, er.fiber_ids
    warnings += er.warnings
    if rebuild_preview:
        s.retouch = retouch_fibers(s.retouch_settings, s.ingest.fibers_pxc, s.ingest.capillaries_pxc, s.ingest.frame)
    if compute_indices and s.has_indices and s.geometry is not None and s.geometry.length_scale is not None:
        from . import get_morphometric_data

        report(progress, 0.75, "Recomputing the supply indices")
        s.morphometrics = get_morphometric_data(s.geometry, rng=np.random.default_rng(s.index_seed),
                                                emit_warnings=False)
    s.info = dict(meta, warnings=warnings)
    report(progress, 1.0, "Session loaded")
    return s
