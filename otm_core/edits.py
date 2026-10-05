"""Manual corrections of a Dtect segmentation (legacy ``UpdateFibersManually`` /
``UpdateFiberTypesManually``), kept as an edit list.

Edits are stored, not applied in place: the raw data stay untouched, and the
corrected data are always ``apply_edits(ingest, edits)``. That gives undo/redo
for free, lets a session keep the corrections next to the original data, and
lets a batch run re-apply them.

Coordinates are the centred pixel frame of the ingest (``px_c``: origin at the
image centre, y up, 1 unit = 1 image pixel). It does not depend on the retouch
settings or on the tissue size, so the edits stay valid when those change.

Identity
    * Capillaries are identified by position: an edit acts on the capillary
      nearest to the recorded point (within ``tol_px``). Robust to the
      de-duplication and to edits before it in the list.
    * Fibres are identified by a stable id: the index in the data file for the
      original fibres, ``n_original + k`` for the k-th added fibre. Deleting a
      fibre does not renumber the others' ids (the exported "Fiber No." is the
      position in the corrected list, as in the legacy program).

An edit that no longer applies (e.g. removing a capillary that is not there)
is skipped and reported, never raised: a list saved with one version of the
data cannot break loading.
"""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import IngestResult, RetouchResult, RetouchSettings

__all__ = ["Edit", "EditResult", "apply_edits", "nearest_capillary", "fiber_at", "FIBRE_TYPES", "TYPE_NAMES",
           "RetouchCache", "edits_to_list", "edits_from_list"]

FIBRE_TYPES = (1, 21, 22, 0)
TYPE_NAMES = {1: "I", 21: "IIa", 22: "IIb", 0: "unknown"}
KINDS = ("add_capillary", "remove_capillary", "move_capillary", "set_fiber_type", "delete_fiber", "add_fiber",
         "replace_fiber")


@dataclass(frozen=True)
class Edit:
    """One correction. Use the constructors (``Edit.add_capillary(x, y)`` ...)."""

    kind: str
    x: Optional[float] = None
    y: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    fiber: Optional[int] = None
    fiber_type: Optional[int] = None
    ring: Optional[Tuple[Tuple[float, float], ...]] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown edit '{self.kind}'")
        if self.fiber_type is not None and int(self.fiber_type) not in FIBRE_TYPES:
            raise ValueError(f"fibre type must be one of {FIBRE_TYPES}")
        if self.ring is not None and len(self.ring) < 3:
            raise ValueError("a fibre outline needs at least 3 points")

    # ---- constructors
    @classmethod
    def add_capillary(cls, x: float, y: float) -> "Edit":
        return cls("add_capillary", float(x), float(y))

    @classmethod
    def remove_capillary(cls, x: float, y: float) -> "Edit":
        return cls("remove_capillary", float(x), float(y))

    @classmethod
    def move_capillary(cls, x: float, y: float, x2: float, y2: float) -> "Edit":
        return cls("move_capillary", float(x), float(y), float(x2), float(y2))

    @classmethod
    def set_fiber_type(cls, fiber: int, fiber_type: int) -> "Edit":
        return cls("set_fiber_type", fiber=int(fiber), fiber_type=int(fiber_type))

    @classmethod
    def delete_fiber(cls, fiber: int) -> "Edit":
        return cls("delete_fiber", fiber=int(fiber))

    @classmethod
    def add_fiber(cls, ring: Any, fiber_type: int = 0) -> "Edit":
        return cls("add_fiber", ring=_ring_tuple(ring), fiber_type=int(fiber_type))

    @classmethod
    def replace_fiber(cls, fiber: int, ring: Any) -> "Edit":
        return cls("replace_fiber", fiber=int(fiber), ring=_ring_tuple(ring))

    # ---- (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(map(list, v)) if k == "ring" and v is not None else v)
                for k, v in dataclasses.asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edit":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if kw.get("ring") is not None:
            kw["ring"] = _ring_tuple(kw["ring"])
        return cls(**kw)

    def describe(self) -> str:
        """One line for the edit list."""
        k = self.kind
        if k == "add_capillary":
            return f"Add capillary at ({self.x:.0f}, {self.y:.0f}) px"
        if k == "remove_capillary":
            return f"Remove capillary at ({self.x:.0f}, {self.y:.0f}) px"
        if k == "move_capillary":
            return f"Move capillary ({self.x:.0f}, {self.y:.0f}) → ({self.x2:.0f}, {self.y2:.0f}) px"
        if k == "set_fiber_type":
            return f"Fibre {self.fiber + 1}: type {TYPE_NAMES[self.fiber_type]}"
        if k == "delete_fiber":
            return f"Delete fibre {self.fiber + 1}"
        if k == "add_fiber":
            return f"Add fibre ({len(self.ring)} points, type {TYPE_NAMES[self.fiber_type]})"
        return f"Redraw fibre {self.fiber + 1} ({len(self.ring)} points)"


def _ring_tuple(ring: Any) -> Tuple[Tuple[float, float], ...]:
    r = np.asarray(ring, float).reshape(-1, 2)
    if len(r) > 1 and np.array_equal(r[0], r[-1]):
        r = r[:-1]
    return tuple((float(a), float(b)) for a, b in r)


def edits_to_list(edits: Sequence[Edit]) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in edits]


def edits_from_list(items: Sequence[Dict[str, Any]]) -> List[Edit]:
    return [Edit.from_dict(d) for d in items]


# ==========================================================================
# applying
# ==========================================================================


@dataclass
class EditResult:
    ingest: IngestResult                 # corrected data
    fiber_ids: np.ndarray                # stable id of each fibre of ``ingest``
    applied: List[bool]                  # per edit
    warnings: List[str] = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        return self.applied.count(False)


def nearest_capillary(caps: np.ndarray, x: float, y: float, max_dist: float) -> Optional[int]:
    """Index of the capillary nearest to (x, y), if within ``max_dist``."""
    caps = np.asarray(caps, float).reshape(-1, 2)
    if not len(caps):
        return None
    d = np.hypot(caps[:, 0] - x, caps[:, 1] - y)
    i = int(np.argmin(d))
    return i if d[i] <= max_dist else None


def fiber_at(fibers: Sequence[np.ndarray], x: float, y: float) -> Optional[int]:
    """Index of the (smallest) fibre containing (x, y)."""
    from matplotlib.path import Path

    best, best_area = None, np.inf
    for i, f in enumerate(fibers):
        r = np.asarray(f, float).reshape(-1, 2)
        if len(r) < 3:
            continue
        if r[:, 0].min() <= x <= r[:, 0].max() and r[:, 1].min() <= y <= r[:, 1].max() \
                and Path(r).contains_point((x, y)):
            area = 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1)))
            if area < best_area:
                best, best_area = i, area
    return best


def _closed(ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    r = np.asarray(ring, float)
    return np.vstack([r, r[:1]])


def apply_edits(ingest: IngestResult, edits: Sequence[Edit], tol_px: float = 1.0) -> EditResult:
    """Corrected copy of ``ingest``; the input is not modified."""
    caps = np.asarray(ingest.capillaries_pxc, float).reshape(-1, 2).copy()
    fibers = [np.asarray(f, float) for f in ingest.fibers_pxc]
    types = np.asarray(ingest.fiber_types, int).copy()
    if types.size != len(fibers):
        types = np.zeros(len(fibers), int)
    ids = list(range(len(fibers)))
    next_id = len(fibers)
    applied: List[bool] = []
    warnings: List[str] = []

    def pos(fid: Optional[int]) -> Optional[int]:
        return ids.index(fid) if fid in ids else None

    for n, e in enumerate(edits, 1):
        ok = True
        if e.kind == "add_capillary":
            caps = np.vstack([caps, [e.x, e.y]])
        elif e.kind in ("remove_capillary", "move_capillary"):
            i = nearest_capillary(caps, e.x, e.y, tol_px)
            if i is None:
                ok = False
            elif e.kind == "remove_capillary":
                caps = np.delete(caps, i, axis=0)
            else:
                caps[i] = (e.x2, e.y2)
        elif e.kind == "add_fiber":
            fibers.append(_closed(e.ring))
            types = np.append(types, int(e.fiber_type))
            ids.append(next_id)
            next_id += 1
        else:
            i = pos(e.fiber)
            if i is None:
                ok = False
            elif e.kind == "set_fiber_type":
                types[i] = int(e.fiber_type)
            elif e.kind == "delete_fiber":
                del fibers[i]
                del ids[i]
                types = np.delete(types, i)
            elif e.kind == "replace_fiber":
                fibers[i] = _closed(e.ring)
        if not ok:
            warnings.append(f"Correction {n} skipped ({e.describe()}): nothing there any more.")
        applied.append(ok)
    out = dataclasses.replace(ingest, capillaries_pxc=caps, fibers_pxc=fibers, fiber_types=types)
    return EditResult(out, np.asarray(ids, int), applied, warnings)


# ==========================================================================
# retouch with a per-fibre cache
# ==========================================================================


class RetouchCache:
    """``retouch_fibers`` that re-smooths only the fibres that changed.

    Smoothing and vertex reduction act on each outline separately, so after an
    edit only the edited outline needs work (1 s -> a few ms on the 2014
    sample). The "remove overlaps" and "separate touching fibres" options act
    on neighbours too; with either of them on, everything is recomputed.
    Thread-safe (the retouch runs on a worker thread).
    """

    def __init__(self, max_entries: int = 20000):
        self._d: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def retouch(self, settings: RetouchSettings, fibers_pxc: Sequence[np.ndarray], capillaries_pxc: np.ndarray,
                frame: Any) -> RetouchResult:
        from .geometry import as_ring, retouch_fibers, scale_geometry

        if settings.disjoint or settings.tangent:
            return retouch_fibers(settings, fibers_pxc, capillaries_pxc, frame)
        skey = (settings.smooth, settings.smooth_tol, settings.reduce, settings.reduce_tol, settings.max_vertices,
                settings.reduce_tol_step)
        rings = [as_ring(f) for f in fibers_pxc]
        keys = [(skey, r.shape, r.tobytes()) for r in rings]
        with self._lock:
            missing = [i for i, k in enumerate(keys) if k not in self._d]
        if missing:
            r = retouch_fibers(settings, [rings[i] for i in missing], np.zeros((0, 2)), frame)
            with self._lock:
                if len(self._d) + len(missing) > self.max_entries:
                    self._d.clear()
                for j, i in enumerate(missing):
                    self._d[keys[i]] = (r.smooth[j], r.reduced[j])
        with self._lock:
            got = [self._d[k] for k in keys]
            self.hits += len(keys) - len(missing)
            self.misses += len(missing)
        smooth = [g[0] for g in got]
        reduced = [g[1] for g in got]
        rescaled, caps = scale_geometry(reduced, capillaries_pxc, frame)
        return RetouchResult(smooth=smooth, reduced=reduced, disjoint=reduced, tangent=reduced, rescaled=rescaled,
                             capillaries_ndim=caps)
