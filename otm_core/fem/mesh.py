"""Conforming, graded triangular meshes of a muscle cross-section (gmsh).

Replaces the legacy PDE-Toolbox geometry/mesh chain::

    SkeletalFunctionalHeterogeneity  (model box, RemoveCapillaryFibreOverlap,
                                      GetModelCapillaries / GetModelFibers)
    GetGeometryAndMeshData           (SkeletalDGMBox / decsg, initmesh)
    MinimalRegionToFiberMap          (sub-domain -> fibre, ">99 % of nodes" rule)

Pipeline
--------
1. :func:`prepare_model_geometry` reproduces the legacy model selection in the
   non-dimensional frame: box, capillaries inside it, fibres with the capillary
   exclusion zone removed, fibre types (unknown -> IIb, as the legacy code does).
2. :func:`plan_regions` turns the polygons into a planar partition with Shapely:
   an area covered by two fibres is given to one of them
   (:func:`resolve_fiber_overlaps`; by default to the fibre MATLAB picks, i.e.
   the first in its ``FiberMatrix`` order), everything else inside the box is
   interstitial space (IS), and each capillary disc is a hole.
3. :func:`build_tissue_mesh` writes that partition into gmsh's built-in
   ("geo") kernel. Shared boundaries are single gmsh curves, so the mesh is
   conforming across fibre/IS interfaces without any Boolean operation in gmsh
   (the OpenCASCADE kernel is not needed). Capillaries are true circles (four
   circle arcs) carved out of the surface that contains them. A
   Distance + Threshold background field grades the element size from
   ``h_wall`` on capillary walls to ``h_max`` far away.

Every triangle carries its region (``-1`` = IS, ``k`` = model fibre ``k``) and
every boundary facet is labelled (capillary id or box), so the solver needs no
geometric search. Physical groups are also written, so ``.msh`` exports keep the
tags: surfaces ``IS`` (1) and ``fiber_k`` (1000 + k); curves ``box`` (2) and
``capillary_c`` (100000 + c).

All coordinates are in the non-dimensional frame of the legacy model
(``ndim = px_c / (min(ImageSize)/2)``; lengths in units of Ly/2).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely

from ..geometry import (
    fiber_geometries,
    is_closed,
    is_multipart,
    ndim_box,
    remove_capillary_fibre_overlap,
    select_model_capillaries,
    select_model_fibers,
)
from ..progress import ProgressCallback, report, scaled
from ..models import FIBER_TYPE_IIB, FIBER_TYPE_UNKNOWN, Geometry

IS_REGION = -1
"""Region id of the interstitial space."""

COMPARTMENT_IS = 0
"""Compartment code of the interstitial space (fibres use their type code 1/21/22)."""

PHYS_IS = 1
PHYS_BOX = 2
PHYS_FIBER0 = 1000
PHYS_CAPILLARY0 = 100000


# --------------------------------------------------------------------------
# Model geometry (SkeletalFunctionalHeterogeneity)
# --------------------------------------------------------------------------


@dataclass
class ModelGeometry:
    """Inputs of the FEM model in the ndim frame."""

    box: np.ndarray                    # (5, 2) closed rectangle (NdimBox)
    capillaries: np.ndarray            # (C, 2) capillary centres inside the box
    rcap: np.ndarray                   # (C,) capillary radii (ndim)
    fibers: List[Any]                  # rings / lists of rings / Shapely polygons
    fiber_types: np.ndarray            # (F,) 1, 21, 22 (unknown already mapped to 22)
    capillary_index: np.ndarray        # 0-based index into the source capillary list
    fiber_index: np.ndarray            # 0-based index into the source fibre list
    n_unknown_types: int = 0


def prepare_model_geometry(geometry: Geometry, rcap: float,
                           remove_overlap: bool = True) -> ModelGeometry:
    """Legacy model selection (SkeletalFunctionalHeterogeneity_OpeningFcn).

    ``rcap`` is the non-dimensional capillary radius (``parameter.Rcap`` =
    r / (Ly/2)); see :func:`otm_core.fem.solve.TransportParameters.derive`.
    Exact duplicate capillaries are dropped (RemoveIndistinguishableCapillaries
    with tolerance 0).
    """
    if geometry.length_scale is None:
        raise ValueError("geometry.length_scale must be set")
    box = ndim_box(geometry.length_scale)
    caps_all = np.asarray(geometry.capillaries, dtype=float)
    fibers = list(geometry.fibers)
    if fibers and remove_overlap:
        _, fibers = remove_capillary_fibre_overlap(fibers, caps_all, rcap)
    caps, cap_idx = select_model_capillaries(caps_all, box)
    _, first = np.unique(caps, axis=0, return_index=True)        # exact duplicates
    keep = np.sort(first)
    caps, cap_idx = caps[keep], cap_idx[keep]
    model_fibers, fib_idx = select_model_fibers(fibers, box)
    types = np.asarray(geometry.fiber_types, dtype=int)
    types = types[fib_idx] if types.size else np.full(len(fib_idx), FIBER_TYPE_IIB, dtype=int)
    n_unknown = int(np.count_nonzero(types == FIBER_TYPE_UNKNOWN))
    types = np.where(types == FIBER_TYPE_UNKNOWN, FIBER_TYPE_IIB, types)
    return ModelGeometry(box=box, capillaries=caps, rcap=np.full(len(caps), float(rcap)),
                         fibers=model_fibers, fiber_types=types, capillary_index=cap_idx,
                         fiber_index=fib_idx, n_unknown_types=n_unknown)


# --------------------------------------------------------------------------
# Planar partition (Shapely)
# --------------------------------------------------------------------------


@dataclass
class RegionPlan:
    """Planar partition handed to gmsh."""

    faces: List[Any]                   # Shapely Polygons (with holes), noded consistently
    face_region: np.ndarray            # region id per face (-1 IS, k fibre)
    circle_caps: np.ndarray            # capillaries meshed as true circles
    circle_face: np.ndarray            # face index containing each circle capillary
    polygon_caps: np.ndarray           # capillaries cut as polygons (touch a fibre / the box)
    polygon_cap_shapes: List[Any]      # their discs (same order as polygon_caps)
    box: np.ndarray
    capillaries: np.ndarray
    rcap: np.ndarray
    fiber_types: np.ndarray
    dropped_fibre_area: float = 0.0    # fibre area reassigned by overlap resolution (ndim^2)
    overlap_rule: str = "matlab"
    overlap_transfers: List[Dict[str, Any]] = field(default_factory=list)  # {"from", "to", "area"}
    notes: List[str] = field(default_factory=list)


def _polys(g) -> List[Any]:
    return [p for p in shapely.get_parts(g) if shapely.get_type_id(p) == 3 and not p.is_empty]


OVERLAP_RULES = ("matlab", "index", "larger", "split")
"""How an area covered by two fibres is assigned (see :func:`resolve_fiber_overlaps`)."""


def _unique_vertex_count(f) -> int:
    """``numel(F.x(1:end-1))`` of the legacy fibre struct (closing vertex not counted)."""
    if _is_shapely(f):
        return sum(len(p.exterior.coords) - 1 for p in _polys(f))
    if is_multipart(f):
        return sum(_unique_vertex_count(np.asarray(r)) for r in f)
    r = np.asarray(f)
    return len(r) - 1 if is_closed(r) else len(r)


def _is_shapely(o) -> bool:
    return isinstance(o, shapely.Geometry)


def matlab_fiber_matrix_order(fibers: Sequence[Any]) -> np.ndarray:
    """Column order of the legacy ``FiberMatrix``.

    ``ConvertFiberBoundaryDataFromStrctureToMatrix`` sorts the fibres with
    ``sortrows(H', 2)``, i.e. a *stable* sort by the number of unique vertices.
    ``MinimalRegionToFiberMap`` then scans the fibres in this order and gives
    every sub-domain to the first fibre that contains it, so this is the order
    in which MATLAB awards overlapping areas. (Verified against
    ``fibers_in_matrix_index`` of the baseline, ties included.)
    """
    counts = np.array([_unique_vertex_count(f) for f in fibers], dtype=int)
    return np.argsort(counts, kind="stable")


def _shared_length(a, b) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    return float(shapely.length(shapely.intersection(shapely.boundary(a), shapely.buffer(shapely.boundary(b), 1e-12))))


def _split_lens(lens, gi, gj, own_i, own_j):
    """Split one overlap component of fibres i and j along the chord joining the two
    points where their outlines cross. Returns (part_for_i, part_for_j)."""
    from shapely.ops import split

    cross = shapely.intersection(shapely.boundary(gi), shapely.boundary(gj))
    pts = shapely.get_coordinates(cross)
    if len(pts):
        on = shapely.distance(shapely.points(pts), shapely.boundary(lens)) <= 1e-9
        pts = np.unique(np.round(pts[on], 12), axis=0)
    if len(pts) == 2:
        a, b = pts
        d = b - a
        if np.hypot(*d) > 0:
            span = 10.0 * (np.hypot(*d) + np.sqrt(lens.area))
            u = d / np.hypot(*d)
            cut = shapely.linestrings([a - span * u, b + span * u])
            parts = [q for q in shapely.get_parts(split(lens, cut)) if q.area > 0]
            if len(parts) >= 2:
                to_i = [q for q in parts if _shared_length(q, own_i) >= _shared_length(q, own_j)]
                to_j = [q for q in parts if q not in to_i]
                return shapely.union_all(to_i) if to_i else shapely.Polygon(), \
                    shapely.union_all(to_j) if to_j else shapely.Polygon()
    # no clean chord (nested fibre, several crossings): give it to the fibre it borders most
    if _shared_length(lens, own_i) >= _shared_length(lens, own_j):
        return lens, shapely.Polygon()
    return shapely.Polygon(), lens


def resolve_fiber_overlaps(geoms: Sequence[Any], rule: str = "matlab", fibers: Optional[Sequence[Any]] = None,
                           min_area: float = 1e-12):
    """Make fibre polygons disjoint. Returns ``(pieces, transfers)``.

    ``rule``:

    * ``"matlab"`` (default): the overlap goes to the fibre that comes first in
      the legacy ``FiberMatrix`` order (:func:`matlab_fiber_matrix_order`),
      reproducing MATLAB's choice. ``fibers`` (the raw rings) are needed to count
      vertices exactly as MATLAB does; ``geoms`` are used when omitted.
    * ``"index"``: to the lower-index fibre.
    * ``"larger"``: to the fibre with the larger area.
    * ``"split"``: each overlap is cut along the chord between the two points
      where the outlines cross; each fibre keeps the half next to its own
      territory (falls back to "the fibre it borders most" when there is no
      single chord). Any area still shared by three or more fibres goes to the
      lower-index fibre.

    ``transfers`` lists ``{"from": loser, "to": winner, "area": ndim^2}``.
    """
    if rule not in OVERLAP_RULES:
        raise ValueError(f"overlap rule must be one of {OVERLAP_RULES}")
    g = [x if not x.is_empty else shapely.Polygon() for x in geoms]
    n = len(g)
    transfers: List[Dict[str, Any]] = []
    tree = shapely.STRtree(g)
    a_idx, b_idx = tree.query(g, predicate="intersects")
    pairs = sorted({(int(i), int(j)) for i, j in zip(a_idx, b_idx) if i < j})
    pairs = [(i, j) for i, j in pairs if shapely.intersection(g[i], g[j]).area > min_area]

    if rule == "split":
        P = list(g)
        for i, j in pairs:
            L = shapely.intersection(P[i], P[j])
            if L.area <= min_area:
                continue
            own_i, own_j = shapely.difference(P[i], P[j]), shapely.difference(P[j], P[i])
            for lens in _polys(L):
                if lens.area <= min_area:
                    continue
                pi, pj = _split_lens(lens, g[i], g[j], own_i, own_j)
                if not pj.is_empty:
                    P[i] = shapely.difference(P[i], pj)
                    transfers.append({"from": i, "to": j, "area": float(pj.area)})
                if not pi.is_empty:
                    P[j] = shapely.difference(P[j], pi)
                    transfers.append({"from": j, "to": i, "area": float(pi.area)})
        order = np.arange(n)                       # leftovers (3-way overlaps): index order
        g = P
    elif rule == "matlab":
        order = matlab_fiber_matrix_order(fibers if fibers is not None else g)
    elif rule == "index":
        order = np.arange(n)
    else:                                          # "larger"
        order = np.argsort([-x.area for x in g], kind="stable")

    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    pieces: List[Any] = [shapely.Polygon()] * n
    taken = shapely.Polygon()
    for k in order:
        gk = g[k]
        if gk.is_empty:
            continue
        own = shapely.difference(gk, taken)
        if rule != "split":
            for i, j in pairs:
                if k in (i, j):
                    other = j if k == i else i
                    if rank[other] < rank[k]:
                        lost = shapely.intersection(gk, g[other]).area
                        if lost > min_area:
                            transfers.append({"from": int(k), "to": int(other), "area": float(lost)})
        taken = shapely.union(taken, gk)
        pieces[k] = shapely.union_all([p for p in _polys(own) if p.area > min_area]) if not own.is_empty \
            else shapely.Polygon()
    return pieces, transfers


def plan_regions(model: ModelGeometry, polygon_segments: int = 48,
                 min_face_area: float = 1e-12, clearance: float = 1e-9,
                 snap: float = 2e-4, overlap_rule: str = "matlab") -> RegionPlan:
    """Planar partition of the model box into IS and fibre faces with capillary holes.

    * Fibres are clipped to the box; areas covered by two fibres are assigned
      with ``overlap_rule`` (default ``"matlab"``: the fibre MATLAB would pick;
      see :func:`resolve_fiber_overlaps`). Reassigned areas are listed in
      ``RegionPlan.overlap_transfers`` and summarised in ``notes``.
    * A capillary whose disc lies strictly inside one face (the usual case,
      thanks to the legacy exclusion zone) becomes a true circular hole.
      Discs touching a fibre or the box are subtracted as ``polygon_segments``-gons.
    * Fibre pieces are snapped to a ``snap`` grid (default 2e-4 ndim, i.e.
      0.03 um for Ly = 330 um, a tenth of a pixel) so that near-coincident
      vertices left by polygon clipping do not create needle triangles.
      Snapping is applied identically to shared boundaries, so it never opens
      gaps between neighbouring fibres. ``snap=0`` disables it.
    * All face boundaries are noded together, so neighbouring faces share
      identical vertices and segments.
    """
    notes: List[str] = []
    box_poly = shapely.Polygon(model.box)
    caps = np.asarray(model.capillaries, dtype=float).reshape(-1, 2)
    rcap = np.asarray(model.rcap, dtype=float).ravel()
    if rcap.size == 1 and len(caps) != 1:
        rcap = np.full(len(caps), rcap[0])

    # 1. fibres -> disjoint pieces
    geoms = [shapely.intersection(x, box_poly) if not x.is_empty else x
             for x in fiber_geometries(model.fibers)]
    pieces, transfers = resolve_fiber_overlaps(geoms, overlap_rule, model.fibers, min_face_area)
    lost = float(sum(t["area"] for t in transfers))
    if transfers:
        pairs = len({(t["from"], t["to"]) for t in transfers})
        notes.append(f"{pairs} fibre overlap(s) resolved with rule '{overlap_rule}' "
                     f"({lost:.3e} ndim^2 reassigned)")
    if snap > 0:
        pieces = [shapely.set_precision(p, snap) if not p.is_empty else p for p in pieces]
        pieces = [shapely.union_all([q for q in _polys(p) if q.area > min_face_area])
                  if not p.is_empty else p for p in pieces]

    # 2. capillaries: circle holes when clear of fibres and box edges
    discs = shapely.buffer(shapely.points(caps), rcap, quad_segs=max(polygon_segments // 4, 2))
    fib_union = shapely.union_all([p for p in pieces if not p.is_empty]) if pieces else shapely.Polygon()
    inner_box = shapely.Polygon(model.box).buffer(-clearance)
    circ = np.array([shapely.contains(inner_box, shapely.Point(c).buffer(r + clearance, 64))
                     and shapely.distance(shapely.Point(c), fib_union) > r + clearance
                     if not fib_union.is_empty else
                     shapely.contains(inner_box, shapely.Point(c).buffer(r + clearance, 64))
                     for c, r in zip(caps, rcap)], dtype=bool)
    poly_caps = np.flatnonzero(~circ)
    if poly_caps.size:
        cut = shapely.union_all(discs[poly_caps])
        pieces = [shapely.difference(p, cut) if not p.is_empty else p for p in pieces]
        notes.append(f"{poly_caps.size} capillaries touch a fibre or the box: cut as "
                     f"{polygon_segments}-gons")
    else:
        cut = shapely.Polygon()

    # 3. interstitial space
    is_area = shapely.difference(box_poly, shapely.union_all([fib_union, cut]))

    # 4. node all boundaries together and polygonize
    labelled = [(IS_REGION, p) for p in _polys(is_area)]
    for k, p in enumerate(pieces):
        labelled += [(k, q) for q in _polys(p)]
    lines = shapely.union_all([shapely.boundary(p) for _, p in labelled])
    faces_all = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    tree = shapely.STRtree([p for _, p in labelled])
    faces, region = [], []
    for f in faces_all:
        if f.area <= min_face_area:
            continue
        rp = f.representative_point()
        hit = tree.query(rp, predicate="within")
        if hit.size == 0:
            continue                                   # a polygonal capillary hole
        faces.append(f)
        region.append(labelled[int(hit[0])][0])
    region_arr = np.asarray(region, dtype=int)

    # 5. which face holds each circular capillary
    ftree = shapely.STRtree(faces)
    circle_caps = np.flatnonzero(circ)
    circle_face = np.full(circle_caps.size, -1, dtype=int)
    if circle_caps.size:
        qi, fi = ftree.query(shapely.points(caps[circle_caps]), predicate="within")
        circle_face[qi] = fi
        if (circle_face < 0).any():
            raise RuntimeError("a capillary centre is not inside any face")
        bad = region_arr[circle_face] != IS_REGION
        if bad.any():
            notes.append(f"{int(bad.sum())} capillaries lie inside a fibre face")
    return RegionPlan(faces=faces, face_region=region_arr, circle_caps=circle_caps,
                      circle_face=circle_face, polygon_caps=poly_caps,
                      polygon_cap_shapes=list(discs[poly_caps]), box=np.asarray(model.box),
                      capillaries=caps, rcap=rcap, fiber_types=np.asarray(model.fiber_types),
                      dropped_fibre_area=lost, overlap_rule=overlap_rule,
                      overlap_transfers=transfers, notes=notes)


# --------------------------------------------------------------------------
# Mesh container
# --------------------------------------------------------------------------


@dataclass
class TissueMesh:
    """Linear triangular mesh with region and boundary labels (ndim frame)."""

    points: np.ndarray                 # (N, 2)
    triangles: np.ndarray              # (M, 3) 0-based, counter-clockwise
    cell_region: np.ndarray            # (M,) -1 IS, k = model fibre k
    fiber_types: np.ndarray            # (F,) type code of every model fibre
    capillary_facets: np.ndarray       # (K, 2) node pairs on capillary walls
    capillary_facet_owner: np.ndarray  # (K,) capillary index of each wall facet
    box_facets: np.ndarray             # (B, 2) node pairs on the outer box
    capillaries: np.ndarray            # (C, 2)
    rcap: np.ndarray                   # (C,)
    box: np.ndarray                    # (5, 2)
    info: Dict[str, Any] = field(default_factory=dict)

    # ---- derived quantities
    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def cell_areas(self) -> np.ndarray:
        p = self.points[self.triangles]
        return 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

    def cell_compartment(self) -> np.ndarray:
        """Per-triangle compartment code: 0 = IS, 1 = I, 21 = IIa, 22 = IIb."""
        out = np.full(self.n_triangles, COMPARTMENT_IS, dtype=int)
        fib = self.cell_region >= 0
        out[fib] = self.fiber_types[self.cell_region[fib]]
        return out

    def quality(self) -> Dict[str, float]:
        """Radius-ratio quality (1 = equilateral): min and mean."""
        p = self.points[self.triangles]
        a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
        b = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
        c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
        s = (a + b + c) / 2
        area = self.cell_areas()
        q = 2 * (area / s) / (a * b * c / (4 * area))      # 2 r_in / r_circ
        return {"min": float(q.min()), "mean": float(q.mean())}

    def to_skfem(self):
        """``skfem.MeshTri`` (same node and element order)."""
        from skfem import MeshTri
        return MeshTri(self.points.T.copy(), self.triangles.T.copy())

    def save(self, path: str) -> None:
        """Cache the mesh as ``.npz`` (replaces the legacy ``Mesh.mat`` cache)."""
        np.savez_compressed(path, points=self.points, triangles=self.triangles,
                            cell_region=self.cell_region, fiber_types=self.fiber_types,
                            capillary_facets=self.capillary_facets,
                            capillary_facet_owner=self.capillary_facet_owner,
                            box_facets=self.box_facets, capillaries=self.capillaries,
                            rcap=self.rcap, box=self.box)

    @classmethod
    def load(cls, path: str) -> "TissueMesh":
        d = np.load(path)
        return cls(**{k: d[k] for k in d.files})

    # ---- legacy PDE Toolbox meshes (for regression tests)
    @classmethod
    def from_pdetoolbox(cls, p: np.ndarray, t: np.ndarray, region_to_fiber: Dict[int, int],
                        fiber_types: np.ndarray, capillaries: np.ndarray, rcap: np.ndarray,
                        box: np.ndarray, wall_tol: float = 1e-6) -> "TissueMesh":
        """Wrap a MATLAB ``[p, e, t]`` mesh. ``t`` is 4 x M (1-based nodes, sub-domain);
        ``region_to_fiber`` maps sub-domain ids to model-fibre indices (others are IS).
        Boundary facets are labelled geometrically (capillary wall vs box)."""
        pts = np.asarray(p, dtype=float).T.copy()
        tt = np.asarray(t)
        tri = tt[:3].T.astype(np.int64) - 1
        sub = tt[3].astype(int)
        region = np.array([region_to_fiber.get(int(s), IS_REGION) for s in sub], dtype=int)
        mesh = cls(points=pts, triangles=_ccw(pts, tri), cell_region=region,
                   fiber_types=np.asarray(fiber_types, dtype=int),
                   capillary_facets=np.zeros((0, 2), int), capillary_facet_owner=np.zeros(0, int),
                   box_facets=np.zeros((0, 2), int), capillaries=np.asarray(capillaries, float),
                   rcap=np.asarray(rcap, float).ravel(), box=np.asarray(box, float),
                   info={"source": "pdetoolbox"})
        mesh._label_boundary(wall_tol)
        return mesh

    def _label_boundary(self, wall_tol: float) -> None:
        edges = np.sort(np.vstack([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]],
                                   self.triangles[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        bnd = uniq[counts == 1]
        mid = self.points[bnd].mean(axis=1)
        from scipy.spatial import KDTree
        d, j = KDTree(self.capillaries).query(mid)
        # an arc chord's midpoint lies inside the circle by r(1 - cos(theta/2))
        L = np.linalg.norm(self.points[bnd[:, 0]] - self.points[bnd[:, 1]], axis=1)
        sag = self.rcap[j] - np.sqrt(np.maximum(self.rcap[j] ** 2 - (L / 2) ** 2, 0))
        on_cap = np.abs(d - (self.rcap[j] - sag)) <= wall_tol + 1e-3 * self.rcap[j]
        self.capillary_facets = bnd[on_cap]
        self.capillary_facet_owner = j[on_cap]
        self.box_facets = bnd[~on_cap]


def _ccw(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = points[tri]
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
            (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    tri = tri.copy()
    neg = cross < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    return tri


# --------------------------------------------------------------------------
# gmsh
# --------------------------------------------------------------------------


@dataclass
class MeshSettings:
    """Element sizes (ndim units; 1 ndim = Ly/2, i.e. 165 um for the 440 x 330 um sample).

    ``h_wall``  size on capillary walls; default ``2*pi*Rcap / wall_segments``
    ``h_max``   size far from capillaries
    ``grade_from`` / ``grade_to``  distances (from the capillary walls) over
                which the size grows from ``h_wall`` to ``h_max``
    ``gap_width`` / ``h_gap``  interstitial channels between fibres narrower
                than ``gap_width`` (default 0.004, ~0.7 um) are meshed with size
                ``h_gap`` (default ``h_wall``) to avoid flat triangles across them;
                ``gap_width=0`` disables it. On the 2014 sample this lifts the
                worst radius ratio from 0.0025 to 0.045 for +48 % nodes.
    """

    wall_segments: int = 24
    h_wall: Optional[float] = None
    h_max: float = 0.02
    grade_from: float = 0.0
    grade_to: float = 0.15
    gap_width: float = 0.004               # IS channels narrower than this are refined (0 = off)
    h_gap: Optional[float] = None          # size in such channels (default h_wall)
    algorithm: int = 6                 # 6 = Frontal-Delaunay, 5 = Delaunay, 8 = Frontal quads off
    optimize: bool = True
    smoothing: int = 2
    polygon_segments: int = 48         # for capillaries that cannot be true circles
    overlap_rule: str = "matlab"       # fibre overlaps: "matlab" | "index" | "larger" | "split"
    verbosity: int = 0
    threads: int = 0                   # 0 = gmsh default
    backend: str = "auto"              # "gmsh" | "triangle" | "auto" (gmsh when installed, else Triangle)


def mesh_backend(settings: Optional["MeshSettings"] = None) -> str:
    """The mesher :func:`build_tissue_mesh` will use: "gmsh" or "triangle"."""
    choice = (settings.backend if settings is not None else "auto") or "auto"
    if choice not in ("auto", "gmsh", "triangle"):
        raise ValueError("MeshSettings.backend must be 'auto', 'gmsh' or 'triangle'")
    if choice != "auto":
        return choice
    try:
        import gmsh  # noqa: F401
        return "gmsh"
    except ImportError:                 # e.g. in a browser (Pyodide): no gmsh build
        return "triangle"


class _GeoBuilder:
    """Points/lines with de-duplication so that neighbouring faces share curves."""

    def __init__(self, gmsh_mod, decimals: int = 12):
        self.g = gmsh_mod
        self.geo = gmsh_mod.model.geo
        self.decimals = decimals
        self.points: Dict[Tuple[float, float], int] = {}
        self.lines: Dict[Tuple[int, int], int] = {}

    def point(self, x: float, y: float) -> int:
        key = (round(float(x), self.decimals), round(float(y), self.decimals))
        tag = self.points.get(key)
        if tag is None:
            tag = self.geo.addPoint(key[0], key[1], 0.0)
            self.points[key] = tag
        return tag

    def line(self, a: int, b: int) -> int:
        """Signed line tag from point a to point b (shared with the reverse direction)."""
        if a == b:
            raise ValueError("degenerate segment")
        key = (a, b) if a < b else (b, a)
        tag = self.lines.get(key)
        if tag is None:
            tag = self.geo.addLine(key[0], key[1])
            self.lines[key] = tag
        return tag if key == (a, b) else -tag

    def ring_loop(self, coords) -> Tuple[int, List[int]]:
        pts = [self.point(x, y) for x, y in np.asarray(coords)[:-1]]
        pts = [p for i, p in enumerate(pts) if p != pts[i - 1]]   # drop repeats
        lines = [self.line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        return self.geo.addCurveLoop(lines), [abs(t) for t in lines]

    def circle_loop(self, cx: float, cy: float, r: float) -> Tuple[int, List[int]]:
        c = self.geo.addPoint(cx, cy, 0.0)
        q = [self.geo.addPoint(cx + r * math.cos(a), cy + r * math.sin(a), 0.0)
             for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
        arcs = [self.geo.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
        return self.geo.addCurveLoop(arcs), arcs


def build_tissue_mesh(model: ModelGeometry, settings: Optional[MeshSettings] = None,
                      plan: Optional[RegionPlan] = None, msh_path: Optional[str] = None,
                      progress: Optional[ProgressCallback] = None) -> TissueMesh:
    """Mesh the model with gmsh (built-in kernel) and return a :class:`TissueMesh`.

    ``msh_path`` optionally writes the gmsh ``.msh`` file (physical groups included).
    ``progress(fraction, message)`` is called between the stages (planning,
    geometry, meshing, optimisation, extraction); gmsh itself cannot be
    interrupted inside ``generate``.

    gmsh keeps one global model per process: callers that mesh from several
    threads must serialise the calls (``otm_desktop.workers`` holds a lock).
    """
    s = settings or MeshSettings()
    if mesh_backend(s) == "triangle":
        from .mesh_triangle import build_tissue_mesh_triangle

        return build_tissue_mesh_triangle(model, s, plan, progress)
    import gmsh  # imported lazily: the rest of otm_core does not need it

    t0 = time.perf_counter()
    report(progress, 0.0, "Planning mesh regions")
    plan = plan or plan_regions(model, polygon_segments=s.polygon_segments, overlap_rule=s.overlap_rule)
    report(progress, 0.1, "Building the gmsh geometry")
    rmin = float(np.min(plan.rcap)) if plan.rcap.size else 0.01
    h_wall = s.h_wall if s.h_wall is not None else 2 * math.pi * rmin / s.wall_segments

    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Verbosity", s.verbosity)
        if s.threads:
            gmsh.option.setNumber("General.NumThreads", s.threads)
        gmsh.model.add("otm_tissue")
        B = _GeoBuilder(gmsh)

        # holes per face (circular capillaries)
        holes_of: Dict[int, List[int]] = {}
        for c, f in zip(plan.circle_caps, plan.circle_face):
            holes_of.setdefault(int(f), []).append(int(c))

        surf_region: Dict[int, int] = {}
        cap_curves: Dict[int, List[int]] = {}
        all_ring_lines: List[int] = []
        for fi, face in enumerate(plan.faces):
            outer, lines = B.ring_loop(face.exterior.coords)
            loops = [outer]
            all_ring_lines += lines
            for hole in face.interiors:
                lp, lines = B.ring_loop(hole.coords)
                loops.append(lp)
                all_ring_lines += lines
            for c in holes_of.get(fi, []):
                lp, arcs = B.circle_loop(plan.capillaries[c, 0], plan.capillaries[c, 1], plan.rcap[c])
                loops.append(lp)
                cap_curves[c] = arcs
            tag = gmsh.model.geo.addPlaneSurface(loops)
            surf_region[tag] = int(plan.face_region[fi])
        gmsh.model.geo.synchronize()

        # classify straight boundary segments: box or polygonal capillary wall
        box_lines, polycap_lines = _classify_lines(gmsh, B, plan)
        for c, tags in polycap_lines.items():
            cap_curves.setdefault(c, []).extend(tags)

        # physical groups
        by_region: Dict[int, List[int]] = {}
        for tag, r in surf_region.items():
            by_region.setdefault(r, []).append(tag)
        for r, tags in by_region.items():
            pg = PHYS_IS if r == IS_REGION else PHYS_FIBER0 + r
            gmsh.model.addPhysicalGroup(2, tags, pg, name="IS" if r == IS_REGION else f"fiber_{r}")
        if box_lines:
            gmsh.model.addPhysicalGroup(1, box_lines, PHYS_BOX, name="box")
        for c, tags in cap_curves.items():
            gmsh.model.addPhysicalGroup(1, tags, PHYS_CAPILLARY0 + c, name=f"capillary_{c}")

        # size fields: capillary walls (Distance -> Threshold) and thin IS channels
        all_cap_curves = [t for tags in cap_curves.values() for t in tags]
        gap_w = s.gap_width
        gap_lines = _thin_channel_lines(B, plan, gap_w) if gap_w > 0 else []
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", s.h_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(h_wall, s.h_max) * 0.25)
        fields = []
        if all_cap_curves:
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", all_cap_curves)
            gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_wall)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", s.h_max)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", s.grade_from)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", s.grade_to)
            fields.append(ft)
        if gap_lines:
            h_gap = s.h_gap if s.h_gap is not None else h_wall
            fg = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fg, "CurvesList", gap_lines)
            gmsh.model.mesh.field.setNumber(fg, "Sampling", 50)
            fgt = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(fgt, "InField", fg)
            gmsh.model.mesh.field.setNumber(fgt, "SizeMin", h_gap)
            gmsh.model.mesh.field.setNumber(fgt, "SizeMax", s.h_max)
            gmsh.model.mesh.field.setNumber(fgt, "DistMin", 0.5 * gap_w)
            gmsh.model.mesh.field.setNumber(fgt, "DistMax", 2.0 * gap_w)
            fields.append(fgt)
        if fields:
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.Algorithm", s.algorithm)
        gmsh.option.setNumber("Mesh.Smoothing", s.smoothing)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        report(progress, 0.25, "Generating the triangle mesh (gmsh)")
        gmsh.model.mesh.generate(2)
        if s.optimize:
            report(progress, 0.8, "Optimising the mesh")
            gmsh.model.mesh.optimize("Laplace2D")
        if msh_path:
            gmsh.write(msh_path)
        report(progress, 0.9, "Extracting regions and boundaries")
        mesh = _extract(gmsh, surf_region, cap_curves, box_lines, plan)
    finally:
        if owns_session:
            gmsh.finalize()
    mesh.info.update({"source": "gmsh", "gmsh_version": gmsh.__version__, "h_wall": h_wall,
                      "h_max": s.h_max, "grade_to": s.grade_to, "seconds": time.perf_counter() - t0,
                      "notes": list(plan.notes), "n_faces": len(plan.faces),
                      "n_circle_capillaries": int(plan.circle_caps.size),
                      "n_polygon_capillaries": int(plan.polygon_caps.size)})
    report(progress, 1.0, f"Mesh ready: {mesh.n_nodes} nodes, {mesh.n_triangles} triangles")
    return mesh


def _thin_channel_lines(B: _GeoBuilder, plan: RegionPlan, width: float) -> List[int]:
    """Line tags bordering interstitial channels narrower than ``width``
    (morphological opening of the IS by a disc of diameter ``width``)."""
    is_faces = [f for f, r in zip(plan.faces, plan.face_region) if r == IS_REGION]
    if not is_faces or len(plan.faces) == len(is_faces):
        return []
    IS = shapely.union_all(is_faces)
    opened = shapely.buffer(shapely.buffer(IS, -0.5 * width, quad_segs=4), 0.5 * width, quad_segs=4)
    thin = shapely.difference(IS, opened)
    thin = shapely.union_all([p for p in _polys(thin) if p.area > 1e-12])
    if thin.is_empty:
        return []
    coords = {tag: key for key, tag in B.points.items()}
    keys = list(B.lines.items())
    segs = shapely.linestrings([[coords[a], coords[b]] for (a, b), _ in keys])
    hit = np.unique(shapely.STRtree(segs).query(shapely.buffer(thin, 1e-9), predicate="intersects"))
    return [keys[i][1] for i in hit.tolist()]


def _classify_lines(gmsh, B: _GeoBuilder, plan: RegionPlan):
    """Straight segments on the box edges, or on a polygonal capillary wall."""
    box = plan.box[:4]
    xmin, ymin = box.min(axis=0)
    xmax, ymax = box.max(axis=0)
    tol = 1e-9 * max(xmax - xmin, ymax - ymin)
    coords = {tag: key for key, tag in B.points.items()}
    box_lines: List[int] = []
    cap_lines: Dict[int, List[int]] = {}
    pcaps = plan.polygon_caps
    for (a, b), tag in B.lines.items():
        (x1, y1), (x2, y2) = coords[a], coords[b]
        on_box = ((abs(x1 - x2) <= tol and (abs(x1 - xmin) <= tol or abs(x1 - xmax) <= tol)) or
                  (abs(y1 - y2) <= tol and (abs(y1 - ymin) <= tol or abs(y1 - ymax) <= tol)))
        if on_box:
            box_lines.append(tag)
            continue
        if pcaps.size:
            seg = shapely.linestrings([[x1, y1], [x2, y2]])
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            d = np.hypot(plan.capillaries[pcaps, 0] - mx, plan.capillaries[pcaps, 1] - my)
            j = int(np.argmin(d))
            ring = plan.polygon_cap_shapes[j].exterior
            if d[j] < plan.rcap[pcaps[j]] and shapely.hausdorff_distance(seg, shapely.intersection(
                    seg, shapely.buffer(ring, 1e-9))) <= 1e-9:
                cap_lines.setdefault(int(pcaps[j]), []).append(tag)
    return box_lines, cap_lines


def _extract(gmsh, surf_region: Dict[int, int], cap_curves: Dict[int, List[int]],
             box_lines: List[int], plan: RegionPlan) -> TissueMesh:
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)[:, :2]
    node_tags = np.asarray(node_tags, dtype=np.int64)
    lookup = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lookup[node_tags] = np.arange(node_tags.size)

    tris, regs = [], []
    for tag, r in surf_region.items():
        etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
        for et, nodes in zip(etypes, enodes):
            if et != 2:
                raise RuntimeError(f"unexpected element type {et} (expected 3-node triangles)")
            tri = lookup[np.asarray(nodes, dtype=np.int64)].reshape(-1, 3)
            tris.append(tri)
            regs.append(np.full(len(tri), r, dtype=int))
    tri = np.vstack(tris)
    region = np.concatenate(regs)

    def curve_edges(tags):
        out = []
        for t in tags:
            etypes, _, enodes = gmsh.model.mesh.getElements(1, t)
            for et, nodes in zip(etypes, enodes):
                if et == 1:
                    out.append(lookup[np.asarray(nodes, dtype=np.int64)].reshape(-1, 2))
        return np.vstack(out) if out else np.zeros((0, 2), dtype=np.int64)

    cap_facets, cap_owner = [], []
    for c, tags in cap_curves.items():
        e = curve_edges(tags)
        cap_facets.append(e)
        cap_owner.append(np.full(len(e), c, dtype=int))
    box_f = curve_edges(box_lines)

    # drop geometry-only nodes (circle centres) and renumber
    used = np.zeros(len(xyz), dtype=bool)
    used[tri.ravel()] = True
    new = np.cumsum(used) - 1
    pts = xyz[used]
    tri = new[tri]
    capf = new[np.vstack(cap_facets)] if cap_facets else np.zeros((0, 2), dtype=np.int64)
    capo = np.concatenate(cap_owner) if cap_owner else np.zeros(0, dtype=int)
    box_f = new[box_f] if len(box_f) else box_f
    return TissueMesh(points=pts, triangles=_ccw(pts, tri), cell_region=region,
                      fiber_types=np.asarray(plan.fiber_types, dtype=int), capillary_facets=capf,
                      capillary_facet_owner=capo, box_facets=box_f, capillaries=plan.capillaries,
                      rcap=plan.rcap, box=plan.box)


def mesh_tissue(geometry: Geometry, rcap: float, settings: Optional[MeshSettings] = None,
                msh_path: Optional[str] = None,
                progress: Optional[ProgressCallback] = None) -> Tuple[TissueMesh, ModelGeometry]:
    """Convenience: :func:`prepare_model_geometry` + :func:`build_tissue_mesh`."""
    report(progress, 0.0, "Preparing the model geometry")
    model = prepare_model_geometry(geometry, rcap)
    return build_tissue_mesh(model, settings, msh_path=msh_path,
                             progress=scaled(progress, 0.05, 1.0)), model
