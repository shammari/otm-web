"""Capillary-supply morphometry (report section 3.3, branch A).

=========================================  ==========================================
MATLAB                                     Python
=========================================  ==========================================
VoronoiCells (voronoin)                    :func:`voronoi_cells` (shapely.voronoi_polygons
                                           + Qhull boundedness via scipy Voronoi)
GetCapillaryDomainsInROI / GetFibersInROI  :func:`select_in_roi` (counting-frame
(inpolygon on 1e-5-spaced ROI edges)       rule, exact STRtree edge tests)
GetCapillaryDomainsStatistics              :func:`capillary_domain_statistics`
GetNearestNeighborStatistics (DelaunayTri, :func:`nearest_neighbour_statistics`
pdist2, RandomNNDistances,                 (scipy Delaunay adjacency)
UniqueMinNNDistances, rand_int)
LocalCapillaryToFiberRatio,                :func:`overlap_table` + :func:`supply_indices`
IndividualCapillaryToFiberRatio,           (one STRtree pass, one vectorised
GetFibersOverlappingRoiCapillaryDomains,   ``shapely.intersection``)
MaximumDiffusionDistanceToFiber
(polybool, polysplit, polyshape)
GetMorphometricData                        :func:`get_morphometric_data`
=========================================  ==========================================

Complexity: the legacy code calls ``polybool`` for every (fibre, cell) pair,
four separate times (LCFR twice, ICFR, FDR): about 4·Nf·Nv Boolean
operations. Here one ``STRtree.query(..., predicate="intersects")`` finds the
candidate pairs (typically 4–10 cells per fibre) and a single vectorised
``shapely.intersection`` computes them, so the cost is
O((Nf + Nv) log Nv + pairs). Every index is then a ``np.bincount`` over the
resulting sparse overlap table.

Known, documented deviations from MATLAB (also listed in
``Morphometrics.warnings`` when they occur):

* Unbounded Voronoi cells (legacy vertex at Inf) are clipped to a large
  envelope for intersection tests; their areas are NaN (legacy: Inf/NaN) and
  they contribute 0 to LCFR (legacy: NaN).
* "Overlap" means intersection area > ``min_overlap_area`` (default 0).
  ``polybool`` may return a zero-area sliver for fibres that only touch a
  cell edge; those are not counted here.
* Self-intersecting fibre outlines are repaired (``make_valid``) before
  Boolean operations; ``polybool``'s handling of such rings is undefined.
* Random nearest-neighbour pairing uses NumPy's RNG, so ``random_pairs`` /
  ``unique_min_pairs`` differ from MATLAB in value (not in distribution).
"""

from __future__ import annotations

import warnings as _warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import shapely
from scipy.spatial import Delaunay, Voronoi

from .geometry import _is_geom, fiber_area_um2, fiber_geometries, inpolygon, is_multipart
from .models import (
    CapillaryDomainStats,
    Geometry,
    LengthScale,
    Morphometrics,
    NearestNeighborStats,
    OverlapTable,
    RoiNdim,
    SummaryStats,
    SupplyIndices,
    VoronoiCells,
    _std,
)

# --------------------------------------------------------------------------
# Voronoi capillary domains
# --------------------------------------------------------------------------


def voronoi_cells(capillaries: np.ndarray, extent_factor: float = 1e3) -> VoronoiCells:
    """Voronoi tessellation of the capillary centres, one cell per capillary
    (``cells[i]`` ↔ ``capillaries[i]``).

    Cells come from ``shapely.voronoi_polygons`` extended to, and clipped by,
    an envelope ``extent_factor`` times the point spread. Which cells are
    *unbounded* (legacy Inf vertex) is taken from Qhull via
    ``scipy.spatial.Voronoi``, the same engine as MATLAB's ``voronoin``.
    """
    P = np.asarray(capillaries, dtype=float)
    if len(P) < 3:
        raise ValueError("at least 3 capillaries are needed for a Voronoi tessellation")
    if len(np.unique(P, axis=0)) != len(P):
        raise ValueError("duplicate capillary centres: run eliminate_duplicate_capillaries first")
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = extent_factor * span
    extent = (float(xmin - pad), float(ymin - pad), float(xmax + pad), float(ymax + pad))
    env = shapely.box(*extent)

    diagram = shapely.voronoi_polygons(shapely.multipoints(P), extend_to=env)
    parts = shapely.intersection(shapely.get_parts(diagram), env)
    # Map each capillary to the cell that contains it (order-independent;
    # does not rely on the GEOS >= 3.12 `ordered` flag).
    tree = shapely.STRtree(parts)
    pt_idx, cell_idx = tree.query(shapely.points(P), predicate="within")
    if len(pt_idx) != len(P) or len(np.unique(pt_idx)) != len(P):
        raise RuntimeError("Voronoi cell/capillary mapping failed")
    cells = np.empty(len(P), dtype=object)
    cells[pt_idx] = parts[cell_idx]

    vor = Voronoi(P)
    bounded = np.array([(-1 not in vor.regions[r]) and len(vor.regions[r]) > 0
                        for r in vor.point_region], dtype=bool)
    areas = shapely.area(cells).astype(float)
    areas[~bounded] = np.nan
    return VoronoiCells(cells=cells, bounded=bounded, areas_ndim=areas, extent=extent)


# --------------------------------------------------------------------------
# ROI selection
# --------------------------------------------------------------------------


def select_in_roi(geoms: np.ndarray, roi: RoiNdim, rule: str = "counting_frame") -> np.ndarray:
    """GetCapillaryDomainsInROI / GetFibersInROI, exact version.

    ``rule="counting_frame"`` (default, = legacy): the ROI is an unbiased
    stereological counting frame. The legacy code tests the ROI edges in a
    fixed order (right, bottom, left, top; each sampled every 1e-5, "in or
    on" the polygon) and only *includes* on a left/top hit, so:

    * a polygon touching the right edge (x = max, corners included) or the
      bottom edge (y = min) is **excluded**, even if it also touches another edge;
    * otherwise a polygon touching the left edge (x = min) or top edge
      (y = max) is **included**;
    * otherwise it is included if its first vertex is strictly inside.

    Verified against the MATLAB baseline: 38/38 ROI fibres and 46/46 ROI
    capillary domains. ``rule="any_edge"`` includes every polygon that
    touches any edge (a symmetric, biased alternative).

    Edge hits are exact Shapely ``intersects`` tests against the edge
    segments (STRtree) instead of 1e-5 sampling. Without an edge hit a
    polygon is entirely inside or outside the ROI, so any vertex decides the
    last test. Works for Polygons and MultiPolygons. Returns sorted 0-based
    indices.
    """
    lb, bb, rb, ub = roi.bounds
    right = shapely.linestrings([[rb, bb], [rb, ub]])
    bottom = shapely.linestrings([[lb, bb], [rb, bb]])
    left = shapely.linestrings([[lb, bb], [lb, ub]])
    top = shapely.linestrings([[lb, ub], [rb, ub]])
    tree = shapely.STRtree(geoms)

    def hits(edge) -> set:
        return set(tree.query(edge, predicate="intersects").tolist())

    excluded = hits(right) | hits(bottom)
    included = hits(left) | hits(top)
    coords, owner = shapely.get_coordinates(geoms, return_index=True)
    if len(owner):
        owners, first = np.unique(owner, return_index=True)
        fx, fy = coords[first, 0], coords[first, 1]
        strict = (fx > lb) & (fx < rb) & (fy > bb) & (fy < ub)
        included.update(owners[strict].tolist())
    if rule == "counting_frame":
        chosen = included - excluded
    elif rule == "any_edge":
        chosen = included | excluded
    else:
        raise ValueError(f"unknown ROI rule {rule!r}")
    return np.array(sorted(chosen), dtype=int)


# --------------------------------------------------------------------------
# Capillary domains
# --------------------------------------------------------------------------


def capillary_domain_statistics(capillaries: np.ndarray, ls: LengthScale, roi: RoiNdim,
                                vor: Optional[VoronoiCells] = None,
                                roi_rule: str = "counting_frame") -> CapillaryDomainStats:
    """GetCapillaryDomainsStatistics."""
    caps = np.asarray(capillaries, dtype=float)
    if vor is None:
        vor = voronoi_cells(caps)
    L = ls.index_length_um
    areas_um2 = ls.area_factor * vor.areas_ndim
    roi_idx = select_in_roi(vor.cells, roi, roi_rule)
    lb, bb, rb, ub = roi.bounds
    roi_box_um = L * np.array([[lb, lb, rb, rb, lb], [bb, ub, ub, bb, bb]])
    inside, _ = inpolygon(L * caps[:, 0], L * caps[:, 1], roi_box_um[0], roi_box_um[1])
    roi_area = roi.xrange * roi.yrange * ls.area_factor
    roi_areas = areas_um2[roi_idx]
    eqd = 2.0 * np.sqrt(roi_areas / np.pi)
    n = len(caps)
    return CapillaryDomainStats(
        voronoi=vor, x_length_um=ls.x_um, y_length_um=ls.y_um,
        tissue_area_um2=ls.tissue_area_um2, nondim_tissue_area=ls.nondim_tissue_area,
        roi_area_um2=roi_area, total_num_capillaries=n,
        capillary_density_per_mm2=1e6 * n / ls.tissue_area_um2,
        roi_capillary_density_per_mm2=1e6 * int(inside.sum()) / roi_area,
        roi_index=roi_idx, capillary_domain_areas_um2=areas_um2,
        roi_capillary_domain_areas_um2=roi_areas, domain_equivalent_diameter_um=eqd,
        area_stats=SummaryStats.of(roi_areas), diameter_stats=SummaryStats.of(eqd),
        roi_box_um=roi_box_um)


# --------------------------------------------------------------------------
# Nearest neighbours
# --------------------------------------------------------------------------


def delaunay_neighbours(points: np.ndarray) -> List[np.ndarray]:
    """Delaunay adjacency (``DelaunayTri`` + ``isEdge``), ascending 0-based ids."""
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    return [np.sort(indices[indptr[i]:indptr[i + 1]]) for i in range(len(points))]


def _pairing(ids1: np.ndarray, lists1: List[np.ndarray], dists: List[np.ndarray],
             rng: np.random.Generator, mode: str) -> Tuple[np.ndarray, List[str]]:
    """Port of RandomNNDistances (``mode='random'``) and UniqueMinNNDistances
    (``mode='min'``). Works with 1-based ids internally, like MATLAB, and keeps
    the legacy quirks (``idx = randsort(idx)``; the last branch pairs a
    capillary with itself). Where MATLAB would raise (several candidate
    partners) or scalar-expand (no partner -> row ``[id id id]``), this
    takes the first candidate / writes ``[id, 0, NaN]`` and records a note.
    Returns connxn (n x 3, 1-based ids, 0 = none) and the notes."""
    notes: List[str] = []
    Ncap = len(ids1)
    randsort = rng.permutation(Ncap) + 1              # rand_int(1, Ncap, Ncap)
    connxn = np.zeros((Ncap, 3))
    unq: List[float] = []
    nonunq: List[List[float]] = []

    def pick(caplist, capdist):
        if mode == "random":
            j = int(rng.integers(1, len(caplist) + 1)) - 1      # randi(N)
        else:
            j = int(np.argmin(capdist))                        # [~, idcap] = min(capdist)
        return j, caplist[j], capdist[j]

    def assign(i, a, capn, d):
        idx = np.flatnonzero(ids1 == capn)
        connxn[i - 1] = [a, capn, d]
        for r in randsort[idx]:                       # legacy: idx = randsort(idx)
            connxn[r - 1] = [capn, a, d]

    for k in range(1, Ncap + 1):
        i = int(randsort[k - 1])
        a = ids1[i - 1]
        if a not in unq:
            caplist = list(lists1[i - 1])
            capdist = list(dists[i - 1])
            N = len(caplist)
            j, capn, d = pick(caplist, capdist)
            if k > 1:
                n = 1
                while capn in unq:
                    if n == N:
                        j, capn, d = pick(list(lists1[i - 1]), list(dists[i - 1]))
                        nonunq.append([a, capn, d])
                        break
                    del caplist[j]
                    del capdist[j]
                    if mode == "random":
                        j = int(rng.integers(1, N - n + 1)) - 1   # randi(N-n)
                    else:
                        j = int(np.argmin(capdist))
                    capn, d = caplist[j], capdist[j]
                    n += 1
            unq.extend([a, capn])
            assign(i, a, capn, d)
        elif a not in connxn[:, 0]:
            idx = np.flatnonzero(connxn[:, 1] == a)
            capn, d = connxn[idx, 0], connxn[idx, 2]
            if nonunq:
                m = ~np.isin(capn, np.array(nonunq)[:, 0])
                capn, d = capn[m], d[m]
            if capn.size:
                if capn.size > 1:
                    notes.append(f"capillary {int(a)}: {capn.size} candidate partners "
                                 "(MATLAB raises here); first one used")
                connxn[i - 1] = [a, capn[0], d[0]]
            else:
                notes.append(f"capillary {int(a)}: no partner (legacy writes [id id id])")
                connxn[i - 1] = [a, 0, np.nan]
        else:
            idx = np.flatnonzero(connxn[:, 0] == a)
            capn, d = connxn[idx, 0], connxn[idx, 2]       # legacy reads column 1 (self id)
            if nonunq:
                m = ~np.isin(capn, np.array(nonunq)[:, 1])
                capn, d = capn[m], d[m]
            if capn.size:
                if capn.size > 1:
                    notes.append(f"capillary {int(a)}: {capn.size} candidate rows "
                                 "(MATLAB raises here); first one used")
                connxn[i - 1] = [capn[0], a, d[0]]
            else:
                notes.append(f"capillary {int(a)}: no partner (legacy writes [id id id])")
                connxn[i - 1] = [a, 0, np.nan]
    return connxn, notes


def nearest_neighbour_statistics(capillaries: np.ndarray, roi_index: np.ndarray, ls: LengthScale,
                                 rng: Optional[np.random.Generator] = None) -> NearestNeighborStats:
    """GetNearestNeighborStatistics. Distances in um (``Ly/2`` scale, as the
    legacy call passes LengthScaleY)."""
    rng = rng if rng is not None else np.random.default_rng()
    caps = np.asarray(capillaries, dtype=float)
    L = ls.index_length_um
    nbrs = delaunay_neighbours(caps)
    roi_index = np.asarray(roi_index, dtype=int)
    neighbours, distances = [], []
    for c in roi_index:
        nb = nbrs[c]
        neighbours.append(nb)
        distances.append(L * np.hypot(*(caps[nb] - caps[c]).T))
    total = np.array([d.size for d in distances], dtype=int)
    means = np.array([d.mean() for d in distances])
    mins = np.array([d.min() for d in distances])
    stds = np.array([_std(d) for d in distances])
    sems = stds / np.sqrt(total)
    alld = np.concatenate(distances) if distances else np.zeros(0)

    ids1 = roi_index + 1.0
    lists1 = [nb + 1.0 for nb in neighbours]
    rnd, n1 = _pairing(ids1, lists1, distances, rng, "random")
    umin, n2 = _pairing(ids1, lists1, distances, rng, "min")

    def to0(c):
        out = c.copy()
        out[:, :2] = np.where(c[:, :2] > 0, c[:, :2] - 1, -1)
        return out

    n = roi_index.size

    def s(v):                                   # MATLAB std, ignoring unpaired (NaN) rows
        v = np.asarray(v, float)
        return float(_std(v[np.isfinite(v)]))

    def m(v):
        v = np.asarray(v, float)
        return float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan")

    summary = {
        "MeanOfMeans": m(means), "STDOfMeans": s(means), "SEMOfMeans": s(means) / np.sqrt(n),
        "MeanOfSTDs": m(stds), "MeanOfSEMs": m(sems),
        "MeanOfMin": m(mins), "STDOfMin": s(mins), "SEMOfMin": s(mins) / np.sqrt(n),
        "MeanOfUniqueMin": m(umin[:, 2]), "STDOfUniqueMin": s(umin[:, 2]),
        "SEMOfUniqueMin": s(umin[:, 2]) / np.sqrt(n),
        "MeanOfRnd": m(rnd[:, 2]), "STDOfRnd": s(rnd[:, 2]), "SEMOfRnd": s(rnd[:, 2]) / np.sqrt(n),
        "MeanAll": m(alld), "STDAll": s(alld),
        "SEMAll_legacy": s(alld) / alld.size if alld.size else float("nan"),  # legacy divides by N
        "SEMAll": s(alld) / np.sqrt(alld.size) if alld.size else float("nan"),
        "unpaired_random": int(np.isnan(rnd[:, 2]).sum()),
        "unpaired_unique_min": int(np.isnan(umin[:, 2]).sum()),
        "pairing_notes": n1 + n2,
    }
    return NearestNeighborStats(neighbours=neighbours, distances=distances, total=total, means=means,
                                mins=mins, stds=stds, sems=sems, all_distances=alld,
                                random_pairs=to0(rnd), unique_min_pairs=to0(umin), summary=summary)


# --------------------------------------------------------------------------
# Fibre / domain overlaps (single STRtree pass)
# --------------------------------------------------------------------------


def overlap_table(fiber_geoms: np.ndarray, vor: VoronoiCells, capillaries: np.ndarray,
                  min_overlap_area: float = 0.0) -> OverlapTable:
    """All (fibre, capillary-domain) overlaps, with the intersection area and
    the maximum distance from the domain's capillary to the vertices of the
    overlap (MaximumDiffusionDistanceToFiber, before averaging).

    One STRtree over the Voronoi cells answers every fibre at once
    (bounding-box filter + exact ``intersects``); the surviving pairs are
    intersected in a single vectorised call. Multi-polygon fibres are handled
    natively: area and vertices cover every part.
    """
    caps = np.asarray(capillaries, dtype=float)
    tree = shapely.STRtree(vor.cells)
    f_idx, c_idx = tree.query(fiber_geoms, predicate="intersects")
    if f_idx.size == 0:
        zi, zf = np.zeros(0, dtype=int), np.zeros(0)
        return OverlapTable(zi, zi.copy(), zf, zf.copy(), len(fiber_geoms), len(vor.cells))
    inter = shapely.intersection(fiber_geoms[f_idx], vor.cells[c_idx])
    area = shapely.area(inter).astype(float)
    keep = area > min_overlap_area
    f_idx, c_idx, inter, area = f_idx[keep], c_idx[keep], inter[keep], area[keep]
    coords, owner = shapely.get_coordinates(inter, return_index=True)
    dmax = np.zeros(len(inter))
    if len(owner):
        d = np.hypot(coords[:, 0] - caps[c_idx[owner], 0], coords[:, 1] - caps[c_idx[owner], 1])
        np.maximum.at(dmax, owner, d)
    order = np.lexsort((c_idx, f_idx))
    return OverlapTable(fiber=f_idx[order], cell=c_idx[order], area_ndim=area[order],
                        max_vertex_distance_ndim=dmax[order], n_fibers=len(fiber_geoms),
                        n_cells=len(vor.cells))


def supply_indices(fibers: Sequence, fiber_geoms: np.ndarray, domains: CapillaryDomainStats,
                   capillaries: np.ndarray, ls: LengthScale, roi: RoiNdim,
                   overlaps: Optional[OverlapTable] = None,
                   min_overlap_area: float = 0.0,
                   roi_rule: str = "counting_frame") -> Tuple[SupplyIndices, OverlapTable]:
    """GetSupplyIndices: LCFR, LCD, DFR, FDR, Dmax, SF, CC, CFi, FPi, CFPE.

    Legacy definitions (F_k = ROI fibre k, V_j = Voronoi cell j):

    * LCFR_k = Σ_j |F_k ∩ V_j| / |V_j|;  LCD_k = 1e6 · LCFR_k / area_um2(F_k)
    * DFR_k = CC_k = #cells overlapping F_k
    * Dmax_k = (Ly/2) · mean_j max_{v ∈ vertices(F_k ∩ V_j)} |v − cap_j|
    * SF_j = #ROI fibres overlapping V_j;  CFi_k = Σ_j 1/SF_j
    * FPi_k = perimeter(F_k) in um;  CFPE_k = 1000 · CFi_k / FPi_k
    * FDR_j = #fibres (all, not only ROI) overlapping ROI cell j
    """
    vor = domains.voronoi
    L = ls.index_length_um
    ov = overlaps if overlaps is not None else overlap_table(fiber_geoms, vor, capillaries,
                                                              min_overlap_area)
    roi_fib = select_in_roi(fiber_geoms, roi, roi_rule)
    in_roi = np.zeros(ov.n_fibers, dtype=bool)
    in_roi[roi_fib] = True
    roi_rows = in_roi[ov.fiber]

    nf = roi_fib.size
    pos = np.full(ov.n_fibers, -1)
    pos[roi_fib] = np.arange(nf)
    rf = pos[ov.fiber[roi_rows]]                 # ROI-fibre position of each ROI row
    rc = ov.cell[roi_rows]
    cell_area = np.where(vor.bounded, shapely.area(vor.cells), np.inf)

    # LocalCapillaryToFiberRatio
    dfr = np.bincount(rf, minlength=nf).astype(float)
    lcfr = np.bincount(rf, weights=ov.area_ndim[roi_rows] / cell_area[rc], minlength=nf)
    fib_area = np.array([fiber_area_um2(fibers[k], ls, legacy=True) for k in roi_fib], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lcd = 1e6 * lcfr / fib_area
    dsum = np.bincount(rf, weights=ov.max_vertex_distance_ndim[roi_rows], minlength=nf)
    with np.errstate(invalid="ignore", divide="ignore"):
        dmax = L * dsum / dfr                    # mean over overlapping domains (NaN if none)
    total_caps = int(np.unique(rc).size)

    # IndividualCapillaryToFiberRatio (SF counts ROI fibres per cell)
    sf_all = np.bincount(rc, minlength=ov.n_cells).astype(float)
    cfi = np.bincount(rf, weights=1.0 / sf_all[rc], minlength=nf) if rc.size else np.zeros(nf)
    fpi = L * shapely.length(fiber_geoms[roi_fib]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cfpe = 1000.0 * cfi / fpi

    # GetFibersOverlappingRoiCapillaryDomains (ALL fibres per ROI cell)
    fdr_all = np.bincount(ov.cell, minlength=ov.n_cells)
    fdr = fdr_all[domains.roi_index].astype(float)
    sf = sf_all[domains.roi_index]

    unb = int(np.count_nonzero(~vor.bounded[rc]))
    si = SupplyIndices(
        roi_fiber_index=roi_fib, fiber_area_um2=fib_area,
        LCFR=SummaryStats.of(lcfr), LCD=SummaryStats.of(lcd), DFR=SummaryStats.of(dfr),
        FDR=SummaryStats.of(fdr), Dmax=SummaryStats.of(dmax), SF=SummaryStats.of(sf),
        CC=SummaryStats.of(dfr), CFi=SummaryStats.of(cfi), FPi=SummaryStats.of(fpi),
        CFPE=SummaryStats.of(cfpe), number_of_fibers=nf, number_of_capillaries=total_caps,
        unbounded_overlaps=unb)
    return si, ov


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def get_morphometric_data(geometry: Geometry, rng: Optional[np.random.Generator] = None,
                          min_overlap_area: float = 0.0,
                          voronoi_extent_factor: float = 1e3,
                          emit_warnings: bool = True,
                          roi_rule: str = "counting_frame") -> Morphometrics:
    """GetMorphometricData for a Geometry with length scale and ROI set."""
    ls = geometry.length_scale
    if ls is None:
        raise ValueError("geometry.length_scale must be set")
    roi = geometry.roi_ndim()
    caps = np.asarray(geometry.capillaries, dtype=float)
    notes: List[str] = []

    vor = voronoi_cells(caps, voronoi_extent_factor)
    domains = capillary_domain_statistics(caps, ls, roi, vor, roi_rule)
    if (~vor.bounded[domains.roi_index]).any():
        notes.append("ROI contains unbounded Voronoi cells (legacy areas would be Inf/NaN)")
    nn = nearest_neighbour_statistics(caps, domains.roi_index, ls, rng)
    notes.extend(nn.summary["pairing_notes"])

    supply, ov = None, None
    if geometry.is_skeletal:
        raw = list(geometry.fibers)
        geoms = fiber_geometries(raw)
        repaired = sum(1 for r, g in zip(raw, geoms)
                       if not _is_geom(r) and not is_multipart(r) and shapely.get_type_id(g) == 6)
        if repaired:
            notes.append(f"{repaired} fibre outline(s) were self-intersecting and became MultiPolygons")
        supply, ov = supply_indices(raw, geoms, domains, caps, ls, roi,
                                    min_overlap_area=min_overlap_area, roi_rule=roi_rule)
        if supply.unbounded_overlaps:
            notes.append(f"{supply.unbounded_overlaps} ROI-fibre overlaps with unbounded cells "
                         "(counted with 0 LCFR contribution)")
    if emit_warnings:
        for n in notes:
            _warnings.warn(n, RuntimeWarning, stacklevel=2)
    return Morphometrics(domains=domains, nearest_neighbours=nn, supply=supply, overlaps=ov,
                         capillaries=caps, warnings=notes)
