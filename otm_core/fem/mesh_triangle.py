"""Tissue meshes with Shewchuk's Triangle (the mesher used where gmsh is not available).

gmsh is a C++ library with no WebAssembly build, so the browser version of OTM
cannot use :func:`otm_core.fem.mesh.build_tissue_mesh`. Triangle (quality
constrained Delaunay triangulation, one C file) runs everywhere: as the Python
``triangle`` package on a desktop and as ``triangle-wasm`` in a browser. This
module builds the same :class:`~otm_core.fem.mesh.TissueMesh` from the same
:class:`~otm_core.fem.mesh.RegionPlan` as the gmsh path:

* the planar partition (fibres, interstitial space, capillary holes) comes from
  :func:`~otm_core.fem.mesh.plan_regions`, unchanged;
* every face boundary is a set of shared segments, so the mesh is conforming;
  circular capillaries are polygons with ``wall_segments`` sides (gmsh puts
  the same number of nodes on each wall); each face carries a region attribute
  (``-A``), and the box and capillary walls carry segment markers;
* gmsh's size field (``h_wall`` at capillary walls growing to ``h_max`` over
  ``grade_to``, finer in thin interstitial channels) is reproduced by Delaunay
  refinement with a per-triangle area limit (``-r -a``), repeated until every
  triangle meets the size the field asks for at its centroid.

The triangulator is injected: ``triangulate(switches, data) -> dict`` with
Triangle's own field names (``pointlist`` (N, 2), ``segmentlist`` (S, 2),
``segmentmarkerlist``, ``holelist``, ``regionlist`` (R, 4), ``trianglelist``,
``triangleattributelist``, ``trianglearealist``). :func:`set_triangulator`
registers one (the browser worker registers the WebAssembly build); by
default the Python ``triangle`` package is used when it is installed.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import shapely

from ..progress import ProgressCallback, report

__all__ = ["build_tissue_mesh_triangle", "set_triangulator", "get_triangulator", "triangle_available",
           "MARKER_BOX", "MARKER_CAPILLARY0"]

Triangulator = Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]

MARKER_BOX = 2
MARKER_CAPILLARY0 = 10
_SQ3_4 = math.sqrt(3.0) / 4.0
_triangulator: Optional[Triangulator] = None


def set_triangulator(fn: Optional[Triangulator]) -> None:
    """Register the Triangle binding to use (None = back to the default)."""
    global _triangulator
    _triangulator = fn


def _python_triangle() -> Triangulator:
    import triangle as tr                      # the "triangle" package (Shewchuk's Triangle)

    names = {"pointlist": "vertices", "segmentlist": "segments", "segmentmarkerlist": "segment_markers",
             "holelist": "holes", "regionlist": "regions", "trianglelist": "triangles",
             "triangleattributelist": "triangle_attributes", "trianglearealist": "triangle_max_area"}
    back = {v: k for k, v in names.items()}

    def run(switches: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d = {}
        for k, v in data.items():
            if v is None or np.size(v) == 0:
                continue
            v = np.asarray(v)
            if k in ("segmentmarkerlist", "triangleattributelist", "trianglearealist"):
                v = v.reshape(-1, 1)
            d[names[k]] = v
        out = tr.triangulate(d, switches.replace("z", "").replace("Q", "") + "Q")
        res = {back[k]: np.asarray(v) for k, v in out.items() if k in back}
        for k in ("segmentmarkerlist", "triangleattributelist"):
            if k in res:
                res[k] = res[k].ravel()
        return res

    return run


def get_triangulator() -> Triangulator:
    if _triangulator is not None:
        return _triangulator
    try:
        return _python_triangle()
    except ImportError as exc:                 # pragma: no cover - depends on the install
        raise RuntimeError("no Triangle binding: install the 'triangle' package, or register one with "
                           "otm_core.fem.mesh_triangle.set_triangulator()") from exc


def triangle_available() -> bool:
    if _triangulator is not None:
        return True
    try:
        import triangle  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# PSLG from the region plan
# --------------------------------------------------------------------------


class _Pslg:
    """Vertices and segments with de-duplication (shared face boundaries)."""

    def __init__(self, decimals: int = 12):
        self.decimals = decimals
        self.index: Dict[Tuple[float, float], int] = {}
        self.points: List[Tuple[float, float]] = []
        self.segments: Dict[Tuple[int, int], int] = {}

    def point(self, x: float, y: float) -> int:
        key = (round(float(x), self.decimals), round(float(y), self.decimals))
        i = self.index.get(key)
        if i is None:
            i = len(self.points)
            self.index[key] = i
            self.points.append(key)
        return i

    def ring(self, coords, marker: int = 0) -> None:
        pts = [self.point(x, y) for x, y in np.asarray(coords)[:-1]]
        pts = [p for i, p in enumerate(pts) if p != pts[i - 1]]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in self.segments or (marker and not self.segments[key]):
                self.segments[key] = marker


def _segment_markers(P: _Pslg, plan: Any) -> None:
    """Box edges -> MARKER_BOX; edges of polygonal capillary holes -> their capillary."""
    pts = np.asarray(P.points)
    box = plan.box[:4]
    xmin, ymin = box.min(axis=0)
    xmax, ymax = box.max(axis=0)
    tol = 1e-9 * max(xmax - xmin, ymax - ymin)
    pcaps = plan.polygon_caps
    for (a, b), m in list(P.segments.items()):
        if m:
            continue
        (x1, y1), (x2, y2) = pts[a], pts[b]
        on_box = ((abs(x1 - x2) <= tol and (abs(x1 - xmin) <= tol or abs(x1 - xmax) <= tol)) or
                  (abs(y1 - y2) <= tol and (abs(y1 - ymin) <= tol or abs(y1 - ymax) <= tol)))
        if on_box:
            P.segments[(a, b)] = MARKER_BOX
            continue
        if pcaps.size:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            d = np.hypot(plan.capillaries[pcaps, 0] - mx, plan.capillaries[pcaps, 1] - my)
            j = int(np.argmin(d))
            if d[j] < plan.rcap[pcaps[j]]:
                seg = shapely.linestrings([[x1, y1], [x2, y2]])
                ring = plan.polygon_cap_shapes[j].exterior
                if shapely.hausdorff_distance(seg, shapely.intersection(seg, shapely.buffer(ring, 1e-9))) <= 1e-9:
                    P.segments[(a, b)] = MARKER_CAPILLARY0 + int(pcaps[j])


def _build_pslg(plan: Any, wall_segments: int, h_wall: float) -> Tuple[_Pslg, np.ndarray, np.ndarray]:
    P = _Pslg()
    for face in plan.faces:
        P.ring(face.exterior.coords)
        for hole in face.interiors:
            P.ring(hole.coords)
    holes = []
    for c in plan.circle_caps:
        cx, cy = plan.capillaries[c]
        r = float(plan.rcap[c])
        n = max(int(wall_segments), int(math.ceil(2 * math.pi * r / h_wall)), 8)
        ang = 2 * math.pi * np.arange(n + 1) / n
        P.ring(np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)], MARKER_CAPILLARY0 + int(c))
        holes.append((cx, cy))
    _segment_markers(P, plan)
    # polygonal capillary holes (and any other uncovered area): points not in any face
    covered = shapely.union_all(plan.faces) if plan.faces else shapely.Polygon()
    rest = shapely.difference(shapely.Polygon(plan.box), covered)
    for part in shapely.get_parts(rest):
        if part.area > 1e-14:
            rp = part.representative_point()
            holes.append((rp.x, rp.y))
    regions = []
    for fi, face in enumerate(plan.faces):
        rp = face.representative_point()
        regions.append((rp.x, rp.y, float(fi + 1), 0.0))          # attribute = face index + 1
    return P, np.asarray(holes, float).reshape(-1, 2), np.asarray(regions, float).reshape(-1, 4)


# --------------------------------------------------------------------------
# size field (as gmsh's Distance + Threshold fields)
# --------------------------------------------------------------------------


def _thin_channels(plan: Any, width: float):
    from .mesh import IS_REGION, _polys

    if width <= 0:
        return None
    is_faces = [f for f, r in zip(plan.faces, plan.face_region) if r == IS_REGION]
    if not is_faces or len(plan.faces) == len(is_faces):
        return None
    IS = shapely.union_all(is_faces)
    opened = shapely.buffer(shapely.buffer(IS, -0.5 * width, quad_segs=4), 0.5 * width, quad_segs=4)
    thin = shapely.union_all([p for p in _polys(shapely.difference(IS, opened)) if p.area > 1e-12])
    return None if thin.is_empty else thin


def _target_size(xy: np.ndarray, plan: Any, s: Any, h_wall: float, thin: Any) -> np.ndarray:
    h = np.full(len(xy), float(s.h_max))
    if len(plan.capillaries):
        from scipy.spatial import cKDTree

        k = min(4, len(plan.capillaries))
        d, j = cKDTree(plan.capillaries).query(xy, k=k)
        d = np.asarray(d).reshape(len(xy), -1)
        j = np.asarray(j).reshape(len(xy), -1)
        dist = np.min(d - plan.rcap[j], axis=1).clip(min=0.0)
        span = max(s.grade_to - s.grade_from, 1e-12)
        frac = np.clip((dist - s.grade_from) / span, 0.0, 1.0)
        h = np.minimum(h, h_wall + (s.h_max - h_wall) * frac)
    if thin is not None:
        h_gap = s.h_gap if s.h_gap is not None else h_wall
        w = s.gap_width
        # nearest thin-channel edge within 2 w (farther points keep h_max): an STRtree query
        # instead of a distance to the whole boundary, which is slow for 10^5 points
        b = thin.boundary
        parts = shapely.get_parts(b)
        coords = [np.asarray(g.coords) for g in parts]
        segs = shapely.linestrings(np.concatenate([np.stack([c[:-1], c[1:]], axis=1) for c in coords if len(c) > 1]))
        dg = np.full(len(xy), np.inf)
        (qi, _ti), dd = shapely.STRtree(segs).query_nearest(shapely.points(xy), max_distance=2.0 * w,
                                                            return_distance=True)
        np.minimum.at(dg, qi, dd)
        frac = np.clip((dg - 0.5 * w) / (1.5 * w), 0.0, 1.0)
        h = np.minimum(h, h_gap + (s.h_max - h_gap) * frac)
    return np.maximum(h, min(h_wall, s.h_max) * 0.25)


# --------------------------------------------------------------------------
# the mesher
# --------------------------------------------------------------------------


def build_tissue_mesh_triangle(model: Any, settings: Any = None, plan: Any = None,
                               progress: Optional[ProgressCallback] = None,
                               min_angle: float = 30.0, max_passes: int = 6) -> Any:
    """Mesh ``model`` with Triangle; same inputs and result type as
    :func:`otm_core.fem.mesh.build_tissue_mesh`."""
    from .mesh import IS_REGION, MeshSettings, TissueMesh, _ccw, plan_regions

    s = settings or MeshSettings()
    run = get_triangulator()
    t0 = time.perf_counter()
    report(progress, 0.0, "Planning mesh regions")
    plan = plan or plan_regions(model, polygon_segments=s.polygon_segments, overlap_rule=s.overlap_rule)
    rmin = float(np.min(plan.rcap)) if plan.rcap.size else 0.01
    h_wall = s.h_wall if s.h_wall is not None else 2 * math.pi * rmin / s.wall_segments
    report(progress, 0.1, "Building the Triangle input")
    P, holes, regions = _build_pslg(plan, s.wall_segments, h_wall)
    seg = np.asarray(list(P.segments.keys()), dtype=np.int32).reshape(-1, 2)
    mark = np.asarray(list(P.segments.values()), dtype=np.int32)
    thin = _thin_channels(plan, s.gap_width)

    q = f"q{min_angle:g}"
    report(progress, 0.2, "Generating the triangle mesh (Triangle)")
    out = run(f"pzAQ{q}a{_SQ3_4 * s.h_max ** 2:.12g}",
              {"pointlist": np.asarray(P.points, float), "segmentlist": seg, "segmentmarkerlist": mark,
               "holelist": holes, "regionlist": regions})
    passes = 0
    for passes in range(1, max_passes + 1):
        pts = np.asarray(out["pointlist"], float).reshape(-1, 2)
        tri = np.asarray(out["trianglelist"], np.int32).reshape(-1, 3)
        cen = pts[tri].mean(axis=1)
        target = _SQ3_4 * _target_size(cen, plan, s, h_wall, thin) ** 2
        p3 = pts[tri]
        area = 0.5 * np.abs((p3[:, 1, 0] - p3[:, 0, 0]) * (p3[:, 2, 1] - p3[:, 0, 1])
                            - (p3[:, 2, 0] - p3[:, 0, 0]) * (p3[:, 1, 1] - p3[:, 0, 1]))
        over = area > 1.5 * target
        if not over.any() or (passes > 1 and over.mean() < 0.005):     # converged (< 0.5 % too large)
            break
        report(progress, 0.2 + 0.6 * passes / max_passes, f"Refining near capillaries (pass {passes})")
        out = run(f"przAQ{q}a",
                  {"pointlist": pts, "trianglelist": tri,
                   "triangleattributelist": np.asarray(out["triangleattributelist"], float).ravel(),
                   "trianglearealist": target,
                   "segmentlist": np.asarray(out["segmentlist"], np.int32).reshape(-1, 2),
                   "segmentmarkerlist": np.asarray(out["segmentmarkerlist"], np.int32).ravel()})

    report(progress, 0.9, "Extracting regions and boundaries")
    pts = np.asarray(out["pointlist"], float).reshape(-1, 2)
    tri = np.asarray(out["trianglelist"], np.int64).reshape(-1, 3)
    attr = np.rint(np.asarray(out["triangleattributelist"], float).ravel()).astype(int) - 1
    if np.any(attr < 0):
        raise RuntimeError("Triangle left triangles outside every region (open face boundary?)")
    region = np.asarray(plan.face_region, int)[attr]
    segs = np.asarray(out["segmentlist"], np.int64).reshape(-1, 2)
    smark = np.asarray(out["segmentmarkerlist"], np.int64).ravel()
    cap = smark >= MARKER_CAPILLARY0
    # drop unused vertices (e.g. eaten by holes) and renumber
    used = np.zeros(len(pts), bool)
    used[tri.ravel()] = True
    new = np.cumsum(used) - 1
    mesh = TissueMesh(points=pts[used], triangles=_ccw(pts[used], new[tri]), cell_region=region,
                      fiber_types=np.asarray(plan.fiber_types, int), capillary_facets=new[segs[cap]],
                      capillary_facet_owner=(smark[cap] - MARKER_CAPILLARY0).astype(int),
                      box_facets=new[segs[smark == MARKER_BOX]], capillaries=plan.capillaries,
                      rcap=plan.rcap, box=plan.box)
    mesh.info.update({"source": "triangle", "h_wall": h_wall, "h_max": s.h_max, "grade_to": s.grade_to,
                      "refine_passes": passes, "min_angle": min_angle,
                      "seconds": time.perf_counter() - t0, "notes": list(plan.notes),
                      "n_faces": len(plan.faces), "n_circle_capillaries": int(plan.circle_caps.size),
                      "n_polygon_capillaries": int(plan.polygon_caps.size)})
    report(progress, 1.0, f"Mesh ready: {mesh.n_nodes} nodes, {mesh.n_triangles} triangles")
    return mesh
