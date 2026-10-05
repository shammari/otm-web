"""Geometry ingest, clean-up and frame transforms (report sections 3.3 and 4.2).

Legacy routines ported here, in pipeline order:

=====================================  =========================================
MATLAB                                 Python
=====================================  =========================================
TissueGeometryType (swap/flip/centre)  :func:`dtect_to_centred`
CorrectCaps + PolyProject              :func:`correct_capillaries`
EliminateDuplicatesCapillaries         :func:`eliminate_duplicate_capillaries`
KMovingAverage / KWindowSmoothing      :func:`k_moving_average` / :func:`k_window_smoothing`
DouglasPeucker / ReduceFiberVertices   :func:`douglas_peucker` / :func:`reduce_fiber_vertices`
NeighboringFibers (+min_dist_between_  :func:`neighbouring_fibers` (Shapely distance,
two_polygons, curveintersect)          KD-tree centroid pre-filter)
SingleIS                               :func:`single_is`
EnlargeISLocally + p_poly_dist         :func:`enlarge_is_locally`, :func:`signed_distance_to_polygon`
ScaleGeometry                          :func:`scale_geometry`
RetouchFibers > smoothfibers           :func:`retouch_fibers`
RemoveCapillaryFibreOverlap            :func:`remove_capillary_fibre_overlap` (Shapely
(polybool/polyjoin/polysplit/          difference / union_all / get_parts, STRtree
ispolycw/poly2cw)                      pre-filter; orientation is irrelevant in GEOS)
GetModelCapillaries / GetModelFibers   :func:`select_model_capillaries` / :func:`select_model_fibers`
inpolygon (base MATLAB)                :func:`inpolygon` (exact in/on semantics)
=====================================  =========================================

Parity policy: where the legacy result depends on the *exact* vertex sequence
(smoothing, Douglas-Peucker, vertex deletion in SingleIS/EnlargeISLocally) the
algorithm is ported verbatim in NumPy so the baseline can be matched to
round-off. Polygon Boolean algebra and distance queries, which do not depend
on vertex order, use Shapely 2.x.

The retouch stage (smoothing ... EnlargeISLocally) works on single rings, as
in the legacy code. Multi-part fibres (lists of rings or Shapely
MultiPolygons) are accepted from :func:`fiber_geometries` onwards.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import shapely
from scipy.spatial import KDTree

from .models import (
    ImageFrame,
    IngestResult,
    LengthScale,
    RetouchResult,
    RetouchSettings,
    Ring,
)

MultipartPolicy = Literal["largest_vertices", "largest_area", "keep"]

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def as_ring(xy) -> Ring:
    """Accept (n,2) arrays, (x, y) tuples, dicts or objects with .x/.y
    (e.g. a MATLAB struct loaded by scipy); return an (n, 2) float array."""
    if hasattr(xy, "x") and hasattr(xy, "y") and not isinstance(xy, np.ndarray):
        return np.column_stack([np.ravel(xy.x), np.ravel(xy.y)]).astype(float)
    if isinstance(xy, dict):
        return np.column_stack([np.ravel(xy["x"]), np.ravel(xy["y"])]).astype(float)
    if (isinstance(xy, tuple) and len(xy) == 2 and np.ndim(xy[0]) == 1
            and np.ndim(xy[1]) == 1 and len(xy[0]) == len(xy[1])):
        return np.column_stack([np.ravel(xy[0]), np.ravel(xy[1])]).astype(float)
    a = np.asarray(xy, dtype=float)
    if a.ndim != 2:
        raise ValueError("ring must be 2-D")
    if a.shape[1] == 2:
        return a
    if a.shape[0] == 2:
        return a.T
    raise ValueError(f"ring must be (n, 2) or (2, n), got {a.shape}")


def is_multipart(f) -> bool:
    """True for a list of rings (a multi-part fibre in ring form)."""
    return isinstance(f, list) and len(f) > 0 and np.ndim(f[0]) == 2


def is_closed(ring: Ring) -> bool:
    return len(ring) > 1 and ring[0, 0] == ring[-1, 0] and ring[0, 1] == ring[-1, 1]


def close_ring(ring: Ring) -> Ring:
    """Append the first vertex if the ring is open (MATLAB tests x AND y equal)."""
    if len(ring) and not is_closed(ring):
        return np.vstack([ring, ring[:1]])
    return ring


def polyarea(ring: Ring) -> float:
    """MATLAB ``polyarea``: |shoelace| of the vertex sequence (net area for
    self-intersecting rings, exactly like the legacy code)."""
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _segments(xv: np.ndarray, yv: np.ndarray):
    """Edges of a closed polygon (closing it if needed)."""
    if xv[0] != xv[-1] or yv[0] != yv[-1]:
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
    return xv[:-1], yv[:-1], xv[1:], yv[1:]


def inpolygon(x, y, xv, yv, chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[in, on] = inpolygon(x, y, xv, yv)`` for a single contour.

    ``in`` is True for points strictly inside **or** on the boundary
    (even-odd rule, like MATLAB); ``on`` is True only on the boundary.
    Non-finite polygon vertices (NaN/Inf) are dropped.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    xv = np.asarray(xv, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[finite], yv[finite]
    inside = np.zeros(x.size, dtype=bool)
    on = np.zeros(x.size, dtype=bool)
    if xv.size < 3 or x.size == 0:
        return inside, on
    x1, y1, x2, y2 = _segments(xv, yv)
    scale = max(np.ptp(xv), np.ptp(yv), 1.0)
    tol = 1e-12 * scale
    for s in range(0, x.size, chunk):
        px = x[s:s + chunk, None]
        py = y[s:s + chunk, None]
        # on-boundary test (collinear and within the segment's bounding box)
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        within = ((px >= np.minimum(x1, x2) - tol) & (px <= np.maximum(x1, x2) + tol) &
                  (py >= np.minimum(y1, y2) - tol) & (py <= np.maximum(y1, y2) + tol))
        seg_len = np.hypot(x2 - x1, y2 - y1)
        on_c = (np.abs(cross) <= tol * np.maximum(seg_len, 1.0)) & within
        # crossing number (even-odd)
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        crossings = np.count_nonzero(cond & (px < xint), axis=1)
        o = on_c.any(axis=1)
        on[s:s + chunk] = o
        inside[s:s + chunk] = (crossings % 2 == 1) | o
    return inside, on


def closest_point_on_ring(px: float, py: float, ring: Ring) -> Tuple[float, float, float]:
    """Nearest point on a closed polyline (PolyProject / p_poly_dist core).
    Returns (distance, x, y)."""
    x1, y1, x2, y2 = _segments(ring[:, 0], ring[:, 1])
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(L2 > 0, ((px - x1) * dx + (py - y1) * dy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = x1 + t * dx, y1 + t * dy
    d = np.hypot(cx - px, cy - py)
    k = int(np.argmin(d))
    return float(d[k]), float(cx[k]), float(cy[k])


def signed_distance_to_polygon(points: np.ndarray, ring: Ring) -> np.ndarray:
    """``p_poly_dist``: distance to the polygon boundary, negative inside
    (inside includes the boundary, where the distance is 0 anyway)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if len(pts) == 0:
        return np.zeros(0)
    x1, y1, x2, y2 = _segments(ring[:, 0], ring[:, 1])
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    px, py = pts[:, 0:1], pts[:, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(L2 > 0, ((px - x1) * dx + (py - y1) * dy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = np.hypot(x1 + t * dx - px, y1 + t * dy - py).min(axis=1)
    inside, _ = inpolygon(pts[:, 0], pts[:, 1], ring[:, 0], ring[:, 1])
    return np.where(inside, -d, d)


# --------------------------------------------------------------------------
# Frames (report 3.3)
# --------------------------------------------------------------------------


def dtect_to_centred(capillaries_px: np.ndarray, fibers_px: Sequence, frame: ImageFrame,
                     capillaries_from_dtect: bool = True):
    """TissueGeometryType coordinate transform (px -> px_c).

    Exactly as the MATLAB code (note the asymmetry between the two inputs):

    * fibres:      ``x' = fib.y - W/2``,     ``y' = -fib.x + H/2``
    * capillaries: ``x' = Xcap(:,1) - W/2``, ``y' = -Xcap(:,2) + H/2``

    ``capillaries_from_dtect=False`` reproduces the "Capillaries Only"
    branch when the user answers *No* to "produced by Dtect?" (no transform).
    """
    W, H = frame.width, frame.height
    caps = np.asarray(capillaries_px, dtype=float)
    if caps.ndim == 2 and caps.shape[0] < caps.shape[1]:
        caps = caps.T
    if capillaries_from_dtect:
        out = np.empty_like(caps)
        out[:, 0] = caps[:, 0] - W / 2.0
        out[:, 1] = -caps[:, 1] + H / 2.0
    else:
        out = caps.copy()
    fibers = []
    for f in fibers_px:
        r = as_ring(f)          # columns: legacy .x, .y
        fibers.append(np.column_stack([r[:, 1] - W / 2.0, -r[:, 0] + H / 2.0]))
    return out, fibers


def scale_geometry(fibers: Sequence[Ring], capillaries: np.ndarray, frame: ImageFrame):
    """ScaleGeometry: px_c -> ndim (divide by ``min(ImageSize)/2``)."""
    s = frame.ndim_scale_px
    return [np.asarray(f, float) / s for f in fibers], np.asarray(capillaries, float) / s


def ndim_to_um(xy: np.ndarray, frame: ImageFrame, ls: LengthScale) -> np.ndarray:
    """Display frame used by every legacy plot: ``ld * (ndim + ss)``."""
    return ls.ld * (np.asarray(xy, float) + frame.ss)


def um_to_ndim(xy: np.ndarray, frame: ImageFrame, ls: LengthScale) -> np.ndarray:
    """Inverse of :func:`ndim_to_um`."""
    return np.asarray(xy, float) / ls.ld - frame.ss


def ndim_to_centred_um(xy: np.ndarray, ls: LengthScale) -> np.ndarray:
    """Frame of the index routines (``LengthScale = Ly/2``, origin at centre)."""
    return np.asarray(xy, float) * ls.index_length_um


def ndim_box(ls: LengthScale) -> np.ndarray:
    """``NdimBox`` of SkeletalFunctionalHeterogeneity (closed 5x2, +0.01 margin).
    Kept verbatim: both half-sides are divided by Ly/2."""
    by = (ls.y_um / (ls.y_um / 2.0)) / 2.0 + 0.01
    bx = (ls.x_um / (ls.y_um / 2.0)) / 2.0 + 0.01
    return np.array([[-bx, by], [bx, by], [bx, -by], [-bx, -by], [-bx, by]])


# --------------------------------------------------------------------------
# Ingest (TissueGeometryType)
# --------------------------------------------------------------------------


def eliminate_duplicate_capillaries(caps: np.ndarray, tol: float) -> np.ndarray:
    """EliminateDuplicatesCapillaries: greedy, order-preserving. For each
    surviving capillary in turn, every *other* capillary within ``tol``
    (inclusive) is removed."""
    y = np.asarray(caps, dtype=float).copy()
    n = 0
    while n < len(y) - 1:
        d = np.hypot(y[:, 0] - y[n, 0], y[:, 1] - y[n, 1])
        rm = d <= tol
        rm[n] = False
        if rm.any():
            y = y[~rm]
        n += 1
    return y


def correct_capillaries(fibers: Sequence[Ring], caps: np.ndarray) -> np.ndarray:
    """CorrectCaps: capillaries strictly inside a fibre are moved to the
    nearest point of that fibre's outline. Membership is always tested on
    the *original* positions; if several fibres contain a point the last
    fibre wins (legacy loop order)."""
    caps = np.asarray(caps, dtype=float)
    out = caps.copy()
    for ring in fibers:
        ring = as_ring(ring)
        if len(ring) < 3:
            continue
        inside, on = inpolygon(caps[:, 0], caps[:, 1], ring[:, 0], ring[:, 1])
        for i in np.flatnonzero(inside & ~on):
            _, xn, yn = closest_point_on_ring(caps[i, 0], caps[i, 1], ring)
            out[i] = (xn, yn)
    return out


def ingest_dtect(capillaries_px: np.ndarray, fibers_px: Sequence, image_size: Sequence[int],
                 fiber_types: Optional[Iterable[int]] = None, dedup_tol_px: float = 5.0,
                 capillaries_from_dtect: bool = True) -> IngestResult:
    """Full TissueGeometryType ingest (manual editing steps excluded).

    With fibres (skeletal branches) the Dtect transform, CorrectCaps and the
    5-px de-duplication always run. Without fibres ("Capillaries Only") the
    transform and de-duplication only run when ``capillaries_from_dtect``.
    """
    rows, cols = (int(v) for v in np.ravel(image_size)[:2])
    frame = ImageFrame((rows, cols))
    raw = np.asarray(capillaries_px, dtype=float)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    has_fibres = len(fibers_px) > 0
    from_dtect = capillaries_from_dtect or has_fibres
    caps_c, fib_c = dtect_to_centred(raw, fibers_px, frame, from_dtect)
    ft = None if fiber_types is None else np.ravel(np.asarray(fiber_types))
    types = (np.ones(len(fib_c), dtype=int) if ft is None or ft.size == 0
             else ft.astype(int))
    corrected = correct_capillaries(fib_c, caps_c) if has_fibres else caps_c.copy()
    final = eliminate_duplicate_capillaries(corrected, dedup_tol_px) if from_dtect else corrected
    return IngestResult(frame=frame, capillaries_px_raw=raw, capillaries_pxc_centred=caps_c,
                        capillaries_pxc_corrected=corrected, capillaries_pxc=final,
                        fibers_pxc=fib_c, fiber_types=types)


# --------------------------------------------------------------------------
# Retouch (RetouchFibers > smoothfibers)
# --------------------------------------------------------------------------


def _mrange(a: int, b: int) -> List[int]:
    """MATLAB ``a:b`` (inclusive, empty if a > b)."""
    return list(range(a, b + 1)) if a <= b else []


def k_moving_average(ring: Ring, k: int) -> Ring:
    """KMovingAverage: circular k-point moving average (k odd), returned closed.
    The window construction is ported index-for-index (1-based) from MATLAB;
    for odd k it equals a centred circular moving average."""
    k = int(k)
    if k % 2 == 0:
        raise ValueError("KMovingAverage window must be odd (legacy colon ranges)")
    x, y = ring[:, 0].copy(), ring[:, 1].copy()
    if x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]
    Nx = x.size
    wL, wR = -(k - 1) // 2, (k - 1) // 2
    xs = np.zeros(Nx)
    ys = np.zeros(Nx)
    for loop in range(1, Nx + 1):
        if wL + loop <= 0:
            a = abs(wL + loop)
            n_roll = len(_mrange(Nx - a, Nx))
            lshift = wR - n_roll
            idx = _mrange(Nx - a, Nx) + _mrange(loop - lshift, loop) + _mrange(loop + 1, loop + wR)
        elif wR + loop > Nx:
            rshift = len(_mrange(1, abs(Nx - wR - loop)))
            idx = _mrange(loop + wL, loop - 1) + _mrange(loop, Nx) + _mrange(1, rshift)
        else:
            idx = _mrange(loop + wL, loop + wR)
        ii = np.asarray(idx, dtype=int) - 1
        xs[loop - 1] = x[ii].mean()
        ys[loop - 1] = y[ii].mean()
    out = np.column_stack([xs, ys])
    return np.vstack([out, out[:1]])


def k_window_smoothing(fibers: Sequence[Ring], k: int) -> List[Ring]:
    """KWindowSmoothing: smooth only when ``ceil(100*k/numel(x)) < 10``
    (i.e. the window is < ~10 % of the outline); otherwise leave as is."""
    out = []
    for f in fibers:
        f = as_ring(f)
        if f.shape[0] and math.ceil(100.0 * k / f.shape[0]) < 10:
            out.append(k_moving_average(f, k))
        else:
            out.append(f.copy())
    return out


def _abs_det3_matlab(px: np.ndarray, py: np.ndarray, ax: float, ay: float,
                     bx: float, by: float) -> np.ndarray:
    """``abs(det([1 1 1; px ax bx; py ay by]))`` with MATLAB's rounding.

    MATLAB's ``det`` is Gaussian elimination with partial pivoting (first
    largest pivot), multipliers formed by *division*, no fused multiply-add,
    and ``prod(diag(U))`` taken left to right. Pixel outlines contain many
    collinear runs where several vertices are exactly tied in Douglas-Peucker;
    which one wins is decided by this rounding, so the elimination is
    replicated step by step (vectorised over points) instead of using the
    algebraically equal cross product. Verified against the MATLAB baseline:
    89/89 fibres identical (the cross product or np.linalg.det give 12-16
    differing fibres). The sign is irrelevant because of ``abs``.
    """
    m = px.size
    A = np.empty((m, 3, 3))
    A[:, 0, :] = 1.0
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = px, ax, bx
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = py, ay, by
    rows = np.arange(m)
    for k in range(2):
        p = k + np.argmax(np.abs(A[:, k:, k]), axis=1)          # first largest pivot
        swap = p != k
        if swap.any():
            r = rows[swap]
            tmp = A[r, k, :].copy()
            A[r, k, :] = A[r, p[swap], :]
            A[r, p[swap], :] = tmp
        piv = A[:, k, k]
        nz = piv != 0                                           # zero pivot: LAPACK skips the column
        safe = np.where(nz, piv, 1.0)
        for i in range(k + 1, 3):
            A[:, i, k] = np.where(nz, A[:, i, k] / safe, 0.0)  # division, not reciprocal
            for j in range(k + 1, 3):
                A[:, i, j] = A[:, i, j] - A[:, i, k] * A[:, k, j]   # two ufuncs: no FMA
    return np.abs((A[:, 0, 0] * A[:, 1, 1]) * A[:, 2, 2])


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """DouglasPeucker (File Exchange) on an *open* point list. Same tie rule
    (first maximum), same degenerate-chord rule (point distance when the
    chord length <= eps) and the same floating-point evaluation of the
    perpendicular distance (see :func:`_abs_det3_matlab`). Iterative, so no
    recursion limit."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n <= 2:
        return P.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    eps = np.finfo(float).eps
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        A, B = P[a], P[b]
        seg = P[a + 1:b]
        dn = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        if dn > eps:
            d = _abs_det3_matlab(seg[:, 0], seg[:, 1], A[0], A[1], B[0], B[1]) / dn
        else:
            d = np.sqrt((seg[:, 0] - A[0]) ** 2 + (seg[:, 1] - A[1]) ** 2)
        j = int(np.argmax(d))       # first maximum, like `if d > dmax`
        if d[j] > epsilon:
            idx = a + 1 + j
            keep[idx] = True
            stack.append((idx, b))
            stack.append((a, idx))
    return P[keep]


def reduce_fiber_vertices(fibers: Sequence[Ring], tol: float, max_vertices: int = 50,
                          step: float = 0.5) -> List[Ring]:
    """ReduceFiberVertices: Douglas-Peucker on the closed vertex list, raising
    the tolerance by ``n*0.5`` until at most 50 vertices remain."""
    out = []
    for f in fibers:
        f = as_ring(f)
        z = douglas_peucker(f, tol)
        n = 0
        while len(z) > max_vertices:
            n += 1
            z = douglas_peucker(f, tol + n * step)
        out.append(z)
    return out


def _outline(f):
    """Closed outline of one ring as a LineString (Point if a single vertex)."""
    r = close_ring(as_ring(f))
    if len(r) >= 2:
        return shapely.linestrings(r)
    if len(r) == 1:
        return shapely.points(r[0])
    return None


def neighbouring_fibers(fibers: Sequence[Ring], centroid_radius: float = 100.0,
                        max_gap: float = 10.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """NeighboringFibers (pixel units).

    Candidates: vertex-mean centroids closer than ``centroid_radius`` (strict),
    found with a KD-tree instead of a full ``pdist2``. Their outline gap is
    the Shapely distance between the closed outlines (0 when they cross),
    which equals ``min_dist_between_two_polygons`` (outline-to-outline, so a
    fibre nested in another is *not* at distance 0). Neighbours with a gap
    > ``max_gap`` are dropped. Returns, per fibre, (ids ascending, gaps).
    """
    fibers = [as_ring(f) for f in fibers]
    n = len(fibers)
    empty = (np.zeros(0, int), np.zeros(0))
    cent = np.array([f.mean(axis=0) if len(f) else (np.nan, np.nan) for f in fibers]).reshape(-1, 2)
    ok = np.isfinite(cent).all(axis=1)
    if not ok.any():
        return [empty for _ in range(n)]
    tree = KDTree(cent[ok])
    okidx = np.flatnonzero(ok)
    lines = np.asarray([_outline(f) for f in fibers], dtype=object)
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        if not ok[i]:
            result.append(empty)
            continue
        cand = okidx[np.asarray(tree.query_ball_point(cent[i], centroid_radius), dtype=int)]
        dc = np.hypot(*(cent[cand] - cent[i]).T) if cand.size else np.zeros(0)
        cand = np.sort(cand[(dc < centroid_radius) & (dc > 0)])
        if cand.size == 0:
            result.append((cand, np.zeros(0)))
            continue
        gaps = np.asarray(shapely.distance(lines[i], lines[cand]), dtype=float)
        keep = gaps <= max_gap          # NaN (degenerate outline) is dropped
        result.append((cand[keep], gaps[keep]))
    return result


def single_is(fibers: Sequence[Ring]) -> List[Ring]:
    """SingleIS: delete every vertex that lies in or on a neighbouring fibre.
    Sequential like MATLAB (later fibres see earlier fibres' edits)."""
    F = [as_ring(f).copy() for f in fibers]
    nb = neighbouring_fibers(F)
    for i in range(len(F)):
        X = F[i]
        for j in nb[i][0]:
            V = F[j]
            if len(X) == 0 or len(V) < 3:
                continue
            inside, _ = inpolygon(X[:, 0], X[:, 1], V[:, 0], V[:, 1])
            X = X[~inside]
        F[i] = X
    return F


def enlarge_is_locally(fibers: Sequence[Ring], tol: float) -> List[Ring]:
    """EnlargeISLocally: for each pair of fibres closer than ``tol`` (each
    unordered pair handled once, in legacy order), delete the vertices of the
    second fibre that are within ``tol`` of (or inside) the first. The
    closing vertex is never tested; the ring is re-closed when needed."""
    F = [as_ring(f).copy() for f in fibers]
    nb = neighbouring_fibers(F)
    pairs: List[List[int]] = []
    for i, (ids, gaps) in enumerate(nb):
        for j in ids[gaps < tol]:
            pairs.append([i, int(j)])
    # verbatim port of the reverse-pair removal loop (1-based counter, the
    # list shrinks while it is scanned)
    counter = 1
    while counter < len(pairs) / 2 + 1:
        a, b = pairs[counter - 1]
        pairs = [p for p in pairs if not (p[1] == a and p[0] == b)]
        counter += 1
    for a, b in pairs:
        xv = F[a]
        x = F[b]
        if len(x) < 2 or len(xv) < 3:
            continue
        d = signed_distance_to_polygon(x[:-1], xv)
        rm = np.zeros(len(x), dtype=bool)
        rm[:-1] = d <= tol
        x = x[~rm]
        # verbatim legacy test (`x(end)~=x(1) && y(end)~=y(1)`): re-close only
        # when BOTH coordinates differ
        if len(x) and x[-1, 0] != x[0, 0] and x[-1, 1] != x[0, 1]:
            x = np.vstack([x, x[:1]])
        F[b] = x
    return F


def retouch_fibers(settings: RetouchSettings, fibers_pxc: Sequence[Ring],
                   capillaries_pxc: np.ndarray, frame: ImageFrame) -> RetouchResult:
    """RetouchFibers > smoothfibers (each stage is the identity when off)."""
    smooth = (k_window_smoothing(fibers_pxc, int(settings.smooth_tol)) if settings.smooth
              else [as_ring(f).copy() for f in fibers_pxc])
    reduced = (reduce_fiber_vertices(smooth, settings.reduce_tol, settings.max_vertices,
                                     settings.reduce_tol_step) if settings.reduce else smooth)
    disjoint = single_is(reduced) if settings.disjoint else reduced
    tangent = enlarge_is_locally(disjoint, settings.tangent_tol) if settings.tangent else disjoint
    rescaled, caps = scale_geometry(tangent, capillaries_pxc, frame)
    return RetouchResult(smooth=smooth, reduced=reduced, disjoint=disjoint, tangent=tangent,
                         rescaled=rescaled, capillaries_ndim=caps)


# --------------------------------------------------------------------------
# Shapely conversion (multi-polygon aware)
# --------------------------------------------------------------------------


def _is_geom(o) -> bool:
    """Scalar 'is this one Shapely geometry?' (shapely.is_geometry is vectorised)."""
    return isinstance(o, shapely.Geometry)


def _polygonal(g):
    """Keep only the areal part of a geometry (Polygon / MultiPolygon or empty)."""
    if g is None or shapely.is_empty(g):
        return shapely.Polygon()
    t = shapely.get_type_id(g)
    if t in (3, 6):        # Polygon, MultiPolygon
        return g
    parts = [p for p in shapely.get_parts(g) if shapely.get_type_id(p) in (3, 6)]
    if not parts:
        return shapely.Polygon()
    return shapely.union_all(parts)


def _ring_polygon(r) -> "shapely.Polygon":
    r = as_ring(r)
    r = r[np.isfinite(r).all(axis=1)]
    if len(np.unique(r, axis=0)) < 3:
        return shapely.Polygon()
    return shapely.Polygon(close_ring(r))


def fiber_geometries(fibers: Sequence, repair: bool = True) -> np.ndarray:
    """Fibres -> object array of Shapely Polygon/MultiPolygon.

    Each entry may be a ring, a list of rings (multi-part fibre, parts are
    unioned) or a Shapely geometry. Self-intersecting outlines (common after
    smoothing/vertex deletion) are repaired with ``make_valid`` and reduced
    to their areal part, so Boolean ops never raise. Degenerate fibres
    (< 3 distinct vertices) become empty polygons and never overlap anything.
    """
    out = []
    for f in fibers:
        if _is_geom(f):
            g = f
        elif is_multipart(f):
            parts = [p for p in (_ring_polygon(r) for r in f) if not p.is_empty]
            if repair:
                parts = [_polygonal(shapely.make_valid(p)) if not p.is_valid else p for p in parts]
            g = shapely.union_all(parts) if parts else shapely.Polygon()
        else:
            g = _ring_polygon(f)
        if repair and not shapely.is_empty(g) and not shapely.is_valid(g):
            g = _polygonal(shapely.make_valid(g))
        out.append(g)
    arr = np.empty(len(out), dtype=object)
    arr[:] = out
    return arr


def geometry_parts_as_rings(g) -> List[Ring]:
    """Exterior rings of every polygon part (holes are ignored, like polysplit
    on the legacy outer contours)."""
    return [np.asarray(p.exterior.coords) for p in shapely.get_parts(g) if not p.is_empty]


def fiber_area_um2(fiber, ls: LengthScale, legacy: bool = True) -> float:
    """Fibre area in um^2.

    ``legacy=True``: ``area_factor * polyarea`` of the raw vertex sequence,
    summed over parts for a multi-part fibre (LocalCapillaryToFiberRatio).
    ``legacy=False`` (or a Shapely input): area of the repaired geometry.
    """
    if legacy and not _is_geom(fiber):
        if is_multipart(fiber):
            return ls.area_factor * sum(polyarea(as_ring(r)) for r in fiber)
        return ls.area_factor * polyarea(as_ring(fiber))
    g = fiber if _is_geom(fiber) else fiber_geometries([fiber])[0]
    return ls.area_factor * float(shapely.area(g))


# --------------------------------------------------------------------------
# FEM pre-processing (SkeletalFunctionalHeterogeneity)
# --------------------------------------------------------------------------


def _ngon(cx, cy, r, step=np.pi / 6):
    t = np.arange(0.0, 2 * np.pi + 1e-12, step)          # 0:pi/6:2*pi (13 points)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def capillary_exclusion_zone(capillaries: np.ndarray, rcap):
    """Union of the shapes RemoveCapillaryFibreOverlap subtracts from fibres.

    * a 12-gon of radius ``1.5*Rcap`` around every isolated capillary;
    * for each pair of capillaries with ``0 < d <= 4*max(Rcap)``
      (findAdjacentCaps) an ellipse (12-gon) with semi-axes
      ``a = 3.5*max(Rcap)``, ``b = a/2``, centred at the midpoint and aligned
      with the pair; capillaries that belong to any pair get no circle.

    Legacy passes all shapes to ``polybool`` as one NaN-separated
    multi-contour clip polygon; overlapping contours are treated here as a
    union (the geometric intent).
    """
    X = np.asarray(capillaries, float)
    if X.ndim == 2 and X.shape[0] == 2 and X.shape[1] != 2:
        X = X.T                                          # legacy passes Xcap' (2 x N)
    if len(X) == 0:
        return shapely.Polygon()
    rc = np.asarray(rcap, float).ravel()
    rc = np.full(len(X), rc[0]) if rc.size == 1 else rc
    r = float(np.max(rc))
    pairs = np.array(sorted(KDTree(X).query_pairs(4 * r)), dtype=int).reshape(-1, 2)
    if len(pairs):
        d = np.hypot(*(X[pairs[:, 0]] - X[pairs[:, 1]]).T)
        pairs = pairs[(d > 0) & (d <= 4 * r)]
    in_pair = np.zeros(len(X), bool)
    in_pair[pairs.ravel()] = True
    shapes = [shapely.Polygon(_ngon(X[k, 0], X[k, 1], 1.5 * rc[k])) for k in np.flatnonzero(~in_pair)]
    a, b = 3.5 * r, 3.5 * r / 2.0
    t = np.arange(0.0, 2 * np.pi + 1e-12, np.pi / 6)
    for i, j in pairs:
        (x1, y1), (x2, y2) = X[i], X[j]
        phi = math.atan2(y1 - y2, x1 - x2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ex = cx + a * math.cos(phi) * np.cos(t) - b * math.sin(phi) * np.sin(t)
        ey = cy + a * math.sin(phi) * np.cos(t) + b * math.cos(phi) * np.sin(t)
        shapes.append(shapely.Polygon(np.column_stack([ex, ey])))
    return shapely.union_all(shapes) if shapes else shapely.Polygon()


def _merge_close_vertices(poly, rtol: float = 1e-12):
    """Drop consecutive vertices that coincide to round-off, as ``polybool`` does.

    GEOS can emit an intersection point twice with coordinates that differ in
    the last bit; MATLAB's ``polybool`` merges them. Keeping them would change
    the vertex count, which the legacy code uses (largest part of a split fibre,
    ``FiberMatrix`` order).
    """
    def clean(ring):
        r = np.asarray(ring.coords)[:-1]
        if len(r) < 3:
            return r
        tol = rtol * max(1.0, float(np.abs(r).max()))
        keep = np.ones(len(r), dtype=bool)
        last = 0
        for i in range(1, len(r)):
            if np.abs(r[i] - r[last]).max() <= tol:
                keep[i] = False
            else:
                last = i
        if keep.sum() > 1 and np.abs(r[last] - r[0]).max() <= tol:
            keep[last] = False
        return r[keep]

    ext = clean(poly.exterior)
    if len(ext) < 3:
        return shapely.Polygon()
    holes = [h for h in (clean(i) for i in poly.interiors) if len(h) >= 3]
    return shapely.Polygon(ext, holes)


def remove_capillary_fibre_overlap(fibers: Sequence, capillaries: np.ndarray, rcap,
                                   multipart: MultipartPolicy = "largest_vertices"):
    """RemoveCapillaryFibreOverlap: ``fibre \\ exclusion_zone`` with Shapely.

    Only fibres that intersect the zone are touched (one STRtree query
    instead of one ``polybool`` per fibre). When a fibre is split:

    * ``'largest_vertices'`` (legacy): keep the part with the most vertices,
      as a simple polygon (``polysplit`` treats holes as separate contours
      and the outer contour is kept);
    * ``'largest_area'``: keep the biggest part (holes kept);
    * ``'keep'``: keep the whole (Multi)Polygon.

    Returns ``(geoms, rings)``: Shapely geometries and, per fibre, the
    exterior ring of the kept polygon (or a list of rings for ``'keep'``).
    """
    geoms = fiber_geometries(fibers)
    zone = capillary_exclusion_zone(capillaries, rcap)
    out = geoms.copy()
    touched = np.zeros(len(out), dtype=bool)
    if not shapely.is_empty(zone):
        tree = shapely.STRtree(shapely.get_parts(zone))
        hit = np.unique(tree.query(geoms, predicate="intersects")[0])
        if hit.size:
            out[hit] = shapely.difference(geoms[hit], zone)
            touched[hit] = True
    rings: List = []
    for k in range(len(out)):
        g = _polygonal(out[k])
        parts = [p for p in shapely.get_parts(g) if not p.is_empty]
        if touched[k]:
            parts = [q for q in (_merge_close_vertices(p) for p in parts) if not q.is_empty]
        if multipart == "keep":
            rings.append([np.asarray(p.exterior.coords) for p in parts])
        else:
            if multipart == "largest_vertices":
                kept = (shapely.Polygon(max(parts, key=lambda p: len(p.exterior.coords)).exterior)
                        if parts else shapely.Polygon())
            else:
                kept = max(parts, key=lambda p: p.area) if parts else shapely.Polygon()
            g = kept
            rings.append(np.asarray(kept.exterior.coords) if parts else np.zeros((0, 2)))
        out[k] = g
    return out, rings


def select_model_capillaries(capillaries: np.ndarray, box: np.ndarray):
    """GetModelCapillaries: keep capillaries in or on the box. Returns
    (capillaries, 0-based indices)."""
    caps = np.asarray(capillaries, float)
    inside, _ = inpolygon(caps[:, 0], caps[:, 1], box[:, 0], box[:, 1])
    idx = np.flatnonzero(inside)
    return caps[idx], idx


def _vertices(f) -> np.ndarray:
    if _is_geom(f):
        return shapely.get_coordinates(f)
    if is_multipart(f):
        return np.vstack([as_ring(r) for r in f]) if f else np.zeros((0, 2))
    return as_ring(f)


def select_model_fibers(fibers: Sequence, box: np.ndarray):
    """GetModelFibers: keep fibres with at least one vertex in or on the box
    (any part of a multi-part fibre counts). Returns (fibres, 0-based indices)."""
    keep = []
    for k, f in enumerate(fibers):
        v = _vertices(f)
        if len(v) and inpolygon(v[:, 0], v[:, 1], box[:, 0], box[:, 1])[0].any():
            keep.append(k)
    keep_arr = np.asarray(keep, dtype=int)
    return [fibers[k] for k in keep_arr], keep_arr
