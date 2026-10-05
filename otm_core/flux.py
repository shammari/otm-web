"""Oxygen flux lines (replaces ``mfiles/flux solver``).

Flux lines are streamlines of the oxygen flux direction ``-grad u`` (u = PO2/Pcap),
started on a small circle around every capillary of the ROI and followed
"downhill" into the tissue.

Legacy pipeline (``PO2FluxLinesModule`` -> ``GenerateFluxLines`` -> ``FluxSolver``)::

    [ux, uy] = pdegrad(p, t, u)                       % one gradient per triangle
    Ux = TriScatteredInterp(mesh2midpt(p,t)', -ux', 'natural')   % on centroids
    Uy = TriScatteredInterp(mesh2midpt(p,t)', -uy', 'natural')
    ODE = ODEConditions(...)                          % Ni seeds per ROI capillary
    FluxSolver: fixed-step Heun, stop rules checked on every proposed step

This module provides

* :func:`element_gradients` / :func:`element_centroids` (``pdegrad`` / ``mesh2midpt``);
* :class:`FluxField`, the interpolated vector field ``-grad u``:

  - ``"natural"`` (default): Sibson natural-neighbour interpolation of the element
    gradients on the centroids, i.e. exactly what ``TriScatteredInterp(..., 'natural')``
    does (:class:`NaturalNeighbourInterpolator`);
  - ``"linear"``: ``matplotlib.tri.LinearTriInterpolator`` on the Delaunay
    triangulation of the centroids (``TriScatteredInterp(..., 'linear')``);
  - ``"p1"``: ``LinearTriInterpolator`` on the FEM mesh itself with the nodal PO2;
    its ``gradient`` is the exact P1 gradient of the containing element
    (piecewise constant, no smoothing);

* two integrators (:func:`compute_flux_lines`, ``FluxSettings.integrator``):

  - ``"heun"`` (default): the legacy fixed-step Heun scheme with the legacy stop
    rules, checked in the legacy order on every proposed step. Reproduces the
    MATLAB streams to round-off;
  - ``"ivp"``: ``scipy.integrate.solve_ivp`` (adaptive RK) with **terminal event
    functions**: the line reaches a capillary border, leaves the ROI box, leaves
    the interpolation domain, or stalls.

How far lines run (``FluxSettings.parametrization``)

* ``"arc_length"`` (default): unit speed, ``dx/ds = -grad u / |grad u|``, so the
  parameter is the length travelled. A line stops at its PO2 minimum or at the
  length cap ``max_length`` ("auto" = 1.25 x the mean nearest-neighbour
  capillary distance, ~30 um in the regression sample). Past that length lines
  run along the boundaries between supply regions into the minimum: long,
  practically 1-D tails that add length but almost no supplied area (the cap
  keeps 99 % of the covered area with ~36 % less line length).
* ``"time"`` (legacy): ``dx/dt = -grad u`` up to ``tfinal``. Lines slow down where
  the gradient flattens, so they stop well before the low-PO2 regions
  (regression sample: median 55 % of the path to the minimum).

All coordinates are in the non-dimensional (ndim) model frame.

Legacy behaviour worth knowing (kept in ``"heun"`` mode, reported per line):

* A step is rejected when it lands within ``2*max(Rcap)`` of *any* capillary
  centre, including the seed's own. The legacy seed radius (0.015 ndim) lies
  inside that zone unless the tissue is taller than ~480 um, so most lines stop
  on their very first step (310 of 368 in the regression sample; only 15 of 46
  capillaries get lines). The default here is therefore an automatic seed radius
  of 2.2 x the capillary radius (``FluxSettings.initial_radius = None``), always
  outside the zone; ``FluxSettings.legacy()`` restores 0.015 for regression runs.
  ``"ivp"`` mode only stops on *entering* a zone.
* ``Streams`` always has ``Ns = 1 + tfinal/dS`` rows; after a stop the last
  accepted point is repeated to the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .progress import ProgressCallback, report

__all__ = [
    "FluxSettings", "FluxField", "FluxLines", "NaturalNeighbourInterpolator",
    "STOP_REASONS", "compute_flux_lines", "element_centroids", "element_gradients",
    "heun_step", "roi_capillary_index", "seed_points",
]

STOP_REASONS = ("tfinal", "capillary", "box", "domain", "stagnation", "backtrack", "skipped", "length")
"""Why a flux line ended:

* ``tfinal``     ran to the end of the time parameter (``parametrization="time"``)
* ``capillary``  reached the stop zone around a capillary (``capillary_margin * max(Rcap)``)
* ``box``        would leave the model box (ROI box)
* ``domain``     left the region where the field is defined (outside the centroid hull)
* ``stagnation`` the step did not move the point (``"heun"``) / speed below ``stall_speed``
* ``backtrack``  legacy rule: the first step fell back into the seed capillary
* ``skipped``    legacy rule: seed exactly at (0, 0) is not integrated
* ``length``     reached the length cap (``parametrization="arc_length"``)

In arc-length mode ``stagnation`` means the line reached its PO2 minimum
(|grad u| below ``stall_fraction`` x the largest element gradient, or the
direction turned back within one step).
"""


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


@dataclass
class FluxSettings:
    """Flux-line options. Defaults are the ``FEMSimulation`` ``inputdlg`` defaults,
    except the seed radius (see ``initial_radius``); :meth:`legacy` gives all of them.

    ``initial_radius``: radius of the seed circle (ndim). ``None`` (default) means
    ``auto_radius_factor * max(Rcap)`` (2.2 x the capillary radius), which puts every
    seed outside the ``capillary_margin * max(Rcap)`` stop zone whatever the image
    size. The legacy fixed 0.015 lies *inside* that zone whenever the tissue is
    shorter than ~480 um, and most lines then stop on their first step
    (only 15 of 46 capillaries get lines in the regression sample).
    """

    parametrization: str = "arc_length"  # "arc_length" (unit speed) | "time" (legacy dx/dt = -grad u)
    max_length: Any = "auto"            # arc_length: cap in ndim; "auto" = factor x mean capillary NN distance; None = no cap
    max_length_factor: float = 1.25     # "auto" cap = factor x mean nearest-neighbour capillary distance
    stall_fraction: float = 1e-3        # arc_length: stop when |grad u| < fraction x max element |grad u|
    tfinal: float = 1.0                 # time mode: parameter range (legacy "termination time")
    step_size: float = 0.005            # Heun step / output spacing (legacy dS)
    lines_per_capillary: int = 8        # seeds per ROI capillary (legacy Ni)
    initial_radius: Optional[float] = None   # seed circle radius, ndim; None -> automatic (legacy 0.015)
    auto_radius_factor: float = 2.2     # automatic seed radius = factor * max(Rcap)
    interpolation: str = "natural"      # "natural" | "linear" | "p1"
    integrator: str = "heun"            # "heun" | "ivp"
    capillary_margin: float = 2.0       # stop zone radius = margin * max(Rcap) (legacy 2)
    # solve_ivp only
    ivp_method: str = "RK45"
    ivp_max_step: Optional[float] = None    # parameter units; None -> see _ivp_max_step
    rtol: float = 1e-8
    atol: float = 1e-11
    stall_speed: float = 1e-10          # |grad u| below this ends the line

    def __post_init__(self):
        if self.interpolation not in ("natural", "linear", "p1"):
            raise ValueError("interpolation must be 'natural', 'linear' or 'p1'")
        if self.integrator not in ("heun", "ivp"):
            raise ValueError("integrator must be 'heun' or 'ivp'")
        if self.step_size <= 0 or self.tfinal <= 0 or self.lines_per_capillary < 1:
            raise ValueError("tfinal, step_size and lines_per_capillary must be positive")
        if self.initial_radius is not None and self.initial_radius <= 0:
            raise ValueError("initial_radius must be positive (or None for automatic)")
        if self.parametrization not in ("arc_length", "time"):
            raise ValueError("parametrization must be 'arc_length' or 'time'")
        if not (self.max_length is None or self.max_length == "auto"
                or (isinstance(self.max_length, (int, float)) and self.max_length > 0)):
            raise ValueError("max_length must be 'auto', None or a positive length (ndim)")
        if self.auto_radius_factor <= 1:
            raise ValueError("auto_radius_factor must be > 1 (seeds outside the capillary)")

    LEGACY_INITIAL_RADIUS = 0.015

    @classmethod
    def legacy(cls, **changes) -> "FluxSettings":
        """Exactly the MATLAB defaults: fixed seed radius 0.015 ndim and the time
        parametrization dx/dt = -grad u up to ``tfinal``."""
        changes.setdefault("initial_radius", cls.LEGACY_INITIAL_RADIUS)
        changes.setdefault("parametrization", "time")
        return cls(**changes)

    def seed_radius(self, rcap) -> float:
        """Seed circle radius for capillaries of radius ``rcap`` (ndim).

        Automatic: ``auto_radius_factor * max(Rcap)``, raised to 1.1 x the stop
        zone if ``capillary_margin`` is set above the factor (seeds must start outside it).
        """
        if self.initial_radius is not None:
            return float(self.initial_radius)
        r = float(np.max(np.asarray(rcap, dtype=float)))
        return max(self.auto_radius_factor * r, 1.1 * self.capillary_margin * r)

    def length_cap(self, capillaries: np.ndarray, box: Optional[np.ndarray] = None) -> float:
        """Resolved arc-length cap (ndim) for ``parametrization="arc_length"``.

        ``"auto"``: ``max_length_factor`` x mean nearest-neighbour distance of the
        capillaries. Beyond about that length a line runs along the boundary
        between two supply regions into the PO2 minimum: a practically 1-D tail
        that adds length but no supplied area (regression sample: area within
        5 um of a line saturates at 30 um = 1.25 x the 24 um mean spacing, while
        38 % of the full drawn length lies beyond it). ``None``: no cap (the line
        runs to its minimum; bounded by 4 box diagonals as a safeguard).
        """
        caps = np.asarray(capillaries, dtype=float).reshape(-1, 2)
        if box is not None and len(np.asarray(box)):
            b = np.asarray(box, dtype=float).reshape(-1, 2)
            diag = float(np.hypot(*np.ptp(b, axis=0)))
        else:
            diag = float(np.hypot(*np.ptp(caps, axis=0))) if len(caps) > 1 else 1.0
        if self.max_length is None:
            return 4.0 * diag
        if self.max_length != "auto":
            return float(self.max_length)
        if len(caps) < 2:
            return diag
        from scipy.spatial import cKDTree

        d = cKDTree(caps).query(caps, k=2)[0][:, 1]
        return float(self.max_length_factor * d.mean())

    @property
    def n_samples(self) -> int:
        """``Ns = 1 + Sf/dS`` (rows per stream)."""
        return 1 + int(round(self.tfinal / self.step_size))


# --------------------------------------------------------------------------
# element gradients (pdegrad) and centroids (mesh2midpt)
# --------------------------------------------------------------------------


def element_centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """``mesh2midpt``: ``(p1 + p2 + p3) / 3`` per triangle, shape (M, 2)."""
    p = np.asarray(points, float)
    t = np.asarray(triangles)
    return (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3


def element_gradients(points: np.ndarray, triangles: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``pdegrad``: gradient of the P1 field ``u`` on every triangle, shape (M, 2).

    Uses the ``pdetrg`` shape-function gradients (``g_i = +-0.5 * edge / area``),
    in the same operation order.
    """
    p = np.asarray(points, float)
    t = np.asarray(triangles)
    u = np.asarray(u, float).ravel()
    x1, y1 = p[t[:, 0], 0], p[t[:, 0], 1]
    x2, y2 = p[t[:, 1], 0], p[t[:, 1], 1]
    x3, y3 = p[t[:, 2], 0], p[t[:, 2], 1]
    r21x, r21y = x2 - x1, y2 - y1
    r31x, r31y = x3 - x1, y3 - y1
    r32x, r32y = x3 - x2, y3 - y2
    ar = np.abs(r21x * r31y - r31x * r21y) / 2
    g1x, g1y = -0.5 * r32y / ar, 0.5 * r32x / ar
    g2x, g2y = 0.5 * r31y / ar, -0.5 * r31x / ar
    g3x, g3y = -0.5 * r21y / ar, 0.5 * r21x / ar
    u1, u2, u3 = u[t[:, 0]], u[t[:, 1]], u[t[:, 2]]
    ux = u1 * g1x + u2 * g2x + u3 * g3x
    uy = u1 * g1y + u2 * g2y + u3 * g3y
    return np.column_stack([ux, uy])


# --------------------------------------------------------------------------
# natural-neighbour (Sibson) interpolation
# --------------------------------------------------------------------------


class NaturalNeighbourInterpolator:
    """Sibson natural-neighbour interpolation of scattered data, NaN outside the hull.

    Matches ``TriScatteredInterp(X, V, 'natural')`` / ``scatteredInterpolant`` with
    ``ExtrapolationMethod = 'none'``. ``values`` may have several columns
    (all interpolated with the same weights).

    For a query point q the Delaunay triangles whose circumcircle contains q form
    the cavity; q's Voronoi cell steals from each cavity vertex v a convex region
    whose corners are the circumcentres of the cavity triangles around v and the
    circumcentres of (q, v, w) for v's two neighbours w on the cavity boundary.
    The weights are those areas.
    """

    def __init__(self, points: np.ndarray, values: np.ndarray):
        from scipy.spatial import Delaunay

        self.points = np.ascontiguousarray(points, dtype=float)
        v = np.asarray(values, dtype=float)
        self._squeeze = v.ndim == 1
        self.values = v.reshape(len(self.points), -1)
        self.tri = Delaunay(self.points)
        s = self.tri.simplices
        self._cc, self._r2 = _circumcircles(self.points[s[:, 0]], self.points[s[:, 1]], self.points[s[:, 2]])
        # plain Python containers: the per-query work is scalar and branchy
        self._simp = s.tolist()
        self._nbr = self.tri.neighbors.tolist()
        self._P = self.points.tolist()
        self._ccl = self._cc.tolist()
        self._r2l = self._r2.tolist()

    def weights(self, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Natural neighbours (indices) and their normalised Sibson weights.
        Empty arrays outside the convex hull."""
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return np.zeros(0, int), np.zeros(0)
        s0 = int(self.tri.find_simplex(np.array([[x, y]]))[0])
        if s0 < 0:
            return np.zeros(0, int), np.zeros(0)
        simp, nbr, P, cc, r2 = self._simp, self._nbr, self._P, self._ccl, self._r2l
        for v in simp[s0]:
            if P[v][0] == x and P[v][1] == y:
                return np.array([v]), np.array([1.0])
        # cavity: triangles whose circumcircle contains q (connected, contains s0)
        cav = {s0}
        stack = [s0]
        while stack:
            t = stack.pop()
            for n in nbr[t]:
                if n >= 0 and n not in cav:
                    dx, dy = cc[n][0] - x, cc[n][1] - y
                    if dx * dx + dy * dy < r2[n]:
                        cav.add(n)
                        stack.append(n)
        # boundary edges of the cavity (oriented as in the CCW triangles)
        succ: Dict[int, int] = {}
        incident: Dict[int, List[int]] = {}
        for t in cav:
            a, b, c = simp[t]
            incident.setdefault(a, []).append(t)
            incident.setdefault(b, []).append(t)
            incident.setdefault(c, []).append(t)
            na, nb, nc = nbr[t]
            if na < 0 or na not in cav:
                succ[b] = c
            if nb < 0 or nb not in cav:
                succ[c] = a
            if nc < 0 or nc not in cav:
                succ[a] = b
        pred = {j: i for i, j in succ.items()}
        idx = list(succ.keys())
        w = []
        for v in idx:
            pts = [cc[t] for t in incident[v]]
            pts.append(_circumcentre(x, y, P[v], P[succ[v]]))
            pts.append(_circumcentre(x, y, P[pred[v]], P[v]))
            w.append(_convex_area(pts))
        w_arr = np.array(w)
        tot = w_arr.sum()
        if not math.isfinite(tot) or tot <= 0:
            return np.zeros(0, int), np.zeros(0)
        return np.array(idx), w_arr / tot

    def __call__(self, x: float, y: float) -> np.ndarray:
        idx, w = self.weights(x, y)
        if idx.size == 0:
            out = np.full(self.values.shape[1], np.nan)
        else:
            out = w @ self.values[idx]
        return out[0] if self._squeeze else out


def _circumcircles(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0] - ax, b[:, 1] - ay
    cx, cy = c[:, 0] - ax, c[:, 1] - ay
    d = 2 * (bx * cy - by * cx)
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return np.column_stack([ax + ux, ay + uy]), ux * ux + uy * uy


def _circumcentre(qx: float, qy: float, b, c) -> Tuple[float, float]:
    bx, by = b[0] - qx, b[1] - qy
    cx, cy = c[0] - qx, c[1] - qy
    d = 2 * (bx * cy - by * cx)
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    return (qx + (cy * b2 - by * c2) / d, qy + (bx * c2 - cx * b2) / d)


def _convex_area(pts) -> float:
    """Area of the convex polygon with corners ``pts`` (any order)."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    ring = sorted(pts, key=lambda p: math.atan2(p[1] - my, p[0] - mx))
    a = 0.0
    for i in range(n):
        x1, y1 = ring[i - 1]
        x2, y2 = ring[i]
        a += x1 * y2 - x2 * y1
    return 0.5 * abs(a)


# --------------------------------------------------------------------------
# the vector field -grad u
# --------------------------------------------------------------------------


class FluxField:
    """Interpolated flux direction ``-grad u`` (NaN where undefined).

    Parameters
    ----------
    points, triangles, u
        FEM mesh (N x 2, M x 3, 0-based) and nodal solution.
    interpolation
        ``"natural"`` | ``"linear"`` | ``"p1"`` (see the module docstring).
    """

    def __init__(self, points: np.ndarray, triangles: np.ndarray, u: np.ndarray,
                 interpolation: str = "natural"):
        import matplotlib.tri as mtri

        self.interpolation = interpolation
        self.points = np.asarray(points, float)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.u = np.asarray(u, float).ravel()
        self.gradients = element_gradients(self.points, self.triangles, self.u)
        self.centroids = element_centroids(self.points, self.triangles)
        self.max_speed = float(np.hypot(self.gradients[:, 0], self.gradients[:, 1]).max()) \
            if len(self.gradients) else 0.0
        self._u_interp_cache = None
        self._cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        if interpolation == "p1":
            self._u_interp                  # build now
        if interpolation == "natural":
            self._nn = NaturalNeighbourInterpolator(self.centroids, -self.gradients)
        elif interpolation == "linear":
            from scipy.spatial import Delaunay

            d = Delaunay(self.centroids)
            tri = mtri.Triangulation(self.centroids[:, 0], self.centroids[:, 1], d.simplices)
            self._fx = mtri.LinearTriInterpolator(tri, -self.gradients[:, 0])
            self._fy = mtri.LinearTriInterpolator(tri, -self.gradients[:, 1])
        elif interpolation != "p1":
            raise ValueError("interpolation must be 'natural', 'linear' or 'p1'")

    @property
    def _u_interp(self):
        """``LinearTriInterpolator`` of u on the FEM mesh (built on first use)."""
        if self._u_interp_cache is None:
            import matplotlib.tri as mtri

            tri = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.triangles)
            self._u_interp_cache = mtri.LinearTriInterpolator(tri, self.u)
        return self._u_interp_cache

    @classmethod
    def from_solution(cls, solution, interpolation: str = "natural") -> "FluxField":
        """From a :class:`otm_core.fem.PO2Solution`."""
        return cls(solution.mesh.points, solution.mesh.triangles, solution.u, interpolation)

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        """``(-du/dx, -du/dy)`` at (x, y); ``(nan, nan)`` outside the field's domain."""
        key = (float(x), float(y))
        hit = self._cache.get(key)
        if hit is None:
            hit = self._evaluate(*key)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def _evaluate(self, x: float, y: float) -> Tuple[float, float]:
        if self.interpolation == "natural":
            v = self._nn(x, y)
            return float(v[0]), float(v[1])
        if self.interpolation == "linear":
            fx, fy = self._fx(x, y), self._fy(x, y)
            if np.ma.is_masked(fx) or np.ma.is_masked(fy):
                return math.nan, math.nan
            return float(fx), float(fy)
        gx, gy = self._u_interp.gradient(x, y)
        if np.ma.is_masked(gx) or np.ma.is_masked(gy):
            return math.nan, math.nan
        return -float(gx), -float(gy)

    def u_at(self, x, y):
        """P1 value of u at points (NaN outside the mesh)."""
        v = self._u_interp(np.atleast_1d(x), np.atleast_1d(y))
        return np.ma.filled(v.astype(float), np.nan)


# --------------------------------------------------------------------------
# seeds, ROI capillaries
# --------------------------------------------------------------------------


def roi_capillary_index(tissue_capillaries: np.ndarray, roi_index: Sequence[int],
                        model_capillaries: np.ndarray) -> np.ndarray:
    """``GetModelROICapillaryIndex``: model capillaries (0-based, ascending) whose
    centres are *exactly* those of the tissue capillaries in the ROI."""
    xt = np.asarray(tissue_capillaries, float)[np.asarray(roi_index, dtype=int)]
    xm = np.asarray(model_capillaries, float)
    hit = (xt[:, None, 0] == xm[None, :, 0]) & (xt[:, None, 1] == xm[None, :, 1])
    return np.flatnonzero(hit.any(axis=0))


def seed_points(centres: np.ndarray, radius: float, n: int) -> np.ndarray:
    """``ODEConditions``: ``n`` seeds per centre at angles ``2*pi*j/n`` (j = 1..n).
    Returns (n_centres, n, 2)."""
    c = np.asarray(centres, float).reshape(-1, 2)
    j = np.arange(1, n + 1, dtype=float)
    ang = 2 * math.pi * j / n
    xs = radius * np.cos(ang)[None, :] + c[:, 0:1]
    ys = radius * np.sin(ang)[None, :] + c[:, 1:2]
    return np.stack([xs, ys], axis=-1)


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class FluxLines:
    """Flux lines from ``Nf`` capillaries x ``Ni`` seeds."""

    streams: np.ndarray                 # (Nf, Ni, Ns, 2), legacy layout (last point repeated)
    seeds: np.ndarray                   # (Nf, Ni, 2)
    capillary_index: np.ndarray         # (Nf,) model capillary of each seed group (0-based)
    stop_reason: np.ndarray             # (Nf, Ni) entries of STOP_REASONS
    n_steps: np.ndarray                 # (Nf, Ni) accepted steps
    arc_parameter: np.ndarray           # (Nf, Ni) parameter value where the line ended
    settings: FluxSettings
    paths: List[List[np.ndarray]] = field(default_factory=list)   # exact path incl. event point
    seed_radius: float = math.nan       # radius of the seed circles actually used (ndim)
    max_length: float = math.nan        # resolved arc-length cap (ndim); NaN in time mode

    @property
    def n_capillaries(self) -> int:
        return int(self.streams.shape[0])

    def lengths(self) -> np.ndarray:
        """Polyline length of every stream (ndim)."""
        d = np.diff(self.streams, axis=2)
        return np.sqrt((d ** 2).sum(axis=-1)).sum(axis=-1)

    def summary(self) -> Dict[str, Any]:
        """Same quantities as the baseline's ``stageC_flux.summary``."""
        nonempty = np.any(self.streams != 0, axis=(2, 3))
        reasons = {r: int((self.stop_reason == r).sum()) for r in STOP_REASONS}
        return {"n_seed_capillaries": self.n_capillaries,
                "n_nonempty_lines": int(nonempty.sum()),
                "total_length_ndim": float(self.lengths()[nonempty].sum()),
                "n_moving_lines": int((self.n_steps > 0).sum()),
                "n_capillaries_with_lines": int((self.n_steps > 0).any(axis=1).sum()),
                "seed_radius_ndim": float(self.seed_radius),
                "max_length_ndim": float(self.max_length),
                "stop_reasons": {k: v for k, v in reasons.items() if v}}

    def to_legacy(self) -> List[List[np.ndarray]]:
        """``FluxData(i).IC(j).Streams`` as nested lists of (Ns, 2) arrays."""
        return [[self.streams[i, j] for j in range(self.streams.shape[1])] for i in range(self.n_capillaries)]


# --------------------------------------------------------------------------
# integrators
# --------------------------------------------------------------------------


def heun_step(field: FluxField, x: float, y: float, h: float) -> Tuple[float, float]:
    """``HeunMethod``: one explicit trapezoidal step; returns the start point when
    the field is NaN at either stage."""
    zx, zy, _ = _heun(field, x, y, h)
    return zx, zy


def _heun(field, x, y, h):
    h1, v1 = field(x, y)
    if math.isnan(v1) or math.isnan(h1):
        return x, y, True
    h2, v2 = field(x + h * h1, y + h * v1)
    if math.isnan(v2) or math.isnan(h2):
        return x, y, True
    return x + (h / 2) * (h1 + h2), y + (h / 2) * (v1 + v2), False


def _trace_heun(field: FluxField, seed: np.ndarray, own: np.ndarray, own_r: float, caps: np.ndarray,
                r_stop: float, box: Tuple[float, float, float, float], s: FluxSettings):
    """``FluxSolver`` inner loop for one seed. Returns (stream, reason, n_steps)."""
    ns = s.n_samples
    X = np.zeros(ns)
    Y = np.zeros(ns)
    X[0], Y[0] = seed
    if X[0] == 0 and Y[0] == 0:
        return np.zeros((ns, 2)), "skipped", 0
    xmin, xmax, ymin, ymax = box
    reason, n = "tfinal", ns - 1
    for k in range(1, ns):
        zx, zy, undefined = _heun(field, X[k - 1], Y[k - 1], s.step_size)
        dx, dy = zx - caps[:, 0], zy - caps[:, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) <= r_stop):
            reason = "capillary"
        elif k == 1 and math.sqrt((zx - own[0]) ** 2 + (zy - own[1]) ** 2) <= own_r:
            # legacy sets X = [] here (and then fails on the assignment); report an empty line
            return np.zeros((ns, 2)), "backtrack", 0
        elif math.isnan(zx) or math.isnan(zy) or (zx == X[k - 1] and zy == Y[k - 1]):
            reason = "domain" if undefined else "stagnation"
        elif zx < xmin or zx > xmax or zy < ymin or zy > ymax:
            reason = "box"
        else:
            X[k], Y[k] = zx, zy
            continue
        X[k:] = X[k - 1]
        Y[k:] = Y[k - 1]
        n = k - 1
        break
    return np.column_stack([X, Y]), reason, n


class _UnitField:
    """Unit flux direction ``-grad u / |grad u|`` (arc-length parametrization)."""

    def __init__(self, field: FluxField):
        self.field = field

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        fx, fy = self.field(x, y)
        n = math.hypot(fx, fy)
        if not n > 0:                       # NaN outside, or exactly zero at a minimum
            return (fx, fy) if math.isnan(n) else (0.0, 0.0)
        return fx / n, fy / n


def _trace_heun_arc(field: FluxField, seed: np.ndarray, caps: np.ndarray, r_stop: float,
                    box: Tuple[float, float, float, float], ds: float, length: float, ns: int, g_min: float):
    """Heun in arc length: fixed steps of length ``ds`` (last one shortened) until the
    length cap, a capillary zone, the box, the end of the field, or the PO2 minimum.
    Returns (stream padded to ``ns``, reason, n_steps, arc length reached)."""
    xmin, xmax, ymin, ymax = box
    x, y = float(seed[0]), float(seed[1])
    pts = [(x, y)]
    s_done, reason = 0.0, "length"
    while length - s_done > 1e-12 * max(1.0, length):
        gx, gy = field(x, y)
        g = math.hypot(gx, gy)
        if math.isnan(g):
            reason = "domain"
            break
        if g < g_min:
            reason = "stagnation"
            break
        h = min(ds, length - s_done)
        t1x, t1y = gx / g, gy / g
        px, py = x + h * t1x, y + h * t1y
        qx, qy = field(px, py)
        q = math.hypot(qx, qy)
        if math.isnan(q):
            reason = "domain"
            break
        if q < g_min or (qx * t1x + qy * t1y) < 0:   # passed the minimum within this step
            reason = "stagnation"
            break
        zx = x + (h / 2) * (t1x + qx / q)
        zy = y + (h / 2) * (t1y + qy / q)
        dx, dy = zx - caps[:, 0], zy - caps[:, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) <= r_stop):
            reason = "capillary"
            break
        if zx < xmin or zx > xmax or zy < ymin or zy > ymax:
            reason = "box"
            break
        s_done += h                          # arc-length parameter (unit speed)
        x, y = zx, zy
        pts.append((x, y))
    P = np.asarray(pts)
    n = len(P) - 1
    stream = np.empty((ns, 2))
    stream[:len(P)] = P
    stream[len(P):] = P[-1]
    return stream, reason, n, s_done


def _ivp_max_step(field: FluxField, r_stop: float, s: FluxSettings) -> float:
    """Events are only seen as a sign change between two accepted steps, so no step
    may jump across a capillary stop zone: at most half its radius at the fastest
    speed of the field (all three interpolations stay within the element-gradient
    range), and never more than 10 legacy steps."""
    if s.ivp_max_step is not None:
        return float(s.ivp_max_step)
    h = 10 * s.step_size
    if r_stop > 0 and field.max_speed > 0:
        h = min(h, 0.5 * r_stop / field.max_speed)
    return h


def _trace_ivp(field: FluxField, seed: np.ndarray, caps: np.ndarray, r_stop: float,
               box: Tuple[float, float, float, float], s: FluxSettings,
               span: Optional[float] = None, ns: Optional[int] = None, g_min: Optional[float] = None):
    """solve_ivp with terminal events. Returns (stream, reason, n_samples_moved, t_end, path).

    Time mode: ``dx/dt = -grad u`` on [0, tfinal]. Arc-length mode (``span`` = the
    length cap): ``dx/ds`` = unit direction on [0, span]; a cap reached is "length".
    """
    from scipy.integrate import solve_ivp

    xmin, xmax, ymin, ymax = box
    arc = s.parametrization == "arc_length"
    span = float(span if span is not None else s.tfinal)
    ns = int(ns if ns is not None else s.n_samples)
    g_min = float(g_min if g_min is not None else s.stall_speed)
    drive = _UnitField(field) if arc else field

    def rhs(t, z):
        fx, fy = drive(z[0], z[1])
        if math.isnan(fx) or math.isnan(fy):
            return [0.0, 0.0]
        return [fx, fy]

    def ev_capillary(t, z):
        d = np.sqrt((z[0] - caps[:, 0]) ** 2 + (z[1] - caps[:, 1]) ** 2)
        return float(d.min() - r_stop)

    def ev_box(t, z):
        return float(min(z[0] - xmin, xmax - z[0], z[1] - ymin, ymax - z[1]))

    def ev_domain(t, z):
        fx, _ = field(z[0], z[1])
        return -1.0 if math.isnan(fx) else 1.0

    def ev_stall(t, z):
        fx, fy = field(z[0], z[1])
        if math.isnan(fx):
            return 1.0                         # handled by ev_domain
        return math.hypot(fx, fy) - g_min

    events = [ev_capillary, ev_box, ev_domain, ev_stall]
    names = ["capillary", "box", "domain", "stagnation"]
    for e in events:
        e.terminal = True
        e.direction = -1

    t_eval = np.minimum(np.arange(ns) * (span / (ns - 1) if not arc else s.step_size), span)
    t_eval[-1] = span
    if ev_box(0, seed) < 0:
        return np.tile(seed, (ns, 1)), "box", 0, 0.0, seed[None, :]
    if ev_domain(0, seed) < 0:
        return np.tile(seed, (ns, 1)), "domain", 0, 0.0, seed[None, :]
    max_step = _ivp_max_step(field, r_stop, s)
    if arc:                                     # unit speed: the step bound is a length
        max_step = min(10 * s.step_size, 0.5 * r_stop) if r_stop > 0 else 10 * s.step_size
        if s.ivp_max_step is not None:
            max_step = float(s.ivp_max_step)
    sol = solve_ivp(rhs, (0.0, span), np.asarray(seed, float), method=s.ivp_method, t_eval=t_eval,
                    events=events, rtol=s.rtol, atol=s.atol, max_step=max_step)
    pts = sol.y.T
    reason, t_end, end = ("length" if arc else "tfinal"), span, None
    if sol.status == 1:
        hits = [(te[0], k) for k, te in enumerate(sol.t_events) if len(te)]
        t_first = min(h[0] for h in hits)
        # simultaneous events (e.g. the box edge is also the mesh edge): first in `names` wins
        k = min(k for te, k in hits if te <= t_first + 1e-9 * max(1.0, span))
        t_end = sol.t_events[k][0]
        reason = names[k]
        end = sol.y_events[k][0]
    stream = np.empty((ns, 2))
    m = len(pts)
    stream[:m] = pts
    stream[m:] = end if end is not None else pts[-1]
    path = np.vstack([pts, end[None, :]]) if end is not None else pts
    return stream, reason, m - 1, float(t_end), path


def compute_flux_lines(field: FluxField, capillaries: np.ndarray, rcap: np.ndarray,
                       roi_capillaries: Sequence[int], box: np.ndarray,
                       settings: Optional[FluxSettings] = None,
                       progress: Optional[ProgressCallback] = None) -> FluxLines:
    """``GenerateFluxLines`` + ``FluxSolver``.

    Parameters
    ----------
    field
        :class:`FluxField` built from the PO2 solution.
    capillaries, rcap
        All model capillaries (``ModelSolution.Xcap``/``Rcap``), ndim.
    roi_capillaries
        0-based indices of the capillaries to seed from (:func:`roi_capillary_index`).
    box
        Model box polygon (5 x 2) or any array whose min/max give the bounds
        (``TissueBox.Rect``).
    progress
        Optional ``progress(fraction, message)``, called after every capillary
        (see :mod:`otm_core.progress`).
    """
    s = settings or FluxSettings()
    caps = np.asarray(capillaries, float).reshape(-1, 2)
    r = np.asarray(rcap, float).ravel()
    if r.size == 1:
        r = np.full(len(caps), float(r[0]))
    roi = np.asarray(roi_capillaries, dtype=int).ravel()
    b = np.asarray(box, float).reshape(-1, 2)
    bounds = (b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max())
    r_stop = s.capillary_margin * r.max()
    r0 = s.seed_radius(r)
    seeds = seed_points(caps[roi], r0, s.lines_per_capillary)
    arc = s.parametrization == "arc_length"
    length = s.length_cap(caps, b) if arc else math.nan
    g_min = s.stall_fraction * field.max_speed if arc else s.stall_speed
    ns = 1 + int(math.ceil(length / s.step_size - 1e-9)) if arc else s.n_samples
    nf, ni = len(roi), s.lines_per_capillary
    streams = np.zeros((nf, ni, ns, 2))
    reason = np.empty((nf, ni), dtype=object)
    steps = np.zeros((nf, ni), dtype=int)
    t_end = np.zeros((nf, ni))
    paths: List[List[np.ndarray]] = []
    for i in range(nf):
        row = []
        for j in range(ni):
            if s.integrator == "heun" and arc:
                st, why, n, te = _trace_heun_arc(field, seeds[i, j], caps, r_stop, bounds, s.step_size,
                                                 length, ns, g_min)
                path = st[:n + 1]
            elif s.integrator == "heun":
                st, why, n = _trace_heun(field, seeds[i, j], caps[roi[i]], r[roi[i]], caps, r_stop, bounds, s)
                te = n * s.step_size
                path = st[:n + 1]
            else:
                st, why, n, te, path = _trace_ivp(field, seeds[i, j], caps, r_stop, bounds, s,
                                                  span=length if arc else None, ns=ns, g_min=g_min)
            streams[i, j], reason[i, j], steps[i, j], t_end[i, j] = st, why, n, te
            row.append(path)
        paths.append(row)
        report(progress, (i + 1) / max(nf, 1), f"Flux lines: capillary {i + 1} of {nf}")
    return FluxLines(streams=streams, seeds=seeds, capillary_index=roi, stop_reason=reason,
                     n_steps=steps, arc_parameter=t_end, settings=s, paths=paths, seed_radius=r0,
                     max_length=length)


def flux_lines_from_solution(solution, roi_capillaries: Sequence[int],
                             settings: Optional[FluxSettings] = None,
                             progress: Optional[ProgressCallback] = None) -> FluxLines:
    """Convenience: field + lines from a :class:`otm_core.fem.PO2Solution`."""
    from .progress import scaled

    s = settings or FluxSettings()
    m = solution.mesh
    report(progress, 0.0, f"Building the flux field ({s.interpolation} interpolation)")
    f = FluxField.from_solution(solution, s.interpolation)
    report(progress, 0.15, "Integrating flux lines")
    return compute_flux_lines(f, m.capillaries, m.rcap, roi_capillaries, m.box, s,
                              progress=scaled(progress, 0.15, 1.0))


__all__.append("flux_lines_from_solution")
