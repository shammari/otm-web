"""Repair of fibre outlines that cross or touch themselves (port of OTM 2026 ``otm.mesh.repairRing``).

Self-crossing outlines come from Dtect tracing, from the capillary-overlap
clean-up and from outline smoothing. Gmsh cannot mesh them (it loops for ever
recovering the boundary), and the data check reports large loops (DT-F07).

:func:`repair_ring` splits the ring where two non-adjacent edges cross or come
closer than ``tol`` and drops the smaller loop (by area), until the ring is
simple. It returns the repaired ring and the total area dropped.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["repair_ring", "ring_area"]

_TINY = np.finfo(float).tiny


def ring_area(r: np.ndarray) -> float:
    """Signed area of a closed ring (first vertex not repeated)."""
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(p: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (q[:, 0] - p[:, 0]) * (s[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (s[:, 0] - p[:, 0])


def _point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.sum((p - a) * ab, axis=1) / np.maximum(np.sum(ab ** 2, axis=1), _TINY)
    t = np.clip(t, 0.0, 1.0)
    c = a + t[:, None] * ab
    return np.sqrt(np.sum((p - c) ** 2, axis=1))


def _closest_points(P1, P2, Q1, Q2):
    best, c1, c2 = np.inf, P1, P1
    for p, a, b in ((Q1, P1, P2), (Q2, P1, P2), (P1, Q1, Q2), (P2, Q1, Q2)):
        ab = b - a
        t = min(max(float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), _TINY), 0.0), 1.0)
        c = a + t * ab
        d = float(np.hypot(*(p - c)))
        if d < best:
            best, c1, c2 = d, p, c
    return c1, c2


def _first_contact(r: np.ndarray, tol: float) -> Optional[Tuple[int, int, np.ndarray]]:
    """First pair of non-adjacent edges i < j (0-based edge i = r[i] -> r[i+1])
    that cross or come closer than ``tol``; with the contact point."""
    n = len(r)
    a, b = r, np.roll(r, -1, axis=0)
    I, J = np.triu_indices(n, 2)
    keep = ~((I == 0) & (J == n - 1))                     # first and last edges share vertex 0
    I, J = I[keep], J[keep]
    if I.size == 0:
        return None
    lo1, hi1 = np.minimum(a[I], b[I]), np.maximum(a[I], b[I])
    lo2, hi2 = np.minimum(a[J], b[J]), np.maximum(a[J], b[J])
    near = np.all(lo1 <= hi2 + tol, axis=1) & np.all(lo2 <= hi1 + tol, axis=1)
    I, J = I[near], J[near]
    if I.size == 0:
        return None
    p1, p2, q1, q2 = a[I], b[I], a[J], b[J]
    o1, o2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    o3, o4 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    d = np.min(np.column_stack([_point_seg(q1, p1, p2), _point_seg(q2, p1, p2),
                                _point_seg(p1, q1, q2), _point_seg(p2, q1, q2)]), axis=1)
    d[cross] = 0.0
    hits = np.flatnonzero(d < tol)
    if hits.size == 0:
        return None
    h = int(hits[0])
    i, j = int(I[h]), int(J[h])
    if cross[h]:
        t = o3[h] / (o3[h] - o4[h])                       # p1 + t (p2 - p1) lies on segment q
        X = p1[h] + t * (p2[h] - p1[h])
    else:
        c1, c2 = _closest_points(p1[h], p2[h], q1[h], q2[h])
        X = (c1 + c2) / 2.0
    return i, j, X


def repair_ring(r: np.ndarray, tol: float = 1e-9, max_passes: int = 100) -> Tuple[np.ndarray, float]:
    """Remove self-crossings and pinches; returns (ring, area removed).

    ``r`` is an (n, 2) closed ring without the repeated first vertex. Where two
    edges cross or touch (closer than ``tol``) the ring is split into two loops
    and the smaller one is dropped; repeated until the ring is simple."""
    r = np.asarray(r, float).copy()
    if len(r) > 1 and np.all(r[0] == r[-1]):
        r = r[:-1]
    removed = 0.0
    for _ in range(max_passes):
        if len(r) < 4:
            break
        hit = _first_contact(r, tol)
        if hit is None:
            break
        i, j, X = hit
        A = np.vstack([X, r[i + 1:j + 1]])                # X -> r(i+1 .. j)
        B = np.vstack([X, r[j + 1:], r[:i + 1]])          # X -> r(j+1 .. n-1, 0 .. i)
        aA, aB = abs(ring_area(A)), abs(ring_area(B))
        if aA >= aB:
            r, removed = A, removed + aB
        else:
            r, removed = B, removed + aA
        keep = np.r_[True, np.any(np.abs(np.diff(r, axis=0)) > 0, axis=1)]
        r = r[keep]
        if len(r) > 1 and np.all(r[0] == r[-1]):
            r = r[:-1]
    return r, removed
