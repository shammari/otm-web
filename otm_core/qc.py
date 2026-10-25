"""Data check of a Dtect export: problems that would make OTM results wrong.

Port of the OTM 2026 (MATLAB) data check (``otm.qc``), with the same problem
codes, severities, thresholds and report format, so a report from either
program names the same problem the same way.

Typical use::

    from otm_core.io import load_dtect_mat
    from otm_core import qc

    d = load_dtect_mat(path)
    res = qc.check(d, tissue="skeletal", use_fibre_types=True)
    if qc.is_blocking(res)[0]:
        d, action = qc.apply_fixes(d, res)          # remove the flagged items
        res = qc.check(d, ..., previous=res, action=action)
    print(qc.text(res))

The check runs on the file as exported, before OTM changes anything (in
particular before capillaries inside fibres are moved onto the outlines).
Positions are in image pixels (x to the right, y down, as in Dtect); fibres and
capillaries are named by their number in the original file (1-based).

Severities: ``fatal`` (the file cannot be analysed), ``error`` (the results
would be wrong: the analysis waits until the flagged items are removed or,
where allowed, kept by the user), ``warning`` (worth a look).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import json
import os
import platform
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ringfix import repair_ring

__all__ = ["CATALOGUE", "DEFAULTS", "MODES", "mode_for", "check", "not_dtect", "is_blocking", "override", "apply_fixes",
           "text", "headline", "write_report", "flagged_items", "catalogue"]

FORMAT, FORMAT_VERSION = "otm-data-check", 1
MODES = ("CapillariesAndFibreTypes", "CapillariesAndFibres", "CapillariesOnly")
_ORDER = ("ok", "warning", "error", "fatal")

# ---------------------------------------------------------------------------
# thresholds (chosen well clear of real sections: in the OTM sample data, 6 Dtect
# exports with 68-108 fibres, the largest fibre is at most 3.9 x the median area
# and 5 % of the image, no outline encloses another fibre, at most 3 capillaries
# lie inside any outline, and the capillary-to-fibre ratio is 1.0-1.6)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, float] = dict(
    edgeTolPx=2,             # positions this far outside the image still count as inside
    outsideFatal=0.5,        # this fraction (or more) of points outside -> wrong frame (fatal)
    enclosedFibres=1,        # an outline enclosing this many other fibre centres -> DT-F01
    sizeRatio=15,            # fibre area / median fibre area -> DT-F02
    imageFraction=0.25,      # fibre area / image area -> DT-F02
    minFibresForSize=5,      # the median needs at least this many fibres
    capsInside=6,            # capillaries strictly inside one outline -> DT-F03
    insideDepthPx=2,         # ... deeper than this from the outline
    duplicateOverlap=0.7,    # area ratio of two copies of one fibre -> DT-F04
    overlapWarn=0.25,        # share of a fibre's outline points inside another -> DT-F05
    tinyRatio=0.05,          # fibre area / median fibre area -> DT-F06
    tinyPx=20,               # or an area below this (px^2)
    loopPx=20,               # self-crossing loops smaller than this (px^2) ...
    loopFraction=0.01,       # ... and than this share of the fibre are ignored -> DT-F07
    dedupTolPx=5,            # capillary merge distance (as the data step)
    duplicateCapFraction=0.2,  # share of capillaries with a twin -> DT-C01
    farDiameters=1,          # capillary further than this many median fibre diameters -> DT-C02
    cfLow=0.3,               # capillary-to-fibre ratio limits -> DT-C03
    cfHigh=6,
    minCapillaries=3,        # DT-S04
)

# ---------------------------------------------------------------------------
# the problems
# ---------------------------------------------------------------------------
_RESEG = "Re-segment this image in Dtect and export it again"


def _item(code, severity, fix, overridable, title, why, advice):
    return dict(code=code, severity=severity, fix=fix, overridable=overridable, title=title, why=why, advice=advice)


CATALOGUE: Dict[str, Dict[str, Any]] = {c["code"]: c for c in [
    # ---- file structure
    _item("DT-S01", "fatal", "", False, "Not a Dtect export",
          "OTM needs the capillary centres (Xcap), the image size (ImageSize) and, for skeletal muscle, "
          "the fibre outlines (Fibers).",
          "Export the segmentation from Dtect again (the combined capillary and fibre export) and load that "
          ".mat file."),
    _item("DT-S02", "fatal", "", False, "Image size missing or invalid",
          "Every position is measured relative to the image size, so nothing can be placed.",
          "Export the segmentation from Dtect again; ImageSize must hold the image height and width in pixels."),
    _item("DT-S03", "fatal", "", False, "Capillary list has the wrong shape",
          "Capillary centres must be a list of x, y pixel positions (N x 2).",
          "Export the segmentation from Dtect again."),
    _item("DT-S04", "fatal", "", False, "Too few capillaries",
          "Capillary domains, supply indices and the oxygen model need at least 3 capillaries.",
          _RESEG + ", checking that the capillaries were marked."),
    _item("DT-S05", "fatal", "", False, "Positions do not match the image",
          "Most capillaries or fibre outlines lie outside the image, so x and y are swapped or the image size "
          "is wrong.",
          "Export the segmentation from Dtect again from the same image; do not combine files from different "
          "images."),
    _item("DT-S06", "fatal", "", False, "No usable fibre outlines",
          "Skeletal-muscle analysis needs fibre outlines.",
          'Choose "Cardiac muscle (capillaries only)", or segment the fibres in Dtect and export again.'),
    _item("DT-S07", "error", "capillaries", True, "Capillaries outside the image",
          "Points outside the image are not tissue; they distort capillary domains and density.",
          _RESEG + ", or let OTM remove these capillaries."),
    _item("DT-S08", "error", "fibres", True, "Fibre outlines outside the image",
          "Outlines reaching outside the image are not tissue; they distort fibre areas and the model.",
          _RESEG + ", or let OTM remove these fibres."),
    _item("DT-S09", "error", "fibres", False, "Broken fibre outlines",
          "These outlines have fewer than 3 points, no area, missing values or x and y lists of different "
          "lengths.",
          "Let OTM remove them, or re-segment these fibres in Dtect."),
    _item("DT-S10", "error", "capillaries", False, "Broken capillary positions",
          "These capillary positions are missing (NaN) or infinite.",
          "Let OTM remove them, or re-mark these capillaries in Dtect."),
    _item("DT-S11", "warning", "", False, "Fibre types do not match the fibres",
          "The number of fibre types differs from the number of fibre outlines, so the types are ignored "
          "(every fibre is type I).",
          "Re-type the fibres in Dtect and export again if fibre types matter for this analysis."),
    _item("DT-S12", "error", "types", False, "Unknown fibre-type codes",
          "Dtect codes are 1 (type I), 21 (IIa), 22 (IIb) and 0 (unknown); the model has no parameters for "
          "other codes.",
          "Re-type these fibres in Dtect, or let OTM mark them as unknown (modelled as type IIb, as for code "
          "0), then set their types in step 2 (Correct data) if you know them."),
    # ---- fibre outlines
    _item("DT-F01", "error", "fibres", False, "Outline drawn around other fibres",
          "An outline that encloses other fibres is not a fibre: usually Dtect traced the edge of the whole "
          "section or of a fascicle. OTM would move every capillary inside it onto that outline, so indices, "
          "domains and PO2 would all be wrong.",
          _RESEG + ", deleting the outline around the section (or fascicle); or let OTM remove it."),
    _item("DT-F02", "error", "fibres", True, "Fibre far larger than the others",
          "A fibre many times the median size is usually several fibres merged into one, or a region of "
          "background traced as a fibre.",
          _RESEG + ", splitting merged fibres; or let OTM remove it, or keep it if it is right."),
    _item("DT-F03", "error", "fibres", True, "Capillaries inside a fibre",
          "Capillaries lie between fibres, so several capillaries inside one outline mean the outline is wrong "
          "(merged fibres or a traced region). OTM would move them onto the outline.",
          _RESEG + ", or let OTM remove this outline, or keep it if it is right."),
    _item("DT-F04", "error", "fibres", False, "Fibre traced twice",
          "Two outlines cover nearly the same fibre, which counts it twice and breaks the model geometry.",
          "Let OTM remove the second copy, or delete it in Dtect and export again."),
    _item("DT-F05", "warning", "", False, "Overlapping fibre outlines",
          'Fibre outlines overlap. Step 3 (Retouch fibres, "Remove fibre-fibre overlaps") makes them disjoint, '
          "but large overlaps change fibre areas.",
          "Check these fibres; correct them in step 2 (Correct data) or in Dtect."),
    _item("DT-F06", "warning", "fibres", False, "Very small fibres",
          "Outlines this small (fibres cut by the image edge are not counted) are often debris, tracing "
          "artefacts or fibre ends.",
          "Check them; remove them in step 2 (Correct data) or in Dtect if they are not fibres."),
    _item("DT-F07", "warning", "", False, "Outline crosses itself",
          "OTM cuts off the small loop when it builds the mesh.",
          "Usually harmless; re-trace these fibres in Dtect if the loop is large."),
    # ---- capillaries
    _item("DT-C01", "warning", "", False, "Many capillaries marked twice",
          "Capillaries closer than the merge distance are merged into one; many such pairs suggest double "
          "marking.",
          "Check the capillary marking in Dtect."),
    _item("DT-C02", "warning", "capillaries", False, "Capillaries far from any fibre",
          "Capillaries lie between fibres; points far from every fibre are usually outside the section or "
          "debris.",
          "Check them; remove them in step 2 (Correct data) or in Dtect if they are not capillaries."),
    _item("DT-C03", "warning", "", False, "Unusual capillary-to-fibre ratio",
          "The ratio of capillaries to fibres is far from usual values, so capillaries or fibres may be "
          "missing.",
          "Check that every capillary and every fibre was segmented in Dtect."),
]}


def catalogue(code: Optional[str] = None):
    """The whole catalogue (list), or one entry by code."""
    if code is None:
        return [dict(v) for v in CATALOGUE.values()]
    if code not in CATALOGUE:
        raise KeyError(f"unknown data-check code: {code}")
    return dict(CATALOGUE[code])


def mode_for(tissue: str = "skeletal", use_fibre_types: bool = True) -> str:
    """The check mode of a tissue choice (the Load page's)."""
    if tissue == "cardiac":
        return "CapillariesOnly"
    return "CapillariesAndFibreTypes" if use_fibre_types else "CapillariesAndFibres"


def _program_version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:                                                # pragma: no cover
        return "dev"


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===========================================================================
# check
# ===========================================================================


def check(d: Any, tissue: str = "skeletal", use_fibre_types: bool = True, mode: Optional[str] = None,
          opts: Optional[Mapping[str, float]] = None, previous: Optional[Mapping[str, Any]] = None,
          action: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Check a :class:`otm_core.io.DtectData`; returns the data-check result (a dict).

    ``previous``/``action``: after :func:`apply_fixes`, pass the earlier result and
    the action it returned, so the new result keeps what was done and which
    problems were removed (for the reports)."""
    o = dict(DEFAULTS)
    o.update(opts or {})
    mode = mode or mode_for(tissue, use_fibre_types)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    img = tuple(getattr(d, "image_size", ()) or ())
    caps = np.asarray(getattr(d, "capillaries_px", np.zeros((0, 2))), float)
    fib = list(getattr(d, "fibers_px", []) or [])
    Q: Dict[str, Any] = dict(format=FORMAT, version=FORMAT_VERSION, program="OTM Lite (Python)",
                             otm_version=_program_version(), created=_now(),
                             file=os.path.basename(getattr(d, "path", "") or ""), mode=mode,
                             image_size=[float(v) for v in img[:2]] if len(img) >= 2 else [],
                             n_capillaries=int(caps.shape[0]) if caps.ndim == 2 else int(caps.size // 2),
                             n_fibres=len(fib), issues=[], status="ok", overridden=False,
                             actions=[], fixed_issues=[], thresholds=o)
    if previous is not None:
        Q["actions"] = [dict(a) for a in previous.get("actions", [])]
        Q["fixed_issues"] = [dict(i) for i in previous.get("fixed_issues", [])]
        if action is not None:
            Q["actions"].append(dict(action))
            done = set(action.get("codes", []))
            Q["fixed_issues"] += [dict(i) for i in previous.get("issues", []) if i["code"] in done]
    _check(Q, d, caps, fib, o, mode)
    return _finish(Q)


def not_dtect(path: str, reason: str = "", tissue: str = "skeletal", use_fibre_types: bool = True) -> Dict[str, Any]:
    """The result for a file that is not a Dtect export at all (DT-S01, fatal)."""
    Q: Dict[str, Any] = dict(format=FORMAT, version=FORMAT_VERSION, program="OTM Lite (Python)",
                             otm_version=_program_version(), created=_now(), file=os.path.basename(path or ""),
                             mode=mode_for(tissue, use_fibre_types), image_size=[], n_capillaries=0, n_fibres=0,
                             issues=[], status="ok", overridden=False, actions=[], fixed_issues=[],
                             thresholds=dict(DEFAULTS))
    msg = "The file has no Xcap (capillary centres) or no ImageSize"
    _add(Q, "DT-S01", msg + (f" ({reason})." if reason and "Xcap" not in reason else "."))
    return _finish(Q)


def _check(Q, d, caps, fib, o, mode):
    # ---- image size
    img = Q["image_size"]
    if len(img) < 2 or not all(np.isfinite(img)) or min(img) <= 0:
        _add(Q, "DT-S02", "ImageSize must hold the image height and width in pixels.")
        return
    H, W = img[0], img[1]
    tol = o["edgeTolPx"]

    # ---- capillaries
    cap_id = _ids(d, "capillary_ids", len(caps) if caps.ndim == 2 else 0)
    if caps.size and (caps.ndim != 2 or caps.shape[1] != 2):
        _add(Q, "DT-S03", f"Xcap is {caps.shape[0]} x {caps.shape[1] if caps.ndim == 2 else 1}; OTM needs N x 2.")
        return
    if caps.size == 0:
        caps = np.zeros((0, 2))
    bad = ~np.all(np.isfinite(caps), axis=1)
    if bad.any():
        idx = np.flatnonzero(bad)
        _add(Q, "DT-S10", f"{_count(len(idx), 'capillary', 'capillaries')} with a missing or infinite position.",
             caps_=idx, cap_ids=cap_id[idx], items=[f"capillary {cap_id[i]}" for i in idx])
    good = np.flatnonzero(~bad)
    if len(good) < o["minCapillaries"]:
        _add(Q, "DT-S04", f"The file has {_count(len(good), 'capillary', 'capillaries')} with a valid position.")
        return
    out_cap = good[_outside(caps[good], W, H, tol)]
    if len(out_cap) >= o["outsideFatal"] * len(good):
        _add(Q, "DT-S05", f"{len(out_cap)} of {len(good)} capillaries lie outside the {W:.0f} x {H:.0f} px image"
                          f"{_swap_hint(caps[good], W, H, tol)}.")
        return
    if len(out_cap):
        _add(Q, "DT-S07", f"{_count(len(out_cap), 'capillary lies', 'capillaries lie')} outside the {W:.0f} x "
                          f"{H:.0f} px image.", caps_=out_cap, cap_ids=cap_id[out_cap],
             items=[f"capillary {cap_id[i]} at ({caps[i, 0]:.0f}, {caps[i, 1]:.0f}) px" for i in out_cap])
    ok_cap = np.setdiff1d(good, out_cap)
    C = caps[ok_cap]

    if mode == "CapillariesOnly":
        _capillary_checks(Q, C, ok_cap, cap_id, [], 0.0, o)
        return

    # ---- fibre outlines: readable?
    nF = len(fib)
    fib_id = _ids(d, "fiber_ids", nF)
    if nF == 0:
        _add(Q, "DT-S06", "The file has no fibre outlines.")
        return
    probs = list(getattr(d, "fiber_read_problems", None) or [""] * nF)
    if len(probs) != nF:
        probs = [""] * nF
    R: List[np.ndarray] = [np.zeros((0, 2))] * nF          # rings in image pixels (x, y)
    A = np.zeros(nF)
    cen = np.full((nF, 2), np.nan)
    broken = np.zeros(nF, bool)
    why = [""] * nF
    for k in range(nF):
        r = np.asarray(fib[k], float)
        if probs[k]:
            broken[k], why[k] = True, probs[k]
            continue
        if r.size == 0 or r.ndim != 2 or r.shape[1] != 2:
            broken[k], why[k] = True, "no points"
            continue
        r = r[:, ::-1]                                       # Fibers.x = row, Fibers.y = column
        if not np.all(np.isfinite(r)):
            broken[k], why[k] = True, "missing or infinite coordinates"
            continue
        if len(r) > 1 and np.all(r[0] == r[-1]):
            r = r[:-1]
        if len(np.unique(r, axis=0)) < 3:
            broken[k], why[k], R[k] = True, "fewer than 3 distinct points", r
            continue
        a, c = _area_centroid(r)
        if a < 0.5:
            broken[k], why[k], R[k] = True, "no area", r
            continue
        R[k], A[k], cen[k] = r, a, c
    if broken.any():
        idx = np.flatnonzero(broken)
        _add(Q, "DT-S09", f"{_count(len(idx), 'fibre outline', 'fibre outlines')} cannot be used.", fibres=idx,
             fib_ids=fib_id[idx], items=[f"fibre {fib_id[i]}: {why[i]}" for i in idx])
    valid = np.flatnonzero(~broken)
    if len(valid) == 0:
        _add(Q, "DT-S06", "None of the fibre outlines can be used.")
        return
    out_f = np.array([k for k in valid if np.any(_outside(R[k], W, H, tol))], int)
    if len(out_f) >= o["outsideFatal"] * len(valid):
        allp = np.vstack([R[k] for k in valid])
        _add(Q, "DT-S05", f"{len(out_f)} of {len(valid)} fibre outlines reach outside the {W:.0f} x {H:.0f} px "
                          f"image{_swap_hint(allp, W, H, tol)}.")
        return
    if len(out_f):
        _add(Q, "DT-S08", f"{_count(len(out_f), 'fibre outline reaches', 'fibre outlines reach')} outside the "
                          f"{W:.0f} x {H:.0f} px image.", fibres=out_f, fib_ids=fib_id[out_f],
             items=[f"fibre {fib_id[i]} at ({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px" for i in out_f])

    # ---- fibre types
    if mode == "CapillariesAndFibreTypes":
        m = getattr(d, "type_count_mismatch", None)
        if m:
            _add(Q, "DT-S11", f"The file has {m[0]} fibre types for {m[1]} fibre outlines.")
        types = getattr(d, "fiber_types", None)
        if types is not None and len(types) == nF:
            types = np.asarray(types)
            odd = np.flatnonzero(~np.isin(types, [0, 1, 21, 22]))
            if len(odd):
                codes = sorted({_num(types[i]) for i in odd})
                _add(Q, "DT-S12", f"{_count(len(odd), 'fibre has', 'fibres have')} a type code OTM does not know "
                                  f"(code{_plural(len(codes))} {', '.join(codes)}).", fibres=odd,
                     fib_ids=fib_id[odd], items=[f"fibre {fib_id[i]}: code {_num(types[i])}" for i in odd])

    # ---- fibre geometry
    medA = float(np.median(A[valid]))
    eqD = 2.0 * np.sqrt(medA / np.pi)
    bb = np.full((nF, 4), np.nan)                              # xmin xmax ymin ymax
    for k in valid:
        bb[k] = [R[k][:, 0].min(), R[k][:, 0].max(), R[k][:, 1].min(), R[k][:, 1].max()]
    flagged = np.zeros(nF, bool)
    Rd = list(R)                                               # simplified to 1 px for the containment tests
    for k in valid:
        Rd[k] = _thin_ring(R[k])

    # DT-F04 duplicates: centres close together and similar areas (the later copy is flagged)
    dup = np.zeros(nF, bool)
    dup_of = np.zeros(nF, int)
    for ii, i in enumerate(valid):
        if dup[i]:
            continue
        for j in valid[ii + 1:]:
            if dup[j]:
                continue
            ratio = A[j] / A[i]
            if ratio < o["duplicateOverlap"] or ratio > 1.0 / o["duplicateOverlap"]:
                continue
            if np.hypot(*(cen[i] - cen[j])) <= 0.2 * np.sqrt(min(A[i], A[j])):
                dup[j], dup_of[j] = True, i
    if dup.any():
        idx = np.flatnonzero(dup)
        _add(Q, "DT-F04", f"{_count(len(idx), 'fibre is', 'fibres are')} traced twice.", fibres=idx,
             fib_ids=fib_id[idx], items=[f"fibre {fib_id[i]} is a copy of fibre {fib_id[dup_of[i]]} at "
                                         f"({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px" for i in idx])
        flagged |= dup
    cand = np.array([k for k in valid if not dup[k]], int)

    # capillaries strictly inside each outline (on the exported positions)
    caps_in = np.zeros(nF, int)
    for k in cand:
        caps_in[k] = len(_inside_deep(C, Rd[k], bb[k], o["insideDepthPx"]))

    # DT-F01 outlines enclosing other (non-tiny, smaller) fibres
    enc = np.zeros(nF, int)
    real = A >= 0.25 * medA
    for k in cand:
        others = cand[(cand != k) & real[cand] & (A[cand] < A[k])]
        b = bb[k]
        sel = others[(cen[others, 0] >= b[0]) & (cen[others, 0] <= b[1]) & (cen[others, 1] >= b[2])
                     & (cen[others, 1] <= b[3])]
        if len(sel):
            enc[k] = int(np.sum(_in_poly(cen[sel], Rd[k])))
    f01 = np.flatnonzero(enc >= o["enclosedFibres"])
    if len(f01):
        if len(f01) == 1:
            k = f01[0]
            msg = (f"Fibre {fib_id[k]} encloses {_count(enc[k], 'other fibre', 'other fibres')} and "
                   f"{_count(caps_in[k], 'capillary', 'capillaries')}"
                   f"{_section_hint(A[k], W * H, enc[k], len(cand))}.")
        else:
            msg = f"{len(f01)} outlines enclose other fibres (fibres {_id_list(fib_id[f01])})."
        _add(Q, "DT-F01", msg, fibres=f01, fib_ids=fib_id[f01], items=[
            f"fibre {fib_id[i]} at ({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px: encloses {enc[i]} fibre(s) and "
            f"{caps_in[i]} capillar{_plural_y(caps_in[i])}; area {A[i] / medA:.0f} x the median fibre, "
            f"{100 * A[i] / (W * H):.0f} % of the image" for i in f01])
        flagged[f01] = True

    # DT-F02 far larger than the rest
    rest = cand[~flagged[cand]]
    big = rest[(A[rest] > o["imageFraction"] * W * H)
               | ((len(valid) >= o["minFibresForSize"]) & (A[rest] > o["sizeRatio"] * medA))]
    if len(big):
        _add(Q, "DT-F02", f"{_count(len(big), 'fibre is', 'fibres are')} more than {o['sizeRatio']:g} x the median "
                          f"fibre area (largest: {A[big].max() / medA:.0f} x).", fibres=big, fib_ids=fib_id[big],
             items=[f"fibre {fib_id[i]} at ({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px: {A[i] / medA:.0f} x the median "
                    f"fibre area, {100 * A[i] / (W * H):.0f} % of the image, {caps_in[i]} "
                    f"capillar{_plural_y(caps_in[i])} inside" for i in big])
        flagged[big] = True

    # DT-F03 several capillaries inside one outline
    rest = cand[~flagged[cand]]
    ci = rest[caps_in[rest] >= o["capsInside"]]
    if len(ci):
        _add(Q, "DT-F03", f"{_count(len(ci), 'fibre has', 'fibres have')} {o['capsInside']:g} or more capillaries "
                          f"inside (most: {caps_in[ci].max()}).", fibres=ci, fib_ids=fib_id[ci],
             items=[f"fibre {fib_id[i]} at ({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px: {caps_in[i]} capillaries inside"
                    for i in ci])
        flagged[ci] = True

    # DT-F06 very small (found first: a speck inside a fibre is not an overlap)
    rest = cand[~flagged[cand]]
    at_edge = ((bb[rest, 0] <= tol + 1) | (bb[rest, 1] >= W - tol - 1) | (bb[rest, 2] <= tol + 1)
               | (bb[rest, 3] >= H - tol - 1))
    tiny = rest[(A[rest] < max(o["tinyRatio"] * medA, o["tinyPx"])) & ~at_edge]

    # DT-F05 overlapping outlines (warning)
    rest = np.setdiff1d(rest, tiny)
    pairs = []
    for ii, i in enumerate(rest):
        for j in rest[ii + 1:]:
            if bb[i, 0] > bb[j, 1] or bb[j, 0] > bb[i, 1] or bb[i, 2] > bb[j, 3] or bb[j, 2] > bb[i, 3]:
                continue
            fi = len(_inside_deep(Rd[i], Rd[j], bb[j], o["insideDepthPx"])) / len(Rd[i])
            fj = len(_inside_deep(Rd[j], Rd[i], bb[i], o["insideDepthPx"])) / len(Rd[j])
            f = max(fi, fj)
            if f >= o["overlapWarn"]:
                pairs.append((i, j, f))
    if pairs:
        idx = np.unique([p[0] for p in pairs] + [p[1] for p in pairs])
        _add(Q, "DT-F05", f"{_count(len(idx), 'outline', 'outlines')} overlap ({len(pairs)} pair{_plural(len(pairs))}).",
             fibres=idx, fib_ids=fib_id[idx], items=[
                 f"fibres {fib_id[i]} and {fib_id[j]}: {100 * f:.0f} % of one outline lies inside the other"
                 for i, j, f in pairs])

    if len(tiny):
        _add(Q, "DT-F06", f"{_count(len(tiny), 'fibre is', 'fibres are')} smaller than {100 * o['tinyRatio']:.0f} % "
                          f"of the median fibre area.", fibres=tiny, fib_ids=fib_id[tiny],
             items=[f"fibre {fib_id[i]} at ({cen[i, 0]:.0f}, {cen[i, 1]:.0f}) px: {A[i]:.0f} px^2 "
                    f"({100 * A[i] / medA:.2g} % of the median)" for i in tiny])

    # DT-F07 self-crossing outlines (warning); pixel-level pinches are ignored
    cross, loop = [], []
    for k in cand:
        if len(Rd[k]) < 4:
            continue
        _, removed = repair_ring(Rd[k], 1e-9)
        if removed >= max(o["loopPx"], o["loopFraction"] * A[k]):
            cross.append(k)
            loop.append(removed)
    if cross:
        cr = np.array(cross, int)
        _add(Q, "DT-F07", f"{_count(len(cr), 'outline crosses', 'outlines cross')} itself.", fibres=cr,
             fib_ids=fib_id[cr], items=[f"fibre {fib_id[k]} at ({cen[k, 0]:.0f}, {cen[k, 1]:.0f}) px: loop of "
                                        f"{la:.0f} px^2" for k, la in zip(cr, loop)])

    # ---- capillaries against the (unflagged) fibres
    good_f = cand[~flagged[cand]]
    _capillary_checks(Q, C, ok_cap, cap_id, [R[k] for k in good_f], eqD, o)


def _capillary_checks(Q, C, ok_cap, cap_id, rings, eqD, o):
    n = len(C)
    if n >= 2:
        nn = _nearest_other(C)
        twin = nn <= o["dedupTolPx"]
        if twin.sum() >= o["duplicateCapFraction"] * n:
            _add(Q, "DT-C01", f"{int(twin.sum())} of {n} capillaries lie within {o['dedupTolPx']:g} px of another one.")
    if not rings:
        return
    V = np.vstack(rings)
    dist = _nearest_distance(C, V)
    far = np.flatnonzero(dist > o["farDiameters"] * eqD)
    if len(far):
        idx = ok_cap[far]
        _add(Q, "DT-C02", f"{_count(len(far), 'capillary lies', 'capillaries lie')} more than "
                          f"{o['farDiameters'] * eqD:.0f} px ({o['farDiameters']:g} median fibre "
                          f"diameter{_plural(o['farDiameters'])}) from every fibre.", caps_=idx, cap_ids=cap_id[idx],
             items=[f"capillary {cap_id[ok_cap[k]]} at ({C[k, 0]:.0f}, {C[k, 1]:.0f}) px: {dist[k]:.0f} px from the "
                    f"nearest fibre" for k in far])
    nf = len(rings)
    if nf >= 5:
        cf = n / nf
        if cf < o["cfLow"] or cf > o["cfHigh"]:
            _add(Q, "DT-C03", f"{n} capillaries for {nf} fibres (ratio {cf:.2f}; usual range {o['cfLow']:g}-"
                              f"{o['cfHigh']:g}).")


# ---------------------------------------------------------------------------
def _add(Q, code, message, fibres=(), caps_=(), fib_ids=(), cap_ids=(), items=()):
    c = CATALOGUE[code]
    Q["issues"].append(dict(code=code, severity=c["severity"], title=c["title"], message=message, why=c["why"],
                            advice=c["advice"], fix=c["fix"], overridable=c["overridable"],
                            fibres=[int(v) for v in fibres], capillaries=[int(v) for v in caps_],
                            fibre_ids=[int(v) for v in fib_ids], capillary_ids=[int(v) for v in cap_ids],
                            items=list(items)))


def _finish(Q):
    s = 0
    for it in Q["issues"]:
        s = max(s, _ORDER.index(it["severity"]))
    Q["status"] = _ORDER[s]
    Q["issues"].sort(key=lambda it: (-_ORDER.index(it["severity"]), it["code"]))
    return Q


def _ids(d, name, n):
    v = getattr(d, name, None)
    v = np.asarray(v if v is not None else [], int).ravel()
    return v if len(v) == n else np.arange(1, n + 1)


def _outside(P, W, H, tol):
    return (P[:, 0] < -tol) | (P[:, 0] > W + tol) | (P[:, 1] < -tol) | (P[:, 1] > H + tol)


def _swap_hint(P, W, H, tol):
    if W != H and np.mean(_outside(P[:, ::-1], W, H, tol)) < 0.05:
        return "; with x and y swapped they would fit, so the coordinates are probably swapped"
    if np.all(P >= -tol):
        return f"; they reach ({P[:, 0].max():.0f}, {P[:, 1].max():.0f}) px, so the image size may be wrong"
    return ""


def _area_centroid(r):
    x, y = r[:, 0], r[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    a2 = cr.sum()
    a = abs(a2) / 2.0
    if abs(a2) < np.finfo(float).eps:
        return a, r.mean(axis=0)
    return a, np.array([np.sum((x + xs) * cr), np.sum((y + ys) * cr)]) / (3.0 * a2)


def _in_poly(P, ring):
    """Points strictly inside a closed ring (crossing number, vectorised)."""
    P = np.atleast_2d(P)
    x, y = P[:, 0:1], P[:, 1:2]
    xa, ya = ring[:, 0][None, :], ring[:, 1][None, :]
    xb, yb = np.roll(xa, -1, axis=1), np.roll(ya, -1, axis=1)
    up = (ya <= y) & (yb > y)
    down = (yb <= y) & (ya > y)
    xc = xa + (y - ya) * (xb - xa) / (yb - ya + (yb == ya))
    return (np.sum((up | down) & (x < xc), axis=1) % 2) == 1


def _dist_to_ring(P, ring):
    a, b = ring, np.roll(ring, -1, axis=0)
    ab = b - a
    L2 = np.maximum(np.sum(ab ** 2, axis=1), np.finfo(float).tiny)[None, :]
    t = ((P[:, 0:1] - a[None, :, 0]) * ab[None, :, 0] + (P[:, 1:2] - a[None, :, 1]) * ab[None, :, 1]) / L2
    t = np.clip(t, 0, 1)
    dx = a[None, :, 0] + t * ab[None, :, 0] - P[:, 0:1]
    dy = a[None, :, 1] + t * ab[None, :, 1] - P[:, 1:2]
    return np.sqrt(np.min(dx ** 2 + dy ** 2, axis=1))


def _inside_deep(P, ring, b, depth):
    """Indices of rows of P strictly inside ring and further than depth from its outline."""
    if len(P) == 0 or len(ring) < 3:
        return np.zeros(0, int)
    sel = np.flatnonzero((P[:, 0] >= b[0]) & (P[:, 0] <= b[1]) & (P[:, 1] >= b[2]) & (P[:, 1] <= b[3]))
    if len(sel) == 0:
        return sel
    sel = sel[_in_poly(P[sel], ring)]
    if len(sel) == 0 or depth <= 0:
        return sel
    return sel[_dist_to_ring(P[sel], ring) > depth]


def _simplify(P, tol):
    """Plain iterative Ramer-Douglas-Peucker."""
    n = len(P)
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        A0, v = P[a], P[b] - P[a]
        Qp = P[a + 1:b] - A0
        L = float(np.hypot(*v))
        d = np.abs(Qp[:, 0] * v[1] - Qp[:, 1] * v[0]) / L if L > 0 else np.hypot(Qp[:, 0], Qp[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return P[keep]


def _thin_ring(r):
    """Douglas-Peucker to 1 px (2 px if still long); corners are kept."""
    if len(r) <= 60:
        return r
    t = r
    for tol in (1, 2):
        t = _simplify(np.vstack([r, r[:1]]), tol)[:-1]
        if len(t) <= 200:
            break
    return t if len(t) >= 3 else r


def _nearest_other(C):
    from scipy.spatial import cKDTree

    if len(C) < 2:
        return np.full(len(C), np.inf)
    d, _ = cKDTree(C).query(C, k=2)
    return d[:, 1]


def _nearest_distance(Qp, P):
    from scipy.spatial import cKDTree

    if len(P) == 0 or len(Qp) == 0:
        return np.full(len(Qp), np.inf)
    d, _ = cKDTree(P).query(Qp, k=1)
    return d


def _section_hint(a, img_area, n_enc, n_all):
    if n_enc >= 0.5 * (n_all - 1) or a >= 0.5 * img_area:
        return f" ({100 * a / img_area:.0f} % of the image): it looks like an outline around the whole section"
    if n_enc >= 3:
        return ": it looks like an outline around a group of fibres"
    return ""


def _count(n, one, many):
    n = int(n)
    return f"1 {one}" if n == 1 else f"{n} {many}"


def _plural(n):
    return "" if n == 1 else "s"


def _plural_y(n):
    return "y" if n == 1 else "ies"


def _num(v):
    v = float(v)
    return f"{v:g}"


def _id_list(ids):
    ids = [str(int(v)) for v in ids]
    return ", ".join(ids[:8]) + (", ..." if len(ids) > 8 else "")


# ===========================================================================
# decisions
# ===========================================================================


def is_blocking(Q: Optional[Mapping[str, Any]]) -> Tuple[bool, bool]:
    """(errors remain unresolved, every remaining error may be kept by the user).
    An empty result (a session saved before the data check existed) never blocks."""
    if not Q or "issues" not in Q:
        return False, True
    err = [i for i in Q["issues"] if i["severity"] in ("error", "fatal")]
    if not err:
        return False, True
    can = all(i["overridable"] for i in err) and not any(i["severity"] == "fatal" for i in err)
    return (not (Q.get("overridden") and can)), can


def override(Q: Dict[str, Any], reason: str = "kept by the user") -> Dict[str, Any]:
    """Keep the flagged items (the user confirmed they are right); recorded in the actions."""
    blocking, can = is_blocking(Q)
    if not blocking:
        return Q
    if not can:
        err = [i for i in Q["issues"] if i["severity"] in ("error", "fatal") and not i["overridable"]]
        raise ValueError("These problems cannot be kept: " + "; ".join(f"{i['code']} {i['title']}" for i in err)
                         + ". Remove the flagged items or re-segment the image in Dtect.")
    Q = copy.deepcopy(Q)
    err = [i for i in Q["issues"] if i["severity"] == "error"]
    Q["overridden"] = True
    Q["actions"].append(dict(time=_now(), action="kept", codes=[i["code"] for i in err],
                             fibre_ids=sum((i["fibre_ids"] for i in err), []),
                             capillary_ids=sum((i["capillary_ids"] for i in err), []),
                             text=f"Data check ({', '.join(i['code'] for i in err)}): flagged items {reason}"))
    return Q


def apply_fixes(d: Any, Q: Mapping[str, Any], which: Any = "errors") -> Tuple[Any, Dict[str, Any]]:
    """Remove (or re-type) the flagged items; returns (new DtectData, action).

    ``which``: "errors" (every fixable error), "all" (fixable errors and
    warnings), a code, or a list of codes. The returned data keep the original
    numbering in ``fiber_ids``/``capillary_ids`` and the removed outlines in
    ``removed_fibres``; check it again with ``check(new, ..., previous=Q, action=act)``."""
    issues = Q.get("issues", [])
    if isinstance(which, str):
        if which == "errors":
            use = [i for i in issues if i["severity"] == "error"]
        elif which == "all":
            use = [i for i in issues if i["severity"] in ("error", "warning")]
        else:
            use = [i for i in issues if i["code"] == which]
    else:
        use = [i for i in issues if i["code"] in set(which)]
    use = [i for i in use if i["fix"]]
    nF = len(d.fibers_px)
    caps = np.asarray(d.capillaries_px, float)
    nC = len(caps)
    fid = _ids(d, "fiber_ids", nF)
    cid = _ids(d, "capillary_ids", nC)
    remF, remC, retype, codes = set(), set(), set(), []
    for it in use:
        if it["fix"] == "fibres":
            remF.update(it["fibres"])
        elif it["fix"] == "capillaries":
            remC.update(it["capillaries"])
        elif it["fix"] == "types":
            retype.update(it["fibres"])
        codes.append(it["code"])
    remF, remC = sorted(remF), sorted(remC)
    retype = sorted(retype - set(remF))
    parts = []
    if remF:
        parts.append(f"removed fibre{_plural(len(remF))} {_ids_text(fid[remF])}")
    if remC:
        parts.append(f"removed capillar{_plural_y(len(remC))} {_ids_text(cid[remC])}")
    types = None if d.fiber_types is None else np.asarray(d.fiber_types).copy()
    if retype:
        parts.append(f"set fibre{_plural(len(retype))} {_ids_text(fid[retype])} to unknown type")
        if types is not None and len(types) == nF:
            types[retype] = 0
    act = dict(time=_now(), action="removed", codes=codes, fibre_ids=[int(v) for v in fid[remF]],
               capillary_ids=[int(v) for v in cid[remC]],
               text=f"Data check ({', '.join(codes)}): {'; '.join(parts) or 'nothing to change'}")
    keepF = [k for k in range(nF) if k not in set(remF)]
    keepC = [k for k in range(nC) if k not in set(remC)]
    removed = list(getattr(d, "removed_fibres", None) or [])
    removed += [(int(fid[k]), np.asarray(d.fibers_px[k], float)) for k in remF]
    probs = getattr(d, "fiber_read_problems", None)
    changes = dict(fibers_px=[d.fibers_px[k] for k in keepF], capillaries_px=caps[keepC] if nC else caps,
                   fiber_types=(types[keepF] if types is not None and len(types) == nF else types),
                   fiber_ids=fid[keepF], capillary_ids=cid[keepC], removed_fibres=removed,
                   notes=list(d.notes) + [act["text"]])
    if probs is not None and len(probs) == nF:
        changes["fiber_read_problems"] = [probs[k] for k in keepF]
    new = dataclasses.replace(d, **changes)
    return new, act


def _ids_text(v):
    v = [str(int(x)) for x in v]
    return ", ".join(v[:12]) + (f" and {len(v) - 12} more" if len(v) > 12 else "")


def flagged_items(Q: Optional[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
    """(fibre indices, capillary indices) flagged by errors and warnings, for drawing."""
    if not Q:
        return [], []
    f, c = set(), set()
    for it in Q.get("issues", []):
        if it["severity"] in ("error", "fatal", "warning"):
            f.update(it["fibres"])
            c.update(it["capillaries"])
    return sorted(f), sorted(c)


# ===========================================================================
# text and reports
# ===========================================================================


def headline(Q: Mapping[str, Any]) -> str:
    n = lambda s: sum(1 for i in Q["issues"] if i["severity"] == s)     # noqa: E731
    fixed = bool(Q.get("fixed_issues"))
    st = Q["status"]
    if st == "ok":
        return ("Data check passed after OTM removed the flagged items." if fixed
                else "Data check passed: no problems found.")
    if st == "warning":
        h = (f"Data check: {n('warning')} warning{_plural(n('warning'))}. The analysis can go on; please look at "
             f"the flagged items.")
        return ("Errors removed. " + h) if fixed else h
    if st == "error":
        if Q.get("overridden"):
            return "Data check: problems found; the flagged items were kept by the user."
        return (f"Data check: {n('error')} problem{_plural(n('error'))} that would make the results wrong. "
                f"Fix them before going on.")
    return "Data check: this file cannot be analysed by OTM."


def _mark(sev):
    return {"fatal": "ERROR  ", "error": "ERROR  ", "warning": "Warning"}.get(sev, "       ")


def text(Q: Optional[Mapping[str, Any]], style: str = "full") -> str:
    """'full' report, 'short' (headline + one line per problem) or 'line' ('DT-F01 x1; ...')."""
    if not Q or "issues" not in Q:
        return ""
    I = Q["issues"]
    if style == "line":
        return "; ".join(f"{i['code']} x{max(1, len(i['items']))}" for i in I)
    if style == "short":
        L = [headline(Q)] + [f"{_mark(i['severity'])}  {i['title']} ({i['code']}): {i['message']}" for i in I]
        e = [i for i in I if i["severity"] in ("fatal", "error")]
        if e and not Q.get("overridden"):
            L.append("What to do: " + e[0]["advice"])
        return "\n".join(L)
    img = Q.get("image_size") or [float("nan")] * 2
    L = [f"{Q.get('program', 'OTM')} {Q.get('otm_version', '')} data check of {Q.get('file', '')} ({Q.get('created', '')})",
         f"Mode: {Q['mode']} | image {img[-1]:.0f} x {img[0]:.0f} px | {Q['n_capillaries']} capillaries | "
         f"{Q['n_fibres']} fibres", headline(Q), ""]
    for it in I:
        L += [f"{it['severity'].upper()} {it['code']}  {it['title']}", "  " + it["message"]]
        for j, s in enumerate(it["items"]):
            if j >= 50:
                L.append(f"    ... and {len(it['items']) - 50} more")
                break
            L.append("    - " + s)
        L += ["  Why it matters: " + it["why"], "  What to do: " + it["advice"], ""]
    if Q.get("fixed_issues"):
        L.append("Problems found and removed by OTM:")
        for it in Q["fixed_issues"]:
            L.append(f"  {it['code']} {it['title']}: {it['message']}")
            L += ["    - " + s for s in it["items"][:50]]
        L.append("")
    if Q.get("actions"):
        L.append("Actions:")
        L += [f"  {a['time']}  {a['text']}" for a in Q["actions"]]
    return "\n".join(L)


def write_report(Q: Mapping[str, Any], stem: str, d: Any = None) -> List[str]:
    """``<stem>.txt`` and ``<stem>.json``. With ``d`` (the data the check ran on),
    the JSON also holds the outlines of the flagged and removed fibres in image
    pixels (x, y). The JSON has the file name but not its folder, and no image."""
    folder = os.path.dirname(os.path.abspath(stem))
    os.makedirs(folder, exist_ok=True)
    with open(stem + ".txt", "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(text(Q))
    J = copy.deepcopy(dict(Q))
    J["environment"] = dict(platform=sys.platform, os=platform.platform(), python=sys.version.split()[0])
    outlines = []
    if d is not None:
        fid = _ids(d, "fiber_ids", len(d.fibers_px))
        f = sorted({k for it in Q["issues"] if it["severity"] in ("error", "fatal") for k in it["fibres"]})
        for k in f:
            if k < len(d.fibers_px):
                r = np.asarray(d.fibers_px[k], float)
                if r.ndim == 2 and r.shape[1] == 2:
                    outlines.append(dict(fibre_id=int(fid[k]), x=np.round(r[:, 1], 2).tolist(),
                                         y=np.round(r[:, 0], 2).tolist()))
        for fid_, r in getattr(d, "removed_fibres", None) or []:
            r = np.asarray(r, float)
            if r.ndim == 2 and r.shape[1] == 2:
                outlines.append(dict(fibre_id=int(fid_), x=np.round(r[:, 1], 2).tolist(),
                                     y=np.round(r[:, 0], 2).tolist()))
    J["flagged_outlines"] = outlines
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(J, fh, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    return [stem + ".txt", stem + ".json"]
