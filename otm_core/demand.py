"""Oxygen demand of the section: the fibre number Λ and what it switches.

For fibre ``j`` (definition by A. A. Al-Shammari, 25 Oct 2026)::

    Λ_j = M0_j · A_j / (K_j · Pcap),     K_j = α_j D_j + C_Mb,j D_Mb / P50,Mb

* ``M0_j``  maximal O2 consumption of the fibre (ml O2 / ml / s): the exercise
  level times M0, shared between the fibre types exactly as the PO2 solver
  shares it (uptake ratios of Popel et al. 2012, I 2.08 m and IIa 1.76 m times IIb
  with m the differential O2 extraction (1, 5 or 10; legacy
  GetCompartmentIndicesAndModelParameters.m), normalised so that the
  mean over the whole section (interstitial space included) is exercise level × M0);
* ``A_j``   fibre area (cm²), from the retouched outline the model uses;
* ``α_j D_j`` O2 solubility × diffusivity of the fibre type, as the solver's
  relative coefficients (non-uniform properties on or off);
* ``C_Mb,j D_Mb / P50,Mb`` the myoglobin-facilitated part (Mb concentration of
  the fibre type: I 1, IIa 20/41, IIb 6.24/41 of c_Mb, as the solver). It is
  always included, whether or not the myoglobin term is switched on in the PDE,
  so that Λ measures demand against the fibre's full transport capacity and
  scales exactly with exercise level × M0;
* ``Pcap``  capillary PO2 (mmHg).

Λ is dimensionless. It compares the O2 a fibre consumes with the O2 diffusion
can bring across a distance of the order of its size from capillaries at Pcap
(a slab of half-width δ has an anoxic core when M δ² / (2 K Pcap) > 1, i.e. the
penetration depth is δ = √(2 K Pcap / M)).

Per fibre, Λ spreads over about a factor ten with fibre size alone, so the
regime is decided on the section: ``Λ_tissue`` (area-weighted mean over the
fibres) compared with ``Λ_rest``, the same section with the tissue's resting
default parameters, exercise level 1 and differential extraction 1. Their ratio is the demand relative to
rest, independent of the fibre sizes (which differ between species and muscles)::

    ratio = Λ_tissue / Λ_rest         non-resting when ratio > cutoff (default 1.5)

The default cut-off lies halfway between rest (1) and the lowest exercise level
(2 for skeletal muscle). What it does (:class:`DemandSettings`):

* **Uptake model.** Some exercise levels solve without Michaelis–Menten uptake
  and myoglobin facilitation (skeletal Resting; cardiac Resting and Low; any
  custom choice). If the section is non-resting at such a level (a raised M0,
  a lower Pcap, cardiac Low ...), both are switched on for the solution
  (``switch_uptake``) and :attr:`DemandResult.warning` says so: the program
  shows it to the user (user's condition, 25 Oct 2026). The capillary
  permeability is not changed. With constant uptake the PO2 deficit simply
  scales with M0, so the flux-line paths would not respond to demand at all.
* **Flux lines.** Λ does not change the flux-line length cap: it stays the
  resting one (1.25 × the mean capillary spacing), as the user decided on
  25 Oct 2026. Demand shortens the lines through the PO2 field itself (the
  minima move towards the capillaries once Michaelis–Menten uptake is on).
  Lines always end: in arc length every step adds a fixed length up to the
  cap, and a line stops where |grad u| < 1e-3 × the largest gradient, which
  is where it runs into an anoxic floor (High exercise with M0 × 6 on the
  reference section: 16 % of the nodes below 1 mmHg, all 368 lines ended, the
  stalled ones at a median 0.11 mmHg).

Cardiac muscle (capillaries only): the "fibre" is a capillary domain with the
cardiac coefficients. ``A`` is the larger of the mean and the median area of
the Voronoi capillary domains that lie wholly inside the section (domains cut
by the section border are left out: unclipped they reach far outside the
tissue, clipped they only return tissue area / number of capillaries). With
fewer than 3 such domains ``A`` = tissue area / number of capillaries.

``DemandSettings(enabled=False)`` is the legacy behaviour (nothing switched,
cap unchanged); Λ is still reported.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["DemandSettings", "DemandResult", "assess", "fibre_type_factors"]

FIBRE_TYPES = (1, 21, 22)
TYPE_NAMES = {1: "I", 21: "IIa", 22: "IIb"}
MB_RELATIVE = {1: 1.0, 21: 20 / 41, 22: 6.24 / 41}        # Mb concentration relative to type I (solver)


@dataclass
class DemandSettings:
    """What the demand measure Λ is allowed to change (see the module text)."""

    enabled: bool = True             # False: legacy, Λ only reported
    cutoff: float = 1.5              # non-resting when Λ_tissue / Λ_rest > cutoff
    switch_uptake: bool = True       # non-resting at a level without MM/Mb: both on (with a warning)

    def __post_init__(self):
        if not (isinstance(self.cutoff, (int, float)) and self.cutoff > 0):
            raise ValueError("the demand cut-off must be a positive number (ratio to rest)")

    @classmethod
    def legacy(cls) -> "DemandSettings":
        return cls(enabled=False)


@dataclass
class DemandResult:
    lambda_fibre: np.ndarray             # Λ_j per fibre (empty for cardiac tissue)
    fibre_area_um2: np.ndarray
    fibre_type: np.ndarray               # 1, 21, 22 (unknown counted as IIb, as the solver)
    lambda_tissue: float                 # area-weighted mean Λ (cardiac: Λ of the capillary domain A)
    lambda_rest: float                   # the same section at the resting defaults
    ratio: float                         # lambda_tissue / lambda_rest
    cutoff: float
    regime: str                          # "resting" | "non-resting"
    switches: Any                        # model switches to solve with
    uptake_switched: bool                # MM and/or Mb switched on because of Λ (see ``warning``)
    settings: DemandSettings = field(default_factory=DemandSettings)
    domain: Optional[Dict[str, Any]] = None   # cardiac: the capillary-domain area A and how it was chosen
    switched_on: tuple = ()              # what Λ switched on: "Michaelis–Menten uptake", "myoglobin facilitation"
    exercise_level: float = 1.0          # the chosen level's demand factor
    tissue: str = "skeletal"

    @property
    def warning(self) -> str:
        """The warning to show when Λ changed the chosen uptake model ("" otherwise)."""
        if not self.uptake_switched:
            return ""
        lvl = "Resting" if self.exercise_level == 1 else f"× {self.exercise_level:g}"
        what = " and ".join(self.switched_on) or "Michaelis–Menten uptake and myoglobin facilitation"
        return (f"Warning: the oxygen demand is {self.ratio:.2f} × the resting value (non-resting, cut-off "
                f"{self.cutoff:g}), but the chosen {self.tissue} exercise level ({lvl}) solves without {what}. "
                f"Λ has switched {'them' if len(self.switched_on) != 1 else 'it'} on for this solution. To keep "
                f"the level's own model, untick “Let Λ adjust the model” (legacy) or the uptake switch.")

    def by_type(self) -> Dict[str, float]:
        """Area-weighted mean Λ per fibre type."""
        out = {}
        for t in FIBRE_TYPES:
            m = self.fibre_type == t
            if m.any():
                a = self.fibre_area_um2[m]
                out[TYPE_NAMES[t]] = float((self.lambda_fibre[m] * a).sum() / a.sum())
        return out

    def summary(self) -> Dict[str, Any]:
        return {"lambda_tissue": self.lambda_tissue, "lambda_rest": self.lambda_rest, "lambda_ratio": self.ratio,
                "lambda_cutoff": self.cutoff, "demand_regime": self.regime,
                "uptake_switched_by_lambda": self.uptake_switched,
                **({"domain_area_um2": self.domain["area_um2"], "domain_statistic": self.domain["statistic"]}
                   if self.domain else {}),
                **({f"lambda_{k}": v for k, v in self.by_type().items()})}

    def text(self) -> str:
        """One or two lines for the program and the reports."""
        where = (f"capillary domain A = {self.domain['area_um2']:.0f} µm², the {self.domain['statistic']} of "
                 f"{self.domain['n_domains']} domains" if self.domain else "section, area-weighted mean over the fibres")
        t = (f"Λ = {self.lambda_tissue:.3g} ({where}; rest {self.lambda_rest:.3g}): "
             f"{self.ratio:.2f} × the resting demand, {self.regime}")
        if not self.settings.enabled:
            return t + " (demand switches off: legacy behaviour)."
        notes = []
        if self.uptake_switched and len(self.switched_on) == 1:
            notes.append(f"{self.switched_on[0]} switched on")
        elif self.uptake_switched:
            notes.append("Michaelis–Menten uptake and myoglobin facilitation switched on")
        return t + (": " + "; ".join(notes) if notes else "") + "."

    def table(self) -> List[Dict[str, Any]]:
        """Per fibre: number, type, area, Λ (the order of the exported fibre tables)."""
        return [{"Fiber No.": k + 1, "Fiber type": TYPE_NAMES.get(int(t), "?"), "Area (um2)": float(a),
                 "Lambda": float(v)} for k, (t, a, v) in
                enumerate(zip(self.fibre_type, self.fibre_area_um2, self.lambda_fibre))]


def fibre_type_factors(switches: Any) -> Dict[str, Dict[int, float]]:
    """Per fibre type, relative to type I, as ``fem.solve.compartment_coefficients``:
    ``uptake`` (M0 ratio), ``K`` (α D ratio), ``Mb`` (Mb concentration ratio)."""
    # differential O2 extraction m (1 | 5 | 10) multiplies the type I and IIa uptake
    # relative to IIb, as the legacy GetCompartmentIndicesAndModelParameters.m
    de = float(getattr(switches, "differential_extraction", 1.0) or 1.0)
    if getattr(switches, "non_uniform", True):            # Popel et al. 2012, relative to IIb
        up = {1: de * 2.08, 21: de * 1.76, 22: 1.0}
        aD = {1: 1.0 * 2.96522, 21: 1.0 * 1.70435, 22: 1.0}
    else:
        up = {1: de * 2.0, 21: de * 1.76, 22: 1.0}
        aD = {1: 1.0, 21: 1.0, 22: 1.0}
    return {"uptake": {t: up[t] / up[1] for t in FIBRE_TYPES}, "K": {t: aD[t] / aD[1] for t in FIBRE_TYPES},
            "Mb": dict(MB_RELATIVE)}


def _area_ndim(f: Any) -> float:
    """Area of a model fibre: (N, 2) ring, list of rings, or Shapely geometry (ndim²)."""
    if hasattr(f, "area") and hasattr(f, "geom_type"):
        return float(f.area)
    if isinstance(f, (list, tuple)) and f and np.ndim(f[0]) == 2:
        return float(sum(_area_ndim(r) for r in f))
    r = np.asarray(f, float).reshape(-1, 2)
    if len(r) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1))))


def _lambda(params: Any, switches: Any, tissue: str, areas_cm2: np.ndarray, types: np.ndarray,
            tissue_area_cm2: float) -> np.ndarray:
    ex = float(switches.exercise_level)
    p = params
    if tissue == "cardiac":
        K = p.alpha * p.D + p.c_Mb * p.D_Mb / p.P50_Mb
        return np.asarray([ex * p.M0 * areas_cm2[0] / (K * p.Pcap)])
    fac = fibre_type_factors(switches)
    share = {t: areas_cm2[types == t].sum() / tissue_area_cm2 for t in FIBRE_TYPES}
    norm = sum(share[t] * fac["uptake"][t] for t in FIBRE_TYPES)
    # as the solver: type-I uptake ex·M0 / norm, so that the mean over the whole section is ex·M0
    scale = 1.0 / norm if norm > 0 else 1.0
    out = np.empty(len(areas_cm2))
    for k, (a, t) in enumerate(zip(areas_cm2, types)):
        M = ex * p.M0 * fac["uptake"][t] * scale
        K = p.alpha * p.D * fac["K"][t] + p.c_Mb * fac["Mb"][t] * p.D_Mb / p.P50_Mb
        out[k] = M * a / (K * p.Pcap)
    return out


def assess(geometry: Any, params: Any, switches: Any, tissue: str = "skeletal",
           settings: Optional[DemandSettings] = None) -> DemandResult:
    """Λ of the section and the switches to solve with.

    ``geometry``: :class:`otm_core.models.Geometry` with a length scale;
    ``params``: :class:`otm_core.fem.TransportParameters` (dimensional);
    ``switches``: :class:`otm_core.fem.ModelSwitches` of the chosen exercise level."""
    from .fem.solve import TransportParameters

    s = settings or DemandSettings()
    ls = geometry.length_scale
    if ls is None:
        raise ValueError("the tissue size is needed for Λ")
    um_per_ndim = ls.y_um / 2.0
    tissue_cm2 = ls.x_um * ls.y_um * 1e-8
    domain = None
    if tissue == "cardiac" or not len(geometry.fibers):
        domain = capillary_domain_area(geometry.capillaries, ls)
        areas = np.array([domain["area_um2"] * 1e-8])
        types = np.zeros(0, int)
        fib_um2 = np.zeros(0)
        kind = "cardiac"
    else:
        fib_um2 = np.array([_area_ndim(f) for f in geometry.fibers]) * um_per_ndim ** 2
        areas = fib_um2 * 1e-8
        types = np.asarray(geometry.fiber_types, int).ravel()
        if types.size != len(areas):
            types = np.ones(len(areas), int)
        types = np.where(np.isin(types, FIBRE_TYPES), types, 22)       # unknown -> IIb, as the solver
        kind = "skeletal"
    lam = _lambda(params, switches, kind, areas, types, tissue_cm2)
    rest_params = TransportParameters.defaults("cardiac" if tissue == "cardiac" else "skeletal")
    # rest: the tissue's default parameters, exercise 1 and differential extraction 1 (basal
    # recruitment); fibre-specific properties (non-uniform) as chosen
    rest_sw = dataclasses.replace(switches, exercise_level=1.0, michaelis_menten=False, myoglobin=False,
                                  differential_extraction=1.0)
    lam_rest = _lambda(rest_params, rest_sw, kind, areas, types, tissue_cm2)
    if kind == "cardiac":
        lt, lr = float(lam[0]), float(lam_rest[0])
        lam_f = np.zeros(0)
    else:
        w = areas / areas.sum() if areas.sum() > 0 else np.full(len(areas), 1 / max(1, len(areas)))
        lt, lr = float((lam * w).sum()), float((lam_rest * w).sum())
        lam_f = lam
    ratio = lt / lr if lr > 0 else math.inf
    regime = "non-resting" if ratio > s.cutoff else "resting"
    sw = switches
    switched = False
    on: tuple = ()
    if s.enabled and s.switch_uptake and regime == "non-resting":
        on = tuple(name for name, flag in (("Michaelis–Menten uptake", switches.michaelis_menten),
                                           ("myoglobin facilitation", switches.myoglobin)) if not flag)
        if on:
            sw = dataclasses.replace(switches, michaelis_menten=True, myoglobin=True)
            switched = True
    return DemandResult(lambda_fibre=lam_f, fibre_area_um2=fib_um2, fibre_type=types, lambda_tissue=lt,
                        lambda_rest=lr, ratio=ratio, cutoff=float(s.cutoff), regime=regime, switches=sw,
                        uptake_switched=switched, settings=s, domain=domain, switched_on=on,
                        exercise_level=float(switches.exercise_level), tissue=kind)


def capillary_domain_area(capillaries: Any, ls: Any) -> Dict[str, Any]:
    """Cardiac ``A`` (µm²): the larger of the mean and the median area of the Voronoi
    capillary domains lying wholly inside the section (ndim box ±Lx/Ly × ±1).

    Domains cut by the section border are left out. Fewer than 3 inside: tissue
    area / number of capillaries (``statistic`` "tissue area / N")."""
    caps = np.asarray(capillaries, float).reshape(-1, 2)
    n = len(caps)
    whole = ls.x_um * ls.y_um / max(1, n)
    out = {"area_um2": whole, "statistic": "tissue area / N", "n_domains": 0, "n_capillaries": n,
           "mean_um2": math.nan, "median_um2": math.nan, "tissue_over_n_um2": whole}
    if n < 3:
        return out
    from .morphometry import voronoi_cells
    import shapely

    try:
        vor = voronoi_cells(np.unique(caps, axis=0))
    except (ValueError, RuntimeError):
        return out
    hx, eps = ls.x_um / ls.y_um, 1e-12
    inside = []
    for cell, b in zip(vor.cells, vor.bounded):
        xy = shapely.get_coordinates(cell)
        inside.append(bool(b) and len(xy) > 0 and bool(np.all((np.abs(xy[:, 0]) <= hx + eps)
                                                               & (np.abs(xy[:, 1]) <= 1 + eps))))
    inside = np.asarray(inside, bool)
    a = vor.areas_ndim[inside] * (ls.y_um / 2.0) ** 2
    if len(a) < 3:
        return out
    mean, median = float(a.mean()), float(np.median(a))
    out.update(n_domains=int(len(a)), mean_um2=mean, median_um2=median,
               area_um2=max(mean, median), statistic="mean" if mean >= median else "median")
    return out


__all__.append("capillary_domain_area")


def write_report(result: DemandResult, stem: str) -> List[str]:
    """``<stem>_Lambda.txt`` (the section) and ``<stem>_Lambda.csv`` (per fibre)."""
    import csv

    files = []
    lines = ["OTM: oxygen demand of the section (Λ)", "",
             "Λ_j = M0_j · A_j / (K_j · Pcap),  K_j = α_j D_j + C_Mb,j D_Mb / P50,Mb", "",
             result.text(), "",
             *([result.warning, ""] if result.warning else []),
             f"Λ section ({'capillary domain A' if result.domain else 'area-weighted mean over the fibres'}): "
             f"{result.lambda_tissue:.6g}",
             f"Λ at rest (tissue defaults, exercise level 1):   {result.lambda_rest:.6g}",
             f"Ratio to rest: {result.ratio:.4f}  (cut-off {result.cutoff:g}: {result.regime})",
             f"Michaelis–Menten and myoglobin switched on by Λ: {'yes' if result.uptake_switched else 'no'}",
             f"Demand switches: {'on' if result.settings.enabled else 'off (legacy)'}"
             f"; uptake switch {'on' if result.settings.switch_uptake else 'off'}"]
    if result.domain:
        dm = result.domain
        lines += ["", "Cardiac capillary domain (Voronoi domains wholly inside the section):",
                  f"  domains used: {dm['n_domains']} of {dm['n_capillaries']} capillaries",
                  f"  mean {dm['mean_um2']:.6g} µm², median {dm['median_um2']:.6g} µm²",
                  f"  A = {dm['area_um2']:.6g} µm² ({dm['statistic']}); tissue area / N = {dm['tissue_over_n_um2']:.6g} µm²"]
    bt = result.by_type()
    if bt:
        lines += ["", "Area-weighted mean Λ per fibre type:"] + [f"  {k}: {v:.6g}" for k, v in bt.items()]
    p = stem + "_Lambda.txt"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    files.append(p)
    rows = result.table()
    if rows:
        p = stem + "_Lambda.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        files.append(p)
    return files


__all__.append("write_report")
