"""Steady-state tissue PO2: P1 finite elements (scikit-fem) + Newton-Raphson.

Replaces the legacy PDE-Toolbox solve chain::

    O2TransportParameters                    -> TransportParameters.derive
    GetCompartmentIndicesAndModelParameters  -> compartment_coefficients
    SkeletalMuscleSimulation / CardiacMuscleSimulation
        (string coefficients, bcMatrix, assempde / pdenonlin)
                                             -> solve_po2
    OxygenProfileStatistics, OxygenUptakeProfileStatistics,
    SingleFiberPO2Statistics                 -> po2_statistics, mo2_statistics,
                                                fiber_po2_statistics

Model (non-dimensional, u = PO2 / Pcap, lengths in units of Ly/2)
-----------------------------------------------------------------
In every compartment X in {IS, I, IIa, IIb}::

    -div( c_X(u) grad u ) + mu_X g(u) = 0

    c_X(u) = K_X + Mb_X * beta' / (p50 + u)^2      (myoglobin-facilitated diffusion)
    g(u)   = 1                (zero-order uptake)   or   u / (u + pc)  (Michaelis-Menten)

with a Robin condition on every capillary wall (outward normal n of the
tissue) and no flux through the outer box::

    c_X du/dn = kappa (1 - u)      on capillary walls
    du/dn     = 0                  on the box

K_X is the relative solubility x diffusivity (type IIb = 1 before scaling by
type I), Mb_X the relative myoglobin content (I 1, IIa 20/41, IIb 6.24/41,
IS 0), mu_X the relative O2 demand (IS 0), all exactly as in the legacy
``GetCompartmentIndicesAndModelParameters`` / ``SkeletalMuscleSimulation``.
The legacy code writes c and f as PDE-Toolbox *strings* with ``%1.4f``
(four decimals) and the Robin coefficient with ``num2str`` (5-6 significant
digits); ``legacy_rounding=True`` (default) reproduces those rounded values so
results can be compared with MATLAB, ``False`` uses full precision.

Discretisation: continuous P1 elements; volume terms are integrated with the
one-point (centroid) rule and wall terms with the two-point Gauss rule. For
P1 this is exactly what PDE Toolbox's ``assempde``/``pdenonlin`` assemble
(coefficients evaluated at triangle centroids, consistent edge mass matrix),
so on the same mesh the two give the same discrete solution.

Newton-Raphson: the full Jacobian includes the derivative of the facilitated
diffusivity (dc/du * du grad u . grad v) and of the Michaelis-Menten uptake
(mu g'(u) du v). Steps are damped by backtracking on the residual norm (like
``pdenonlin``). The linear case converges in one step.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..progress import ProgressCallback, report
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from ..models import FIBER_TYPE_I, FIBER_TYPE_IIA, FIBER_TYPE_IIB, LengthScale
from .mesh import COMPARTMENT_IS, IS_REGION, TissueMesh

COMPARTMENTS = (COMPARTMENT_IS, FIBER_TYPE_I, FIBER_TYPE_IIA, FIBER_TYPE_IIB)
COMPARTMENT_NAMES = {COMPARTMENT_IS: "IS", FIBER_TYPE_I: "I", FIBER_TYPE_IIA: "IIa", FIBER_TYPE_IIB: "IIb"}


# --------------------------------------------------------------------------
# MATLAB number formatting (the legacy solver parses its own printed strings)
# --------------------------------------------------------------------------


def _round_half_up(x: float, quantum: Decimal) -> float:
    # MATLAB rounds the shortest decimal representation half-up
    # (num2str(1.70435) = '1.7044', where C printf gives '1.7043')
    return float(Decimal(repr(float(x))).quantize(quantum, rounding=ROUND_HALF_UP))


def matlab_num2str(x: float) -> float:
    """Value of ``str2num(num2str(x))``: ``num2str`` prints non-integers with
    ``floor(log10(|x|)) + 5`` significant digits."""
    x = float(x)
    if x == 0 or not math.isfinite(x) or x == round(x):
        return x
    digits = max(int(math.floor(math.log10(abs(x)))) + 5, 1)
    exp = int(math.floor(math.log10(abs(x)))) - digits + 1
    return _round_half_up(x, Decimal(1).scaleb(exp))


def matlab_f4(x: float) -> float:
    """Value of ``sprintf('%1.4f', x)`` read back."""
    return _round_half_up(x, Decimal("0.0001"))


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass
class TransportParameters:
    """The 10 biophysical parameters of ``*RestingParameters.dat`` (dimensional)."""

    r_um: float = 1.8              # capillary radius (um)
    Pcap: float = 30.0             # capillary PO2 (mmHg)
    P50_Mb: float = 5.3            # PO2 at half Mb saturation (mmHg)
    P_c: float = 0.5               # PO2 at half-maximal consumption (mmHg)
    alpha: float = 3.89e-5         # O2 solubility (ml O2 / ml / mmHg)
    D: float = 2.41e-5             # O2 diffusivity (cm^2/s)
    D_Mb: float = 1.73e-7          # Mb diffusivity (cm^2/s)
    M0: float = 1.57e-4            # maximal consumption (ml O2 / ml / s)
    c_Mb: float = 0.0102           # Mb concentration (ml/ml)
    k: float = 4e-6                # capillary mass-transfer coefficient (ml O2 / s / cm^2 / mmHg)

    #: legacy ``mfiles/parameters/*RestingParameters.dat``
    DEFAULTS = {
        "skeletal": (1.8, 30.0, 5.3, 0.5, 3.89e-5, 2.41e-5, 1.73e-7, 1.57e-4, 0.0102, 4e-6),
        "cardiac": (2.0, 40.0, 2.39, 0.5, 4.42e-5, 1.45e-5, 2.2e-7, 1.57e-4, 0.002878, 8.6e-4),
    }
    #: (name, symbol, unit) of each field, in file order (BiophysicalParameters labels)
    DESCRIPTIONS = (
        ("Capillary radius", "R_cap", "µm"),
        ("Capillary PO2", "P_cap", "mmHg"),
        ("Mb half-saturation PO2", "P50,Mb", "mmHg"),
        ("Half-maximal consumption PO2", "P_c", "mmHg"),
        ("O2 solubility", "α", "ml O2 / ml / mmHg"),
        ("O2 diffusivity", "D", "cm²/s"),
        ("Mb diffusivity", "D_Mb", "cm²/s"),
        ("Maximal O2 consumption", "M0", "ml O2 / ml / s"),
        ("Mb concentration", "c_Mb", "ml O2 / ml"),
        ("Capillary wall permeability", "k", "ml O2 / s / cm² / mmHg"),
    )

    @classmethod
    def defaults(cls, tissue: str = "skeletal") -> "TransportParameters":
        """The legacy resting defaults for "skeletal" or "cardiac" tissue."""
        return cls(*cls.DEFAULTS[tissue])

    @classmethod
    def from_values(cls, v: Sequence[float]) -> "TransportParameters":
        v = [float(x) for x in np.ravel(v)]
        if len(v) != 10:
            raise ValueError(f"expected 10 parameters, got {len(v)}")
        return cls(*v)

    @classmethod
    def from_dat(cls, path: str) -> "TransportParameters":
        """Read a legacy ``.dat`` file (one value per line or comma-separated)."""
        with open(path, "r", encoding="utf-8") as fh:
            vals = [float(t) for t in fh.read().replace(",", " ").split()]
        return cls.from_values(vals)

    def derive(self, n_capillaries: int, ls: LengthScale) -> "DerivedParameters":
        """O2TransportParameters: non-dimensional groups (length scale L = Ly/2)."""
        NdimL = ls.y_um / 2.0
        L = NdimL / 1e4                                   # cm
        return DerivedParameters(
            raw=self, Nc=int(n_capillaries), rho=1e6 * n_capillaries / (ls.y_um * ls.x_um), NdimL=NdimL,
            beta=self.D_Mb * self.c_Mb / (self.D * self.alpha * self.P50_Mb),
            p50_Mb=self.P50_Mb / self.Pcap, p_c=self.P_c / self.Pcap,
            mu=L ** 2 * self.M0 / (self.D * self.alpha * self.Pcap),
            kappa=L * self.k / (self.D * self.alpha), Rcap=self.r_um / NdimL)


@dataclass
class DerivedParameters:
    raw: TransportParameters
    Nc: int
    rho: float          # capillary density (1/mm^2)
    NdimL: float        # length scale (um)
    beta: float         # Mb facilitation number
    p50_Mb: float       # P50_Mb / Pcap
    p_c: float          # P_c / Pcap
    mu: float           # consumption number
    kappa: float        # capillary wall permeability number
    Rcap: float         # capillary radius (ndim)


EXERCISE_PERMEABILITY_FACTOR = 8.6
"""Capillary permeability k is multiplied by this times the exercise level when exercising."""


@dataclass
class ModelSwitches:
    """Functional heterogeneity (SkeletalFunctionalHeterogeneity / Cardiac...)."""

    michaelis_menten: bool = False
    myoglobin: bool = False
    exercise_level: float = 1.0
    non_uniform: bool = True           # fibre-specific solubility/diffusivity (Popel et al.)
    differential_extraction: float = 1.0

    # GUI presets
    @classmethod
    def skeletal(cls, level: str = "resting", non_uniform: bool = True,
                 differential_extraction: float = 1.0) -> "ModelSwitches":
        table = {"resting": (False, False, 1), "low": (True, True, 2),
                 "moderate": (True, True, 4), "high": (True, True, 6)}
        mm, mb, ex = table[level.lower()]
        return cls(mm, mb, ex, non_uniform, differential_extraction)

    @classmethod
    def cardiac(cls, level: str = "resting") -> "ModelSwitches":
        table = {"resting": (False, False, 1), "low": (False, False, 5),
                 "moderate": (True, True, 10), "high": (True, True, 20)}
        mm, mb, ex = table[level.lower()]
        return cls(mm, mb, ex, False, 1.0)

    @property
    def permeability_factor(self) -> float:
        """Factor applied to the capillary permeability k (1 at rest)."""
        ex = float(self.exercise_level)
        return 1.0 if ex == 1 else EXERCISE_PERMEABILITY_FACTOR * ex


@dataclass
class CompartmentCoefficients:
    """Coefficients actually used in the PDE (after legacy rounding)."""

    K: Dict[int, float]                # relative solubility x diffusivity
    Mb: Dict[int, float]               # Mb factor times beta' (0 when myoglobin is off)
    mu: Dict[int, float]               # O2 demand (0 in IS)
    p50: float
    pc: float
    kappa: float
    michaelis_menten: bool
    myoglobin: bool
    volume_fraction: Dict[int, float]  # area fraction of each compartment (of the meshed tissue)
    uptake_ratio: Dict[int, float]     # mu_X / mu_I
    vol_avg_uptake_conversion: float   # 1 / (fI + fIIa r_IIa + fIIb r_IIb)
    scales: Dict[str, float] = field(default_factory=dict)


def compartment_coefficients(mesh: TissueMesh, params: DerivedParameters, switches: ModelSwitches,
                             tissue: str = "skeletal", legacy_rounding: bool = True) -> CompartmentCoefficients:
    """GetCompartmentIndicesAndModelParameters + the coefficient strings of
    SkeletalMuscleSimulation (or CardiacMuscleSimulation for ``tissue='cardiac'``)."""
    r4 = matlab_f4 if legacy_rounding else float
    rn = matlab_num2str if legacy_rounding else float
    ex = float(switches.exercise_level)
    kappa = params.kappa if ex == 1 else EXERCISE_PERMEABILITY_FACTOR * ex * params.kappa
    pMb = params.p50_Mb
    beta = params.beta * pMb ** 2
    comp = mesh.cell_compartment()
    area = mesh.cell_areas()
    total = area.sum()
    frac = {c: float(area[comp == c].sum() / total) for c in COMPARTMENTS}

    if tissue == "cardiac":
        mu = ex * params.mu
        K = {c: 1.0 for c in COMPARTMENTS}
        Mb = {c: (r4(beta) if switches.myoglobin else 0.0) for c in COMPARTMENTS}
        return CompartmentCoefficients(
            K=K, Mb=Mb, mu={c: r4(mu) for c in COMPARTMENTS}, p50=r4(pMb), pc=r4(params.p_c),
            kappa=rn(kappa), michaelis_menten=switches.michaelis_menten, myoglobin=switches.myoglobin,
            volume_fraction=frac, uptake_ratio={c: 1.0 for c in COMPARTMENTS},
            vol_avg_uptake_conversion=1.0)

    de = float(switches.differential_extraction)
    if switches.non_uniform:                             # Popel et al. 2012, relative to IIb
        sc = dict(up_I=de * 2.08, up_IIa=de * 1.76, sol_IS=0.7198, sol_I=1.0, sol_IIa=1.0,
                  dif_IS=2.08696, dif_I=2.96522, dif_IIa=1.70435)
    else:
        sc = dict(up_I=de * 2.0, up_IIa=de * 1.76, sol_IS=1.0, sol_I=1.0, sol_IIa=1.0,
                  dif_IS=1.0, dif_I=1.0, dif_IIa=1.0)
    sc = {k: rn(v) for k, v in sc.items()}               # legacy round-trip through num2str/str2num
    mu_IIa, mu_IIb = sc["up_IIa"] / sc["up_I"], 1.0 / sc["up_I"]
    a_IIa, a_IIb, a_IS = sc["sol_IIa"] / sc["sol_I"], 1.0 / sc["sol_I"], sc["sol_IS"] / sc["sol_I"]
    D_IIa, D_IIb, D_IS = sc["dif_IIa"] / sc["dif_I"], 1.0 / sc["dif_I"], sc["dif_IS"] / sc["dif_I"]
    norm = frac[FIBER_TYPE_I] + frac[FIBER_TYPE_IIA] * mu_IIa + frac[FIBER_TYPE_IIB] * mu_IIb
    mu = ex * params.mu / norm
    K = {COMPARTMENT_IS: a_IS * D_IS, FIBER_TYPE_I: 1.0, FIBER_TYPE_IIA: a_IIa * D_IIa,
         FIBER_TYPE_IIB: a_IIb * D_IIb}
    mb_rel = {COMPARTMENT_IS: 0.0, FIBER_TYPE_I: 1.0, FIBER_TYPE_IIA: 20 / 41, FIBER_TYPE_IIB: 6.24 / 41}
    mus = {COMPARTMENT_IS: 0.0, FIBER_TYPE_I: mu, FIBER_TYPE_IIA: mu_IIa * mu, FIBER_TYPE_IIB: mu_IIb * mu}
    return CompartmentCoefficients(
        K={c: (1.0 if c == FIBER_TYPE_I else r4(v)) for c, v in K.items()},
        Mb={c: (r4(mb_rel[c] * beta) if switches.myoglobin and mb_rel[c] else 0.0) for c in COMPARTMENTS},
        mu={c: r4(v) for c, v in mus.items()}, p50=r4(pMb), pc=r4(params.p_c), kappa=rn(kappa),
        michaelis_menten=switches.michaelis_menten, myoglobin=switches.myoglobin, volume_fraction=frac,
        uptake_ratio={COMPARTMENT_IS: 0.0, FIBER_TYPE_I: 1.0, FIBER_TYPE_IIA: mu_IIa, FIBER_TYPE_IIB: mu_IIb},
        vol_avg_uptake_conversion=1.0 / norm, scales=sc)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@dataclass
class NewtonSettings:
    tol: float = 1e-10                 # ||R||_2 / ||R0||_2 (and absolute floor below)
    atol: float = 1e-13
    step_tol: float = 1e-12            # ||du||_inf
    max_iter: int = 50
    max_halvings: int = 20             # backtracking line search
    initial_guess: Optional[float] = None   # None: solve the linearised problem first
    verbose: bool = False


@dataclass
class PO2Solution:
    mesh: TissueMesh
    u: np.ndarray                      # nodal PO2 / Pcap
    params: DerivedParameters
    switches: ModelSwitches
    coefficients: CompartmentCoefficients
    iterations: int
    residual_history: List[float]
    converged: bool
    seconds: float

    @property
    def po2(self) -> np.ndarray:
        """Nodal PO2 in mmHg."""
        return self.params.raw.Pcap * self.u

    def summary(self) -> Dict[str, Any]:
        uc = _clamped(self.u)
        tri = self.mesh.triangles
        area = self.mesh.cell_areas()
        cen = self.params.raw.Pcap * uc[tri].mean(axis=1)
        return {"n_nodes": self.mesh.n_nodes, "n_triangles": self.mesh.n_triangles,
                "u_min_raw": float(self.u.min()), "u_max_raw": float(self.u.max()),
                "po2_min_mmHg": float(self.params.raw.Pcap * uc.min()),
                "po2_max_mmHg": float(self.params.raw.Pcap * uc.max()),
                "po2_area_weighted_mean_mmHg": float((cen * area).sum() / area.sum()),
                "iterations": self.iterations, "converged": self.converged, "seconds": self.seconds}


class _Assembler:
    """P1 assembly with scikit-fem (centroid rule in the volume, Gauss on walls)."""

    def __init__(self, mesh: TissueMesh, coef: CompartmentCoefficients):
        from skfem import Basis, ElementTriP1, FacetBasis

        self.mesh = mesh
        self.m = mesh.to_skfem()
        self.e = ElementTriP1()
        self.basis = Basis(self.m, self.e, intorder=1)          # 1 point: triangle centroid
        comp = mesh.cell_compartment()
        nq = self.basis.X.shape[1]
        self.K = np.array([coef.K[c] for c in comp])[:, None] * np.ones((1, nq))
        self.Mb = np.array([coef.Mb[c] for c in comp])[:, None] * np.ones((1, nq))
        self.mu = np.array([coef.mu[c] for c in comp])[:, None] * np.ones((1, nq))
        self.coef = coef

        # capillary-wall Robin terms: kappa * (M u - b), assembled once
        facets = _facet_ids(self.m, mesh.capillary_facets)
        if facets.size:
            fb = FacetBasis(self.m, self.e, facets=facets, intorder=2)
            from skfem import BilinearForm, LinearForm, asm

            @BilinearForm
            def mass(u, v, w):
                return u * v

            @LinearForm
            def one(v, w):
                return v

            self.M_wall = coef.kappa * asm(mass, fb).tocsr()
            self.b_wall = coef.kappa * asm(one, fb)
        else:
            n = mesh.n_nodes
            self.M_wall = csr_matrix((n, n))
            self.b_wall = np.zeros(n)

    def coefficients(self, uq: np.ndarray):
        """c, dc/du, g, dg/du at quadrature points.

        PO2 cannot be negative, but Newton iterates can be in anoxic regions.
        For u < 0 the coefficients are continued in a C1, monotone way (c
        frozen at u = 0; g = u / pc, the tangent of u/(u + pc) at 0), which
        removes the poles at u = -pc and u = -p50 and leaves every physical
        (u >= 0) solution unchanged.
        """
        cf = self.coef
        neg = uq < 0
        if cf.myoglobin:
            s = cf.p50 + np.maximum(uq, 0.0)
            c = self.K + self.Mb / s ** 2
            dc = np.where(neg, 0.0, -2.0 * self.Mb / s ** 3)
        else:
            c, dc = self.K, np.zeros_like(uq)
        if cf.michaelis_menten:
            s = np.maximum(uq, 0.0) + cf.pc
            g = np.where(neg, uq / cf.pc, uq / s)
            dg = np.where(neg, 1.0 / cf.pc, cf.pc / s ** 2)
        else:
            g, dg = np.ones_like(uq), np.zeros_like(uq)
        return c, dc, g, dg

    def residual_and_jacobian(self, u: np.ndarray, need_jacobian: bool = True):
        from skfem import BilinearForm, LinearForm, asm
        from skfem.helpers import dot, grad

        uk = self.basis.interpolate(u)
        c, dc, g, dg = self.coefficients(np.asarray(uk))

        @LinearForm
        def res(v, w):
            return w.c * dot(w.uk.grad, grad(v)) + w.mu * w.g * v

        R = asm(res, self.basis, uk=uk, c=c, mu=self.mu, g=g) + self.M_wall @ u - self.b_wall
        if not need_jacobian:
            return R, None

        @BilinearForm
        def jac(du, v, w):
            return (w.c * dot(grad(du), grad(v)) + w.dc * du * dot(w.uk.grad, grad(v))
                    + w.mu * w.dg * du * v)

        J = asm(jac, self.basis, uk=uk, c=c, dc=dc, mu=self.mu, dg=dg).tocsr() + self.M_wall
        return R, J


def _spsolve(A, b) -> np.ndarray:
    return np.asarray(spsolve(A.tocsc(), b), dtype=float)


def _facet_ids(m, pairs: np.ndarray) -> np.ndarray:
    """skfem facet indices of node pairs (order-insensitive)."""
    pairs = np.sort(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=1)
    if pairs.size == 0:
        return np.zeros(0, dtype=np.int64)
    f = np.sort(m.facets.T.astype(np.int64), axis=1)
    n = int(max(f.max(), pairs.max())) + 1
    keys = f[:, 0] * n + f[:, 1]
    order = np.argsort(keys)
    q = pairs[:, 0] * n + pairs[:, 1]
    pos = np.searchsorted(keys, q, sorter=order)
    pos = np.minimum(pos, len(keys) - 1)
    ids = order[pos]
    if not np.array_equal(keys[ids], q):
        raise ValueError("some capillary-wall facets are not mesh edges")
    return ids


def solve_po2(mesh: TissueMesh, params: DerivedParameters, switches: ModelSwitches,
              tissue: str = "skeletal", legacy_rounding: bool = True,
              newton: Optional[NewtonSettings] = None,
              coefficients: Optional[CompartmentCoefficients] = None,
              progress: Optional[ProgressCallback] = None) -> PO2Solution:
    """Solve for the nodal PO2 (u = PO2/Pcap) with Newton-Raphson.

    ``progress(fraction, message)`` (see :mod:`otm_core.progress`) is called after
    assembly, after the linear solve and after every Newton iteration; for Newton
    the fraction is the share of the residual reduction (in decades) achieved so far.
    """
    ns = newton or NewtonSettings()
    report(progress, 0.0, "Assembling the PO2 model")
    t0 = time.perf_counter()
    coef = coefficients or compartment_coefficients(mesh, params, switches, tissue, legacy_rounding)
    A = _Assembler(mesh, coef)
    nonlinear = coef.myoglobin or coef.michaelis_menten

    history: List[float] = []
    converged = False
    if not nonlinear or ns.initial_guess is None:
        # one linear solve: the exact answer for zero-order uptake without
        # facilitation, otherwise the Newton starting point
        lin = A
        if nonlinear:
            lin = copy.copy(A)
            lin.coef = replace(coef, michaelis_menten=False, myoglobin=False)
        R0, J0 = lin.residual_and_jacobian(np.zeros(mesh.n_nodes))
        assert J0 is not None
        report(progress, 0.05, "Solving the linear problem")
        u = _spsolve(J0, -R0)
        if nonlinear:
            u = np.clip(u, 0.0, 1.0)       # zero-order uptake overshoots below 0 in anoxic zones
    else:
        u = np.full(mesh.n_nodes, float(ns.initial_guess))
    it = 0
    if not nonlinear:
        R, _ = A.residual_and_jacobian(u, need_jacobian=False)
        history += [float(np.linalg.norm(R0)), float(np.linalg.norm(R))]
        it, converged = 1, True
    else:
        R, J = A.residual_and_jacobian(u)
        history.append(float(np.linalg.norm(R)))
        r0 = max(history[0], ns.atol)
        target = max(ns.tol * r0, ns.atol)
        decades = max(math.log10(r0 / target), 1e-12) if target < r0 else 1.0
        report(progress, 0.1, "Newton-Raphson: starting")
        for it in range(1, ns.max_iter + 1):
            assert J is not None
            du = _spsolve(J, -R)
            rn = history[-1]
            lam = 1.0
            u_try, r_try = u, rn
            for _ in range(ns.max_halvings + 1):
                u_try = u + lam * du
                R_try, _ = A.residual_and_jacobian(u_try, need_jacobian=False)
                r_try = float(np.linalg.norm(R_try))
                if r_try < (1 - 1e-4 * lam) * rn or r_try <= ns.atol:
                    break
                lam *= 0.5
            u = u_try
            history.append(r_try)
            if ns.verbose:
                print(f"newton {it:2d}: |R| = {r_try:.3e}  step = {lam:g}  |du|inf = "
                      f"{lam * np.abs(du).max():.3e}")
            done = math.log10(r0 / r_try) / decades if 0 < r_try < r0 else 0.0
            report(progress, 0.1 + 0.89 * min(done, 1.0),
                   f"Newton-Raphson iteration {it}: |R| = {r_try:.2e}")
            if r_try <= max(ns.tol * r0, ns.atol) or lam * np.abs(du).max() <= ns.step_tol:
                converged = True
                break
            R, J = A.residual_and_jacobian(u)
    report(progress, 1.0, "PO2 solution ready" if converged else "Newton-Raphson did not converge")
    return PO2Solution(mesh=mesh, u=u, params=params, switches=switches, coefficients=coef,
                       iterations=it, residual_history=history, converged=converged,
                       seconds=time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Post-processing (OxygenProfileStatistics, OxygenUptakeProfileStatistics,
# SingleFiberPO2Statistics)
# --------------------------------------------------------------------------


def _clamped(u: np.ndarray) -> np.ndarray:
    """Legacy ``u(u<0) = min(u(u>0))``."""
    u = np.asarray(u, dtype=float).copy()
    neg = u < 0
    if neg.any() and (u > 0).any():
        u[neg] = u[u > 0].min()
    return u


def _wstats(v: np.ndarray, w: np.ndarray, threshold: float):
    if w.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    m = float((v * w).sum() / w.sum())
    s = float(np.sqrt(((v - m) ** 2 * w).sum() / w.sum()))
    h = float(w[v <= threshold].sum() / w.sum())
    return m, s, h


def po2_statistics(sol: PO2Solution) -> Dict[str, Dict[str, float]]:
    """Area-weighted PO2 per compartment (mmHg) from triangle-centroid values:
    mean, SD and % hypoxia (PO2 <= P_c). Keys as in the legacy text output:
    Interstitia, FiberI, FiberIIa, FiberIIb, AllFibers, Tissue."""
    mesh = sol.mesh
    utr = sol.params.raw.Pcap * _clamped(sol.u)[mesh.triangles].mean(axis=1)
    area = mesh.cell_areas()
    comp = mesh.cell_compartment()
    uc = sol.params.raw.P_c
    out = {}
    for key, mask in (("Interstitia", comp == COMPARTMENT_IS), ("FiberI", comp == FIBER_TYPE_I),
                      ("FiberIIa", comp == FIBER_TYPE_IIA), ("FiberIIb", comp == FIBER_TYPE_IIB),
                      ("AllFibers", comp != COMPARTMENT_IS), ("Tissue", np.ones_like(comp, bool))):
        m, s, h = _wstats(utr[mask], area[mask], uc)
        out[key] = {"mean_mmHg": m, "std_mmHg": s, "hypoxia_pct": 100 * h}
    return out


def mo2_statistics(sol: PO2Solution) -> Dict[str, Dict[str, float]]:
    """O2 consumption per compartment (ml O2 / ml / s), as the legacy report.

    VO2max_X = mu_X^rel * conv * M0 * exercise; MO2 = VO2max_X * S with the
    saturation S = u/(u + pc) at triangle centroids (the legacy report always
    uses this Michaelis-Menten form, also when the PDE used zero-order uptake).
    '% MO2 < 0.5 VO2max' is the area fraction with S <= 0.5 (the legacy code
    indexes the area array incorrectly here; identical when there is no hypoxia).
    """
    mesh, cf = sol.mesh, sol.coefficients
    raw = sol.params.raw
    pc = sol.params.p_c
    utr = _clamped(sol.u)[mesh.triangles].mean(axis=1)
    S = utr / (pc + utr)
    area = mesh.cell_areas()
    comp = mesh.cell_compartment()
    M1 = cf.vol_avg_uptake_conversion * raw.M0 * sol.switches.exercise_level
    vmax = {c: cf.uptake_ratio[c] * M1 for c in COMPARTMENTS}
    fr = cf.volume_fraction
    out = {}
    for key, mask, vm in (
            ("Interstitia", comp == COMPARTMENT_IS, vmax[COMPARTMENT_IS]),
            ("FiberI", comp == FIBER_TYPE_I, vmax[FIBER_TYPE_I]),
            ("FiberIIa", comp == FIBER_TYPE_IIA, vmax[FIBER_TYPE_IIA]),
            ("FiberIIb", comp == FIBER_TYPE_IIB, vmax[FIBER_TYPE_IIB]),
            ("AllFibers", comp != COMPARTMENT_IS,
             sum(area[comp == c].sum() * vmax[c] for c in COMPARTMENTS[1:]) / area[comp != 0].sum()),
            ("Tissue", np.ones_like(comp, bool), sum(fr[c] * vmax[c] for c in COMPARTMENTS))):
        m, s, h = _wstats(S[mask], area[mask], 0.5)
        out[key] = {"vo2max": float(vm), "mean_mo2": float(vm * m), "std_mo2": float(vm * s),
                    "pct_below_half_vo2max": 100 * h}
    return out


def fiber_po2_statistics(sol: PO2Solution) -> Dict[str, np.ndarray]:
    """Per model fibre: area-weighted mean / SD PO2 (mmHg) and hypoxic fraction."""
    mesh = sol.mesh
    utr = sol.params.raw.Pcap * _clamped(sol.u)[mesh.triangles].mean(axis=1)
    area = mesh.cell_areas()
    nf = len(mesh.fiber_types)
    reg = mesh.cell_region
    fib = reg >= 0
    w = np.bincount(reg[fib], weights=area[fib], minlength=nf)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(reg[fib], weights=(utr * area)[fib], minlength=nf) / w
        var = np.bincount(reg[fib], weights=((utr - mean[np.maximum(reg, 0)]) ** 2 * area)[fib],
                          minlength=nf) / w
        hyp = np.bincount(reg[fib], weights=(area * (utr <= sol.params.raw.P_c))[fib], minlength=nf) / w
    return {"fiber": np.arange(nf), "fiber_type": mesh.fiber_types.copy(), "mean_po2_mmHg": mean,
            "std_po2_mmHg": np.sqrt(var), "hypoxic_fraction": hyp}
