"""Result tables as text rows, shared by the desktop pages and the web pages.

Pure functions: no GUI code. Each returns rows of display strings in the order
the desktop app's Supply indices and PO₂ solution pages show them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["indices_rows", "po2_rows", "mo2_rows", "COMPARTMENTS", "hypoxic_compartments", "hypoxia_text"]


def indices_rows(m: Any) -> List[Tuple[str, str, str]]:
    """(name, value, unit) rows of a :class:`otm_core.Morphometrics`, as in the
    legacy IndicesMenu summary table."""
    d, nn, s = m.domains, m.nearest_neighbours, m.supply

    def ms(stats, fmt="{:.2f}"):
        return f"{fmt.format(stats.mean)} ± {fmt.format(stats.std)}" if stats.n else "–"

    rows = [
        ("Number of capillaries", f"{d.total_num_capillaries}", ""),
        ("Capillary density", f"{d.capillary_density_per_mm2:.1f}", "mm⁻²"),
        ("Capillary density in ROI", f"{d.roi_capillary_density_per_mm2:.1f}", "mm⁻²"),
        ("Capillary domains in ROI", f"{d.n_domains_in_roi}", ""),
        ("Domain area", ms(d.area_stats, "{:.0f}"), "µm²"),
        ("Domain equivalent diameter", ms(d.diameter_stats, "{:.1f}"), "µm"),
        ("Nearest-neighbour distance (mean)", f"{nn.summary['MeanOfMeans']:.1f}", "µm"),
        ("Nearest-neighbour distance (minimum)", f"{nn.summary['MeanOfMin']:.1f}", "µm"),
    ]
    if s is not None:
        rows += [
            ("Fibres in ROI", f"{s.number_of_fibers}", ""),
            ("Capillary : fibre ratio", f"{s.capillary_to_fiber_ratio:.2f}", ""),
            ("Fibre area", f"{np.mean(s.fiber_area_um2):.0f}" if len(s.fiber_area_um2) else "–", "µm²"),
            ("LCFR (local capillary-to-fibre ratio)", ms(s.LCFR), ""),
            ("LCD (local capillary density)", ms(s.LCD, "{:.0f}"), "mm⁻²"),
            ("DFR (domains per fibre)", ms(s.DFR), ""),
            ("FDR (fibres per domain)", ms(s.FDR), ""),
            ("Dmax (maximum diffusion distance)", ms(s.Dmax, "{:.1f}"), "µm"),
            ("CC (capillaries around a fibre)", ms(s.CC), ""),
            ("CFi (individual capillary-to-fibre ratio)", ms(s.CFi), ""),
            ("Fibre perimeter", ms(s.FPi, "{:.1f}"), "µm"),
            ("CFPE (capillary-to-fibre perimeter exchange)", ms(s.CFPE), "per 1000 µm"),
            ("SF (sharing factor)", ms(s.SF), ""),
        ]
    return rows


COMPARTMENTS = (("Tissue", "Tissue"), ("Interstitia", "Interstitial space"), ("AllFibers", "All fibres"),
                ("FiberI", "Type I"), ("FiberIIa", "Type IIa"), ("FiberIIb", "Type IIb"))


def po2_rows(stats: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, str, str]]:
    """(compartment, mean, SD, % hypoxic) rows of ``otm_core.fem.po2_statistics``."""
    out = []
    for key, label in COMPARTMENTS:
        v = stats.get(key)
        if v is None or not np.isfinite(v.get("mean_mmHg", np.nan)):
            continue
        out.append((label, f"{v['mean_mmHg']:.2f}", f"{v['std_mmHg']:.2f}", f"{v['hypoxia_pct']:.1f}"))
    return out


def mo2_rows(stats: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, str, str]]:
    """(compartment, VO2max, mean MO2, SD) rows of ``otm_core.fem.mo2_statistics``."""
    out = []
    for key, label in COMPARTMENTS:
        v = stats.get(key)
        if v is None or not np.isfinite(v.get("mean_mo2", np.nan)):
            continue
        out.append((label, f"{v['vo2max']:.3g}", f"{v['mean_mo2']:.3g}", f"{v['std_mo2']:.3g}"))
    return out


def hypoxic_compartments(rows: Sequence[Sequence[str]]) -> List[Tuple[int, str, float]]:
    """(row index, compartment, % hypoxic) of the :func:`po2_rows` rows with any
    hypoxic area (PO₂ at or below P_c)."""
    out = []
    for i, row in enumerate(rows):
        try:
            pct = float(row[3])
        except (ValueError, IndexError):
            continue
        if pct > 0:
            out.append((i, str(row[0]), pct))
    return out


def hypoxia_text(rows: Sequence[Sequence[str]]) -> Optional[str]:
    """The desktop's hypoxia tag for :func:`po2_rows`, or None when nothing is hypoxic."""
    hyp = [(n, p) for _i, n, p in hypoxic_compartments(rows)]
    if not hyp:
        return None
    tissue = dict(hyp).get("Tissue")
    pct = tissue if tissue is not None else max(p for _n, p in hyp)
    where = "tissue" if tissue is not None else hyp[0][0].lower()
    return f"HYPOXIA · {pct:.1f} % of the {where} at or below P_c ({', '.join(n for n, _p in hyp)})"
