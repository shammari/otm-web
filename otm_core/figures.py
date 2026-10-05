"""Publication figures: journal-sized, vector-ready versions of the OTM plots.

The legacy figures in :mod:`otm_core.export` reproduce the MATLAB layouts (and
the regression tests check them). This module draws the same results for
print. ``FigureStyle(layout="legacy")`` keeps the MATLAB look; the default is
``layout="publication"``:

* **Size**: the figure is made at the final printed width - one journal column
  (85 mm), 1.5 columns (114 mm) or two columns (175 mm) - so 7 pt text stays
  7 pt in the article.
* **Text**: one sans-serif family (Arial / Helvetica, else Liberation or
  DejaVu Sans), 7 pt labels, 6 pt ticks; proper symbols (PO₂, µm, mm⁻²);
  fonts embedded as TrueType in PDF and kept as editable text in SVG.
* **Formats**: any of PNG, PDF, SVG and TIFF (LZW) in one go; dense fields
  (the PO₂ map) are rasterised at the chosen dpi inside the vector files so the
  PDF stays small, while outlines, text and axes stay vector.
* **Maps**: a µm scale bar, a slim colour bar beside the map, uniform thin
  fibre borders, capillaries drawn at their true radius, and a legend.
* **Distributions**: flat histograms (no 3-D bars) normalised within each
  group, step cumulative curves, a colour-blind-checked palette for the fibre
  types plus a second encoding (line style) and the group sizes in the legend.
* **VO₂**: the consumption per compartment (area-weighted mean ± SD) and its
  cumulative distribution, instead of density spikes.

Everything is plain matplotlib (Agg); nothing here needs Qt.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FigureStyle", "LAYOUTS", "WIDTHS_MM", "FORMATS", "save_figure", "rc_params",
    "map_figure", "po2_distribution_figure", "vo2_figure", "index_figure", "flux_map_figure",
    "pretty_index_label", "nice_scale_length",
]

LAYOUTS = ("publication", "legacy")
WIDTHS_MM = {"single": 85.0, "onehalf": 114.0, "double": 175.0}
WIDTH_LABELS = {"single": "Single column (85 mm)", "onehalf": "1.5 columns (114 mm)",
                "double": "Double column (175 mm)"}
FORMATS = ("png", "pdf", "svg", "tiff")
MM = 1.0 / 25.4

# fibre types: first three slots of the validated categorical palette (all-pairs
# CVD dE >= 9.2, normal-vision dE >= 24); identity is also carried by line style
TYPE_COLOURS = {1: "#2a78d6", 21: "#eb6834", 22: "#1baf7a", 0: "#8c8c8c"}
TYPE_NAMES = {1: "Type I", 21: "Type IIa", 22: "Type IIb", 0: "Unknown type"}
TYPE_DASHES = {1: "-", 21: (0, (5, 1.6)), 22: (0, (1.4, 1.2)), 0: (0, (3, 1, 1, 1))}
INK, INK_2, GRID = "#1a1a1a", "#555555", "#d9d9d9"
CAPILLARY_RED = "#b2182b"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial missing -> fallback, quietly


@dataclass(frozen=True)
class FigureStyle:
    """How exported figures look and which files are written."""

    layout: str = "publication"                     # "publication" | "legacy" (MATLAB look)
    width: str = "double"                           # "single" | "onehalf" | "double" (journal widths)
    formats: Tuple[str, ...] = ("png", "pdf")
    dpi: int = 600                                  # raster files and rasterised parts of vector files
    font_size: float = 7.0

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if self.width not in WIDTHS_MM:
            raise ValueError(f"width must be one of {tuple(WIDTHS_MM)}")
        fmts = tuple(dict.fromkeys(str(f).lower().lstrip(".").replace("tif", "tiff").replace("tifff", "tiff")
                                   for f in self.formats))
        bad = set(fmts) - set(FORMATS)
        if bad or not fmts:
            raise ValueError(f"formats must be a non-empty subset of {FORMATS}; got {self.formats}")
        object.__setattr__(self, "formats", fmts)
        if not 20 <= int(self.dpi) <= 2400:
            raise ValueError("dpi must be between 20 and 2400")
        object.__setattr__(self, "dpi", int(self.dpi))
        if not 5 <= float(self.font_size) <= 14:
            raise ValueError("font_size must be between 5 and 14 pt")

    @classmethod
    def legacy(cls, formats: Sequence[str] = ("png",), dpi: int = 300) -> "FigureStyle":
        return cls(layout="legacy", formats=tuple(formats), dpi=dpi)

    @property
    def publication(self) -> bool:
        return self.layout == "publication"

    @property
    def width_in(self) -> float:
        return WIDTHS_MM[self.width] * MM

    def with_changes(self, **kw: Any) -> "FigureStyle":
        d = asdict(self)
        d.update(kw)
        return FigureStyle(**d)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["formats"] = list(self.formats)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FigureStyle":
        if not d:
            return cls()
        names = {"layout", "width", "formats", "dpi", "font_size"}
        kw = {k: v for k, v in dict(d).items() if k in names}
        if "formats" in kw:
            kw["formats"] = tuple(kw["formats"])
        return cls(**kw)

    def describe(self) -> str:
        if not self.publication:
            return f"legacy MATLAB layout · {', '.join(f.upper() for f in self.formats)} · {self.dpi} dpi"
        return (f"publication · {WIDTHS_MM[self.width]:g} mm wide · {', '.join(f.upper() for f in self.formats)}"
                f" · {self.dpi} dpi")


# ==========================================================================
# style and saving
# ==========================================================================


def rc_params(style: FigureStyle) -> Dict[str, Any]:
    fs = float(style.font_size)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs, "legend.fontsize": fs - 0.5,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelpad": 2.5, "lines.linewidth": 1.0, "legend.frameon": False, "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.3, "mathtext.default": "regular", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 100, "savefig.dpi": style.dpi, "savefig.facecolor": "white",
        "figure.constrained_layout.h_pad": 2 * MM, "figure.constrained_layout.w_pad": 2 * MM,
    }


def _ctx(style: FigureStyle):
    import matplotlib

    return matplotlib.rc_context(rc_params(style))


def _new_figure(style: FigureStyle, height_in: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(style.width_in, height_in), facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    fig._otm_style = style                      # save_figure uses the same rc settings
    return fig


def save_figure(fig: Any, stem: str, style: FigureStyle, dpi: Optional[int] = None) -> List[str]:
    """Write ``stem.<fmt>`` for every format of ``style``; returns the paths.
    ``dpi`` overrides the style's resolution (tests use a low one)."""
    res = int(dpi or style.dpi)
    out = []
    rc = rc_params(getattr(fig, "_otm_style", style)) if style.publication else {}
    import matplotlib

    with matplotlib.rc_context(rc):
        for fmt in style.formats:
            path = f"{stem}.{fmt}"
            kw: Dict[str, Any] = {"dpi": res, "facecolor": "white"}
            if fmt == "tiff":
                kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            if fmt == "pdf":
                kw["metadata"] = {"Creator": "OTM Lite (Oxygen Transport Modeller)"}
            fig.savefig(path, format=fmt, **kw)
            out.append(path)
    return out


# ==========================================================================
# helpers
# ==========================================================================


def nice_scale_length(span_um: float, fraction: float = 0.2) -> float:
    """A round scale-bar length (1, 2 or 5 x 10^n µm) close to ``fraction`` of ``span_um``."""
    target = max(span_um * fraction, 1e-9)
    p = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * p <= target * 1.0000001:
            return float(m * p)
    return float(p)


def _scale_bar(ax: Any, span_um: float, height_um: float, style: FigureStyle) -> None:
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    L = nice_scale_length(span_um)
    bar = AnchoredSizeBar(ax.transData, L, f"{L:g} µm", "lower right", pad=0.25, borderpad=0.35, sep=1.5,
                          frameon=True, color=INK, size_vertical=max(height_um * 0.007, 1e-9),
                          fontproperties=FontProperties(size=style.font_size - 0.5), label_top=True)
    bar.patch.set_facecolor("white")
    bar.patch.set_alpha(0.85)
    bar.patch.set_edgecolor("none")
    ax.add_artist(bar)


def _fibre_rings_um(geometry: Any) -> List[Tuple[np.ndarray, int]]:
    from .export import _rings, _um

    out = []
    fibers = getattr(geometry, "fibers", None)
    types = getattr(geometry, "fiber_types", None)
    if fibers is None or types is None:
        return []
    for f, t in zip(fibers, types):
        for r in _rings(f):
            out.append((_um(r, geometry), int(t)))
    return out


def _capillary_patches(mesh: Any, geometry: Any, min_radius_um: float = 0.0):
    from matplotlib.patches import Circle

    from .export import _um

    ld = geometry.length_scale.ld if getattr(geometry, "length_scale", None) is not None else 1.0
    caps = _um(mesh.capillaries, geometry)
    return [Circle(c, max(float(r) * ld, min_radius_um)) for c, r in zip(caps, mesh.rcap)]


def _panel_letter(ax: Any, letter: str) -> None:
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right",
            fontsize=ax.xaxis.label.get_fontsize() + 1)


def _two_panels(style: FigureStyle, panel_h_mm: float = 52.0):
    """Side by side when the figure is wide enough, else stacked."""
    if style.width == "single":
        fig = _new_figure(style, 2 * panel_h_mm * MM + 6 * MM)
        return fig, fig.subplots(2, 1)
    fig = _new_figure(style, panel_h_mm * MM + 4 * MM)
    return fig, fig.subplots(1, 2)


# ==========================================================================
# maps (PO2 / VO2)
# ==========================================================================

VO2_LABEL = "VO$_2$ (ml O$_2$ 100 ml$^{-1}$ min$^{-1}$)"
PO2_LABEL = "PO$_2$ (mmHg)"


def map_figure(solution: Any, geometry: Any, field: str = "po2", style: Optional[FigureStyle] = None,
               cmap: str = "viridis", clim: Optional[Tuple[float, float]] = None, tissue: str = "skeletal",
               roi: Any = None):
    """PO2 (Gouraud-shaded, rasterised) or VO2 (per triangle) over the section,
    fibre borders, capillaries at their true size, scale bar, colour bar, legend."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.tri import Triangulation

    from .export import _um, _uptake_field

    style = style or FigureStyle()
    with _ctx(style):
        mesh = solution.mesh
        pts = _um(mesh.points, geometry)
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        w, h = (x1 - x0) or 1.0, (y1 - y0) or 1.0
        map_w = style.width_in - 19 * MM            # the colour bar and its label take ~19 mm
        fig = _new_figure(style, map_w * h / w + 9 * MM)
        ax = fig.add_subplot()
        tri = Triangulation(pts[:, 0], pts[:, 1], mesh.triangles)
        extend = "neither"
        if field == "po2":
            vals = solution.params.raw.Pcap * np.clip(solution.u, 0, None)
            art = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, rasterized=True)
            label = PO2_LABEL
        else:
            V, _, _, _ = _uptake_field(solution, tissue)
            vals = V
            art = ax.tripcolor(tri, facecolors=V, cmap=cmap, rasterized=True, edgecolors="none")
            label = VO2_LABEL
        if clim is not None:
            art.set_clim(*clim)
            v = np.asarray(vals, float)
            lo, hi = v.min() < clim[0], v.max() > clim[1]
            extend = "both" if lo and hi else "min" if lo else "max" if hi else "neither"
        handles = []
        if tissue != "cardiac":
            rings = _fibre_rings_um(geometry)
            if rings:
                ax.add_collection(LineCollection([np.vstack([r, r[:1]]) for r, _ in rings], colors=INK,
                                                 linewidths=0.35, capstyle="round", joinstyle="round"))
                handles.append(Line2D([], [], color=INK, lw=0.6, label="Fibre border"))
        caps = _capillary_patches(mesh, geometry, min_radius_um=0.004 * w)
        ax.add_collection(PatchCollection(caps, facecolors="white", edgecolors=INK, linewidths=0.3, zorder=3))
        handles.append(Line2D([], [], marker="o", ls="none", markersize=3.2, markerfacecolor="white",
                              markeredgecolor=INK, markeredgewidth=0.5, label="Capillary"))
        if roi is not None:
            ax.add_patch(Rectangle((roi.x_min_um, roi.y_min_um), roi.x_max_um - roi.x_min_um,
                                   roi.y_max_um - roi.y_min_um, fill=False, edgecolor="white", lw=0.8,
                                   ls=(0, (4, 2)), zorder=4))
            handles.append(Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)), label="Region of interest"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        _scale_bar(ax, w, h, style)
        cb = fig.colorbar(art, ax=ax, extend=extend, fraction=0.05, pad=0.015, aspect=28,
                          extendfrac=0.03)
        cb.set_label(label + (f", fixed scale" if clim is not None and field == "po2" else ""))
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2)
        cb.ax.minorticks_off()
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), handletextpad=0.5,
                   columnspacing=1.6)
    return fig


# ==========================================================================
# PO2 distribution
# ==========================================================================

_GROUP_STYLE = {   # label -> (colour, dash, width)
    "Tissue": (INK, "-", 1.4),
    "Interstitial space": ("#8c8c8c", (0, (4, 2)), 1.0),
    "Type I": (TYPE_COLOURS[1], TYPE_DASHES[1], 1.0),
    "Type IIa": (TYPE_COLOURS[21], TYPE_DASHES[21], 1.0),
    "Type IIb": (TYPE_COLOURS[22], TYPE_DASHES[22], 1.0),
}
_RENAME = {"Interstitia": "Interstitial space"}


def po2_distribution_figure(curves: Any, style: Optional[FigureStyle] = None):
    """(a) probability density of PO2, (b) cumulative fraction of the tissue area;
    one curve per compartment (compartment curves sum to the tissue curve)."""
    style = style or FigureStyle()
    with _ctx(style):
        fig, (a, b) = _two_panels(style)
        hi = curves.x[-1] if curves.x[-1] > curves.xmin else curves.xmin + 1
        for ax, data in ((a, curves.pdf), (b, curves.cdf)):
            for lab, y in data.items():
                name = _RENAME.get(lab, lab)
                col, dash, lw = _GROUP_STYLE.get(name, (INK_2, "-", 1.0))
                ax.plot(curves.x, y, color=col, ls=dash, lw=lw, label=name, solid_capstyle="round")
            ax.set_xlim(curves.xmin, hi)
            ax.set_xlabel(PO2_LABEL)
            ax.margins(y=0.02)
        a.set_ylim(bottom=0)
        b.set_ylim(0, 1.0)
        a.set_ylabel("Probability density (mmHg$^{-1}$)")
        b.set_ylabel("Cumulative fraction of tissue area")
        a.legend(loc="upper left")
        _panel_letter(a, "a")
        _panel_letter(b, "b")
    return fig


# ==========================================================================
# VO2: per compartment
# ==========================================================================


def _weighted(values: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    if values.size == 0 or w.sum() <= 0:
        return float("nan"), float("nan")
    m = float(np.average(values, weights=w))
    return m, float(math.sqrt(max(np.average((values - m) ** 2, weights=w), 0.0)))


def vo2_figure(solution: Any, tissue: str = "skeletal", style: Optional[FigureStyle] = None):
    """(a) O2 consumption per compartment: area-weighted mean ± SD with the share
    of the tissue area; (b) cumulative fraction of the tissue area against VO2."""
    from .export import _uptake_field
    from .fem.solve import COMPARTMENT_IS, FIBER_TYPE_I, FIBER_TYPE_IIA, FIBER_TYPE_IIB

    style = style or FigureStyle()
    V, _S, area, _M1 = _uptake_field(solution, tissue)
    total = float(area.sum()) or 1.0
    groups = [("Tissue", np.ones(V.size, bool))]
    if tissue != "cardiac":
        comp = solution.mesh.cell_compartment()
        for code, lab in ((COMPARTMENT_IS, "Interstitial space"), (FIBER_TYPE_I, "Type I"),
                          (FIBER_TYPE_IIA, "Type IIa"), (FIBER_TYPE_IIB, "Type IIb")):
            m = comp == code
            if m.any():
                groups.append((lab, m))
    with _ctx(style):
        fig, (a, b) = _two_panels(style)
        ys = np.arange(len(groups))[::-1]
        vmax = float(np.nanmax(V)) if V.size else 1.0
        for y, (lab, m) in zip(ys, groups):
            col, dash, _lw = _GROUP_STYLE.get(lab, (INK_2, "-", 1.0))
            mean, sd = _weighted(V[m], area[m])
            a.errorbar(mean, y, xerr=sd, fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.0, capsize=2,
                       markeredgecolor="white", markeredgewidth=0.5, zorder=3)
            a.annotate(f"{mean:.2f}  ({100 * area[m].sum() / total:.0f} % of area)", (mean + sd, y),
                       xytext=(4, 0), textcoords="offset points", va="center", fontsize=style.font_size - 1,
                       color=INK_2)
            order = np.argsort(V[m])
            xs = V[m][order]
            cum = np.cumsum(area[m][order]) / total
            b.step(np.r_[0.0, xs, max(vmax, 1e-12) * 1.02], np.r_[0.0, cum, cum[-1] if cum.size else 0.0],
                   where="post", color=col, ls=dash, lw=1.4 if lab == "Tissue" else 1.0, label=lab)
        a.set_yticks(ys)
        a.set_yticklabels([g[0] for g in groups])
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.set_xlim(-0.03 * max(vmax, 1e-12), max(vmax, 1e-12) * 1.45)
        a.set_ylim(-0.6, len(groups) - 0.4)
        a.grid(axis="x", color=GRID, lw=0.4)
        a.set_axisbelow(True)
        a.set_xlabel(VO2_LABEL)
        b.set_xlim(0, max(vmax, 1e-12) * 1.02)
        b.set_ylim(0, 1.0)
        b.set_xlabel(VO2_LABEL)
        b.set_ylabel("Cumulative fraction of tissue area")
        b.legend(loc="upper left")
        _panel_letter(a, "a")
        _panel_letter(b, "b")
    return fig


# ==========================================================================
# supply-index distributions
# ==========================================================================

_INDEX_NAMES = {
    "Fiber area": "Fibre cross-sectional area", "Capillary domain Area": "Capillary domain area",
    "Equivalent diameter": "Domain equivalent diameter", "Mean NN distance": "Mean nearest-neighbour distance",
    "Minimum NN distance": "Minimum nearest-neighbour distance",
    "Unique minimum NN distance": "Unique minimum nearest-neighbour distance",
    "Unique random NN distance": "Unique random nearest-neighbour distance",
    "All NN distances": "Nearest-neighbour distance (all pairs)",
}
_UNITS = {" (um sq)": " (µm$^2$)", " (um)": " (µm)", " (1/mm sq)": " (mm$^{-2}$)", "": ""}
_ENTITY = {"Fiber area": "fibres", "DFR": "fibres", "FDR": "fibres", "LCFR": "fibres", "LCD": "fibres",
           "All NN distances": "distances"}


def pretty_index_label(name: str, units: str = "") -> str:
    return _INDEX_NAMES.get(name, name) + _UNITS.get(units, units.replace("um", "µm"))


def index_figure(values: np.ndarray, name: str, units: str = "", fiber_types: Optional[np.ndarray] = None,
                 style: Optional[FigureStyle] = None):
    """(a) histogram (fraction within each group), (b) cumulative fraction; split
    by fibre type for fibre area, LCFR and LCD when there are two types or more."""
    style = style or FigureStyle()
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    groups: List[Tuple[str, np.ndarray, Any, Any, float]] = []      # label, values, colour, dash, width
    entity = _ENTITY.get(name, "capillaries")
    if fiber_types is not None and name in ("LCD", "LCFR", "Fiber area") and np.size(fiber_types) == v.size:
        t = np.asarray(fiber_types, int)
        for code in (1, 21, 22, 0):
            m = (t == code) & ok
            if m.any():
                groups.append((TYPE_NAMES[code], v[m], TYPE_COLOURS[code], TYPE_DASHES[code], 1.0))
        if len(groups) < 2:
            groups = []
    allv = v[ok]
    with _ctx(style):
        fig, (a, b) = _two_panels(style, panel_h_mm=48.0)
        xlabel = pretty_index_label(name, units)
        if allv.size == 0:
            for ax in (a, b):
                ax.text(0.5, 0.5, "no values", transform=ax.transAxes, ha="center", va="center", color=INK_2)
        else:
            edges = np.histogram_bin_edges(allv, bins="auto")
            if edges.size > 31:
                edges = np.linspace(allv.min(), allv.max(), 31)
            if edges.size < 7 and allv.max() > allv.min():
                edges = np.linspace(allv.min(), allv.max(), 7)
            if allv.max() == allv.min():
                edges = np.array([allv.min() - 0.5, allv.max() + 0.5])
            if name in ("DFR",) and np.allclose(allv, np.round(allv)):            # counts: one bar per integer
                edges = np.arange(allv.min() - 0.5, allv.max() + 1.5, 1.0)
            # (a) the whole sample as grey bars; the types on top as step outlines
            wts = np.full(allv.size, 1.0 / allv.size)
            a.hist(allv, bins=edges, weights=wts, color="#c9ccd3", edgecolor="white", linewidth=0.5,
                   label=f"All {entity} (n = {allv.size})")
            if not groups:                                  # one series: no legend box, just n
                a.text(1.0, 1.0, f"n = {allv.size} {entity}", transform=a.transAxes, ha="right", va="top",
                       fontsize=style.font_size - 0.5, color=INK_2)
            for lab, g, col, dash, lw in groups:
                a.hist(g, bins=edges, weights=np.full(g.size, 1.0 / g.size), histtype="step", color=col,
                       ls=dash, lw=lw, label=f"{lab} (n = {g.size})")
            # (b) cumulative
            for lab, g, col, dash, lw in [(f"All {entity}", allv, INK, "-", 1.4)] + groups:
                xs = np.sort(g)
                ys = np.arange(1, xs.size + 1) / xs.size
                b.step(np.r_[edges[0], xs, edges[-1]], np.r_[0.0, ys, 1.0], where="post", color=col, ls=dash,
                       lw=lw, label=lab)
            for ax in (a, b):
                ax.set_xlim(edges[0], edges[-1])
            a.set_ylim(bottom=0)
            b.set_ylim(0, 1.0)
            med = float(np.median(allv))
            b.axhline(0.5, color=GRID, lw=0.5, zorder=0)
            b.plot([med], [0.5], marker="|", ms=6, mew=1.0, color=INK, zorder=4)
            b.text(0.01, 0.52, f"median {med:.3g}", transform=b.transAxes, ha="left", va="bottom",
                   fontsize=style.font_size - 1, color=INK_2)
            if groups:                                      # several series: one legend above both panels
                h, l = a.get_legend_handles_labels()
                fig.legend(h, l, loc="outside upper center", ncol=len(l) if style.width != "single" else 2,
                           handlelength=1.8)
        a.set_xlabel(xlabel)
        b.set_xlabel(xlabel)
        a.set_ylabel(f"Fraction of {entity}" + (" (within group)" if groups else ""))
        b.set_ylabel(f"Cumulative fraction of {entity}")
        _panel_letter(a, "a")
        _panel_letter(b, "b")
    return fig


# ==========================================================================
# flux lines
# ==========================================================================


def _tint(hex_colour: str, amount: float = 0.6) -> Tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(hex_colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def flux_map_figure(flux: Any, geometry: Any, mesh: Any, roi: Any = None, style: Optional[FigureStyle] = None):
    """Fibres tinted by type, capillaries at their true size, flux lines, ROI,
    scale bar and legend."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle

    from .export import _flux_paths, _um

    style = style or FigureStyle()
    with _ctx(style):
        pts = _um(mesh.points, geometry)
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        w, h = (x1 - x0) or 1.0, (y1 - y0) or 1.0
        map_w = min(style.width_in * 0.96, 150 * MM)
        fig = _new_figure(style, map_w * h / w + 9 * MM)
        ax = fig.add_subplot()
        handles = []
        rings = _fibre_rings_um(geometry)
        present = []
        if rings:
            ax.add_collection(PatchCollection([Polygon(r, closed=True) for r, _ in rings],
                                              facecolors=[_tint(TYPE_COLOURS.get(t, TYPE_COLOURS[0])) for _, t in rings],
                                              edgecolors=INK, linewidths=0.3))
            present = [t for t in (1, 21, 22, 0) if any(tt == t for _, tt in rings)]
            handles += [Patch(facecolor=_tint(TYPE_COLOURS[t]), edgecolor=INK, lw=0.3, label=TYPE_NAMES[t])
                        for t in present]
        lines = [_um(p, geometry) for group in _flux_paths(flux) for p in group if len(p) > 1]
        ax.add_collection(LineCollection(lines, colors=INK, linewidths=0.35, capstyle="round", zorder=3))
        handles.append(Line2D([], [], color=INK, lw=0.8, label="Flux line"))
        ax.add_collection(PatchCollection(_capillary_patches(mesh, geometry, min_radius_um=0.004 * w),
                                          facecolors=CAPILLARY_RED, edgecolors="white", linewidths=0.25, zorder=4))
        handles.append(Line2D([], [], marker="o", ls="none", markersize=3.2, markerfacecolor=CAPILLARY_RED,
                              markeredgecolor="white", markeredgewidth=0.3, label="Capillary"))
        if roi is not None:
            ax.add_patch(Rectangle((roi.x_min_um, roi.y_min_um), roi.x_max_um - roi.x_min_um,
                                   roi.y_max_um - roi.y_min_um, fill=False, edgecolor=INK, lw=0.8,
                                   ls=(0, (4, 2)), zorder=5))
            handles.append(Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)), label="Region of interest"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        _scale_bar(ax, w, h, style)
        fig.legend(handles=handles, loc="outside lower center",
                   ncol=min(len(handles), 7 if style.width != "single" else 3), handletextpad=0.5,
                   columnspacing=1.4)
    return fig
