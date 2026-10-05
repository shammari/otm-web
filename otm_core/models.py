"""Plain data containers for the OTM core.

Everything here is NumPy-only (no Shapely import, no GUI), so these objects
can be pickled, cached by Streamlit/Dash, or sent between processes. Shapely
geometries only appear as *values* (object arrays) inside a few results.

Index convention
----------------
All indices are **0-based**. The MATLAB baseline (``baseline_outputs.json``)
is 1-based; use :func:`to_matlab_index` / :func:`from_matlab_index` when
comparing.

Coordinate frames (report section 3.3)
--------------------------------------
``px``    Dtect pixel frame (raw input).
``px_c``  Centred pixel frame after the axis swap/flip done in
          ``TissueGeometryType`` (see :func:`otm_core.geometry.dtect_to_centred`).
``ndim``  Non-dimensional frame: ``px_c / (min(ImageSize)/2)``. Geometry,
          mesh, PDE and flux lines live here.
``um``    Display/ROI frame: ``um = ld * (ndim + ss)`` with
          ``ld = min(Lx, Ly)/2`` and ``ss = [W, H] / min(W, H)``.
``um_c``  Centred micrometre frame of the index routines:
          ``um_c = (Ly/2) * ndim`` (legacy ``LengthScale = HeightLengthScale/2``).

Fibre representation
--------------------
A fibre is either a single ring (``(n, 2)`` array, the legacy ``.x/.y``
struct), a list of rings (a multi-part fibre), or a Shapely
Polygon/MultiPolygon. :func:`otm_core.geometry.fiber_geometries` normalises
all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

Ring = np.ndarray
"""A polygon ring: float array of shape (n, 2). Usually closed (first == last)."""

FiberLike = Union[Ring, List[Ring], Any]
"""Single ring, list of rings (multi-part fibre) or a Shapely (Multi)Polygon."""

FIBER_TYPE_I = 1
FIBER_TYPE_IIA = 21
FIBER_TYPE_IIB = 22
FIBER_TYPE_UNKNOWN = 0


def to_matlab_index(idx) -> np.ndarray:
    """0-based -> 1-based (for comparison with the MATLAB baseline)."""
    return np.asarray(idx, dtype=np.int64) + 1


def from_matlab_index(idx: Sequence[int]) -> np.ndarray:
    """1-based (MATLAB) -> 0-based."""
    return np.asarray(idx, dtype=np.int64) - 1


# --------------------------------------------------------------------------
# Frames and scales
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageFrame:
    """Pixel frame of the source micrograph. ``image_size`` is MATLAB's
    ``ImageSize`` = (rows, cols) = (height, width)."""

    image_size: Tuple[int, int]

    @property
    def height(self) -> float:
        return float(self.image_size[0])

    @property
    def width(self) -> float:
        return float(self.image_size[1])

    @property
    def ndim_scale_px(self) -> float:
        """Pixels per non-dimensional unit: ``min(ImageSize)/2`` (ScaleGeometry)."""
        return min(self.image_size) / 2.0

    @property
    def ss(self) -> np.ndarray:
        """``imShift / min(imShift)`` used by every legacy plot."""
        shift = np.array([self.width, self.height]) / 2.0
        return shift / shift.min()

    @property
    def aspect_ratio(self) -> float:
        """width / height (TissueDimensions ``ratio``)."""
        return self.width / self.height


@dataclass(frozen=True)
class LengthScale:
    """Physical tissue size in micrometres (TissueDimensions)."""

    x_um: float
    y_um: float

    @classmethod
    def from_width(cls, width_um: float, frame: ImageFrame) -> "LengthScale":
        """Height follows the image aspect ratio, as in ``edit1_Callback``."""
        return cls(float(width_um), float(width_um) / frame.aspect_ratio)

    @classmethod
    def from_height(cls, height_um: float, frame: ImageFrame) -> "LengthScale":
        """Width follows the image aspect ratio, as in ``edit2_Callback``."""
        return cls(float(height_um) * frame.aspect_ratio, float(height_um))

    @property
    def ld(self) -> float:
        """``min(Lx, Ly)/2``: um per ndim unit in the display frame."""
        return min(self.x_um, self.y_um) / 2.0

    @property
    def index_length_um(self) -> float:
        """``Ly/2``: the 'LengthScale' every legacy index routine multiplies by."""
        return self.y_um / 2.0

    @property
    def tissue_area_um2(self) -> float:
        return self.x_um * self.y_um

    @property
    def nondim_tissue_area(self) -> float:
        """``4 * Lx / Ly`` (GetCapillaryDomainsStatistics)."""
        return 4.0 * self.x_um / self.y_um

    @property
    def area_factor(self) -> float:
        """um^2 per ndim^2 (``DimAreaConversionFactor`` = (Ly/2)^2)."""
        return self.tissue_area_um2 / self.nondim_tissue_area


@dataclass(frozen=True)
class RoiNdim:
    """ROI descriptors in the ndim frame, as built in IndicesOrFEMSimulation."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def center(self) -> np.ndarray:
        # mean([minX, maxY; maxX, minY])
        return np.array([(self.min_x + self.max_x) / 2.0, (self.max_y + self.min_y) / 2.0])

    @property
    def xrange(self) -> float:
        return self.max_x - self.min_x

    @property
    def yrange(self) -> float:
        return self.max_y - self.min_y

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(lb, bb, rb, ub) re-derived from centre/range exactly like
        GetCapillaryDomainsInROI / GetFibersInROI."""
        c = self.center
        return (c[0] - self.xrange / 2.0, c[1] - self.yrange / 2.0,
                c[0] + self.xrange / 2.0, c[1] + self.yrange / 2.0)


@dataclass(frozen=True)
class ROI:
    """Region of interest in micrometres (display frame, origin bottom-left)."""

    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float

    @classmethod
    def from_imrect(cls, pos: Sequence[float]) -> "ROI":
        """``pos = [xmin, ymin, width, height]`` as returned by ``imrect``."""
        x, y, w, h = map(float, pos)
        return cls(x, x + w, y, y + h)

    @classmethod
    def default(cls, ls: LengthScale) -> "ROI":
        """GUI default rectangle: 20 %..80 % of each side."""
        return cls.from_imrect([0.2 * ls.x_um, 0.2 * ls.y_um, 0.6 * ls.x_um, 0.6 * ls.y_um])

    def to_ndim(self, ls: LengthScale) -> RoiNdim:
        """``(X - L/2) / (min(Lx,Ly)/2)``; GUIDE names X1/X2/Y1(=max)/Y2(=min)."""
        sx, sy = ls.x_um / 2.0, ls.y_um / 2.0
        scale = min(sx, sy)
        return RoiNdim(min_x=(self.x_min_um - sx) / scale, max_x=(self.x_max_um - sx) / scale,
                       min_y=(self.y_min_um - sy) / scale, max_y=(self.y_max_um - sy) / scale)


# --------------------------------------------------------------------------
# Geometry containers
# --------------------------------------------------------------------------
@dataclass
class IngestResult:
    """Output of :func:`otm_core.geometry.ingest_dtect` (TissueGeometryType)."""

    frame: ImageFrame
    capillaries_px_raw: np.ndarray
    capillaries_pxc_centred: np.ndarray
    capillaries_pxc_corrected: np.ndarray
    capillaries_pxc: np.ndarray            # final: corrected + de-duplicated
    fibers_pxc: List[Ring]
    fiber_types: np.ndarray


@dataclass
class RetouchSettings:
    """RetouchFibers defaults (OpeningFcn values, not the .fig labels)."""

    smooth_tol: int = 5            # moving-average window (px, odd)
    reduce_tol: float = 0.5        # Douglas-Peucker tolerance (px)
    tangent_tol: float = 3.0       # px
    smooth: bool = True            # uipanel1 'Yes'
    reduce: bool = True            # uipanel2 'Yes'
    tangent: bool = False          # uipanel3 'Keep'
    disjoint: bool = False         # uipanel4 'Keep'
    max_vertices: int = 50         # ReduceFiberVertices loop limit
    reduce_tol_step: float = 0.5   # extra tolerance per iteration


@dataclass
class RetouchResult:
    smooth: List[Ring]             # px_c
    reduced: List[Ring]
    disjoint: List[Ring]
    tangent: List[Ring]
    rescaled: List[Ring]           # ndim (Geometry.Fibers)
    capillaries_ndim: np.ndarray


@dataclass
class Geometry:
    """Equivalent of the legacy ``handles.Geometry`` struct (ndim frame)."""

    frame: ImageFrame
    capillaries: np.ndarray                                   # (N, 2) ndim
    fibers: List[FiberLike] = field(default_factory=list)     # empty for cardiac tissue
    fiber_types: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    length_scale: Optional[LengthScale] = None
    roi: Optional[ROI] = None

    @property
    def is_skeletal(self) -> bool:
        return len(self.fibers) > 0

    def roi_ndim(self) -> RoiNdim:
        if self.length_scale is None or self.roi is None:
            raise ValueError("length_scale and roi must be set first")
        return self.roi.to_ndim(self.length_scale)


# --------------------------------------------------------------------------
# Statistics containers
# --------------------------------------------------------------------------
def _std(v) -> float:
    """MATLAB ``std``: ddof=1, 0 for a single value, NaN for none."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        return float("nan")
    if v.size == 1:
        return 0.0
    return float(np.std(v, ddof=1))


@dataclass
class SummaryStats:
    """LIST/MEAN/STD/SEM block used by every legacy index."""

    values: np.ndarray
    mean: float
    std: float
    sem: float

    @classmethod
    def of(cls, values) -> "SummaryStats":
        v = np.asarray(values, dtype=float).ravel()
        mean = float(np.mean(v)) if v.size else float("nan")
        std = _std(v)
        sem = std / np.sqrt(v.size) if v.size else float("nan")
        return cls(v, mean, std, float(sem))

    @property
    def n(self) -> int:
        return int(self.values.size)


@dataclass
class VoronoiCells:
    """Capillary domains. ``cells[i]`` belongs to capillary ``i``."""

    cells: np.ndarray            # object array of shapely Polygons (unbounded cells clipped)
    bounded: np.ndarray          # bool, False where Qhull's cell is unbounded (legacy: Inf vertex)
    areas_ndim: np.ndarray       # NaN for unbounded cells (legacy polyarea with Inf)
    extent: Tuple[float, float, float, float]


@dataclass
class OverlapTable:
    """Sparse fibre x capillary-domain overlap (COO), built once with STRtree."""

    fiber: np.ndarray            # fibre index (0-based, into the full fibre list)
    cell: np.ndarray             # Voronoi cell / capillary index (0-based)
    area_ndim: np.ndarray        # |F ∩ V| in ndim^2
    max_vertex_distance_ndim: np.ndarray  # max dist from cell's capillary to vertices of F ∩ V
    n_fibers: int
    n_cells: int

    def to_sparse(self, values: str = "area_ndim"):
        """CSR matrix (n_fibers x n_cells) of ``values``."""
        from scipy.sparse import coo_matrix
        return coo_matrix((getattr(self, values), (self.fiber, self.cell)),
                          shape=(self.n_fibers, self.n_cells)).tocsr()

    def cells_of(self, fiber: int) -> np.ndarray:
        return self.cell[self.fiber == fiber]

    def fibers_of(self, cell: int) -> np.ndarray:
        return self.fiber[self.cell == cell]


@dataclass
class CapillaryDomainStats:
    """GetCapillaryDomainsStatistics."""

    voronoi: VoronoiCells
    x_length_um: float
    y_length_um: float
    tissue_area_um2: float
    nondim_tissue_area: float
    roi_area_um2: float
    total_num_capillaries: int
    capillary_density_per_mm2: float
    roi_capillary_density_per_mm2: float
    roi_index: np.ndarray                    # 0-based capillary/cell ids in the ROI
    capillary_domain_areas_um2: np.ndarray   # all cells (NaN if unbounded)
    roi_capillary_domain_areas_um2: np.ndarray
    domain_equivalent_diameter_um: np.ndarray
    area_stats: SummaryStats
    diameter_stats: SummaryStats
    roi_box_um: np.ndarray                   # 2 x 5, legacy ``ROI`` field (um_c frame)

    @property
    def n_domains_in_roi(self) -> int:
        return int(self.roi_index.size)


@dataclass
class NearestNeighborStats:
    """GetNearestNeighborStatistics (distances in um)."""

    neighbours: List[np.ndarray]     # 0-based Delaunay neighbours of each ROI capillary
    distances: List[np.ndarray]
    total: np.ndarray
    means: np.ndarray
    mins: np.ndarray
    stds: np.ndarray
    sems: np.ndarray
    all_distances: np.ndarray
    random_pairs: np.ndarray         # (n, 3): [cap, partner, distance], -1 = none (stochastic)
    unique_min_pairs: np.ndarray     # (n, 3)                                    (stochastic)
    summary: dict

    @property
    def rnd_list(self) -> np.ndarray:
        return self.random_pairs[:, 2]

    @property
    def unique_min_list(self) -> np.ndarray:
        return self.unique_min_pairs[:, 2]


@dataclass
class SupplyIndices:
    """GetSupplyIndices (skeletal muscle only)."""

    roi_fiber_index: np.ndarray      # 0-based
    fiber_area_um2: np.ndarray
    LCFR: SummaryStats
    LCD: SummaryStats
    DFR: SummaryStats
    FDR: SummaryStats
    Dmax: SummaryStats
    SF: SummaryStats
    CC: SummaryStats
    CFi: SummaryStats
    FPi: SummaryStats
    CFPE: SummaryStats
    number_of_fibers: int
    number_of_capillaries: int
    unbounded_overlaps: int          # overlaps with unbounded cells (legacy would yield NaN)

    @property
    def capillary_to_fiber_ratio(self) -> float:
        return (self.number_of_capillaries / self.number_of_fibers
                if self.number_of_fibers else float("nan"))


@dataclass
class Morphometrics:
    """GetMorphometricData."""

    domains: CapillaryDomainStats
    nearest_neighbours: NearestNeighborStats
    supply: Optional[SupplyIndices]
    overlaps: Optional[OverlapTable]
    capillaries: np.ndarray
    warnings: List[str] = field(default_factory=list)
