"""otm_core — framework-agnostic mathematical core of the Oxygen Transport Modeller.

Python port of the legacy MATLAB OTM Lite (2022) geometry and morphometry
pipeline (see ``matlab_reverse_engineering_report.md`` sections 3.3, 4.2, 6).
No GUI code lives here: the same functions back the PySide6 desktop app and
the Dash/Streamlit web app.

Typical use::

    import numpy as np
    from otm_core import (ingest_dtect, retouch_fibers, RetouchSettings, Geometry,
                          LengthScale, ROI, get_morphometric_data)

    ing = ingest_dtect(Xcap, Fibers, ImageSize, FiberTypes)           # TissueGeometryType
    ret = retouch_fibers(RetouchSettings(), ing.fibers_pxc,           # RetouchFibers
                         ing.capillaries_pxc, ing.frame)
    ls = LengthScale.from_width(440.0, ing.frame)                     # TissueDimensions
    geo = Geometry(ing.frame, ret.capillaries_ndim, ret.rescaled,
                   ing.fiber_types, ls, ROI.default(ls))              # ROI_MouseControlSelection
    m = get_morphometric_data(geo, rng=np.random.default_rng(0))      # GetMorphometricData
    m.supply.LCFR.mean, m.domains.area_stats.mean

Requirements: numpy, scipy, shapely >= 2.0.
"""

from .models import (
    FIBER_TYPE_I,
    FIBER_TYPE_IIA,
    FIBER_TYPE_IIB,
    FIBER_TYPE_UNKNOWN,
    CapillaryDomainStats,
    FiberLike,
    Geometry,
    ImageFrame,
    IngestResult,
    LengthScale,
    Morphometrics,
    NearestNeighborStats,
    OverlapTable,
    RetouchResult,
    RetouchSettings,
    Ring,
    ROI,
    RoiNdim,
    SummaryStats,
    SupplyIndices,
    VoronoiCells,
    from_matlab_index,
    to_matlab_index,
)
from .geometry import (
    as_ring,
    capillary_exclusion_zone,
    close_ring,
    correct_capillaries,
    douglas_peucker,
    dtect_to_centred,
    eliminate_duplicate_capillaries,
    enlarge_is_locally,
    fiber_area_um2,
    fiber_geometries,
    geometry_parts_as_rings,
    ingest_dtect,
    inpolygon,
    is_multipart,
    k_moving_average,
    k_window_smoothing,
    ndim_box,
    ndim_to_centred_um,
    ndim_to_um,
    neighbouring_fibers,
    polyarea,
    reduce_fiber_vertices,
    remove_capillary_fibre_overlap,
    retouch_fibers,
    scale_geometry,
    select_model_capillaries,
    select_model_fibers,
    signed_distance_to_polygon,
    single_is,
    um_to_ndim,
)
from .morphometry import (
    capillary_domain_statistics,
    delaunay_neighbours,
    get_morphometric_data,
    nearest_neighbour_statistics,
    overlap_table,
    select_in_roi,
    supply_indices,
    voronoi_cells,
)

__version__ = "0.1.0"
__all__ = [
    "CapillaryDomainStats",
    "FIBER_TYPE_I",
    "FIBER_TYPE_IIA",
    "FIBER_TYPE_IIB",
    "FIBER_TYPE_UNKNOWN",
    "FiberLike",
    "Geometry",
    "ImageFrame",
    "IngestResult",
    "LengthScale",
    "Morphometrics",
    "NearestNeighborStats",
    "OverlapTable",
    "ROI",
    "RetouchResult",
    "RetouchSettings",
    "Ring",
    "RoiNdim",
    "SummaryStats",
    "SupplyIndices",
    "VoronoiCells",
    "as_ring",
    "capillary_domain_statistics",
    "capillary_exclusion_zone",
    "close_ring",
    "correct_capillaries",
    "delaunay_neighbours",
    "douglas_peucker",
    "dtect_to_centred",
    "eliminate_duplicate_capillaries",
    "enlarge_is_locally",
    "fiber_area_um2",
    "fiber_geometries",
    "from_matlab_index",
    "geometry_parts_as_rings",
    "get_morphometric_data",
    "ingest_dtect",
    "inpolygon",
    "is_multipart",
    "k_moving_average",
    "k_window_smoothing",
    "ndim_box",
    "ndim_to_centred_um",
    "ndim_to_um",
    "nearest_neighbour_statistics",
    "neighbouring_fibers",
    "overlap_table",
    "polyarea",
    "reduce_fiber_vertices",
    "remove_capillary_fibre_overlap",
    "retouch_fibers",
    "scale_geometry",
    "select_in_roi",
    "select_model_capillaries",
    "select_model_fibers",
    "signed_distance_to_polygon",
    "single_is",
    "supply_indices",
    "to_matlab_index",
    "um_to_ndim",
    "voronoi_cells",
]
