"""otm_core.fem — finite-element PO2 model (legacy PDE-Toolbox replacement).

* :mod:`otm_core.fem.mesh`  — model geometry, Shapely planar partition, gmsh meshing
* :mod:`otm_core.fem.solve` — coefficients, P1 assembly (scikit-fem), Newton-Raphson,
  PO2/MO2/per-fibre statistics

Typical use::

    from otm_core.fem import (TransportParameters, ModelSwitches, mesh_tissue,
                              solve_po2, po2_statistics)

    params = TransportParameters.from_dat("SkeletalRestingParameters.dat") \\
        .derive(len(geo.capillaries), geo.length_scale)
    mesh, model = mesh_tissue(geo, params.Rcap)                 # gmsh
    sol = solve_po2(mesh, params, ModelSwitches.skeletal("moderate"))
    po2_statistics(sol)["Tissue"]

Requirements: gmsh (meshing only), scikit-fem, scipy, shapely.
"""

from .mesh import (
    COMPARTMENT_IS,
    IS_REGION,
    MeshSettings,
    ModelGeometry,
    RegionPlan,
    TissueMesh,
    build_tissue_mesh,
    mesh_tissue,
    plan_regions,
    prepare_model_geometry,
)
from .solve import (
    CompartmentCoefficients,
    DerivedParameters,
    ModelSwitches,
    NewtonSettings,
    PO2Solution,
    TransportParameters,
    compartment_coefficients,
    fiber_po2_statistics,
    matlab_f4,
    matlab_num2str,
    mo2_statistics,
    po2_statistics,
    solve_po2,
)

__all__ = [
    "COMPARTMENT_IS", "IS_REGION", "MeshSettings", "ModelGeometry", "RegionPlan", "TissueMesh",
    "build_tissue_mesh", "mesh_tissue", "plan_regions", "prepare_model_geometry",
    "CompartmentCoefficients", "DerivedParameters", "ModelSwitches", "NewtonSettings", "PO2Solution",
    "TransportParameters", "compartment_coefficients", "fiber_po2_statistics", "matlab_f4",
    "matlab_num2str", "mo2_statistics", "po2_statistics", "solve_po2",
]
