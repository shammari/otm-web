"""File input/output: Dtect ``.mat`` exports and legacy parameter ``.dat`` files.

* :func:`load_dtect_mat` reads the segmentation exported by Dtect (``Xcap``,
  ``Fibers``, ``FiberTypes``, ``ImageSize``). MATLAB v7.3 files are HDF5 and need
  ``h5py``; older v5/v7 files are read with ``scipy.io.loadmat``.
* :func:`read_parameters` / :func:`write_parameters` handle the 10-value
  ``*RestingParameters.dat`` files of the BiophysicalParameters panel
  (one value per line, the order of :class:`otm_core.fem.TransportParameters`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

__all__ = ["DtectData", "load_dtect_mat", "read_parameters", "write_parameters", "is_hdf5_mat"]


@dataclass
class DtectData:
    """Raw Dtect export, pixel frame, exactly as stored."""

    capillaries_px: np.ndarray               # (N, 2) Xcap (x, y)
    fibers_px: List[np.ndarray]              # rings (M_i, 2) with columns Fibers(i).x, Fibers(i).y
    fiber_types: Optional[np.ndarray]        # (F,) codes 1 / 21 / 22 / 0, or None
    image_size: tuple                        # MATLAB ImageSize = (rows, cols)
    path: str = ""
    notes: List[str] = field(default_factory=list)
    # data check (otm_core.qc): numbering in the original file once items are removed,
    # outlines that could not be read, and a fibre-type count that did not match
    fiber_ids: Optional[np.ndarray] = None       # (F,) 1-based numbers in the file
    capillary_ids: Optional[np.ndarray] = None   # (N,) 1-based numbers in the file
    fiber_read_problems: Optional[List[str]] = None
    type_count_mismatch: Optional[tuple] = None  # (number of types, number of fibres)
    removed_fibres: List[Any] = field(default_factory=list)   # (fibre id, ring) removed by the data check

    @property
    def has_fibres(self) -> bool:
        return len(self.fibers_px) > 0


def is_hdf5_mat(path: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(128)
    return b"MATLAB 7.3" in head or head[:4] == b"\x89HDF"


def _points(a: Any) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.size == 2 else a.reshape(-1, 2)
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    return a


def _vec(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _load_h5(path: str) -> DtectData:
    try:
        import h5py
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError("reading MATLAB v7.3 files needs h5py (pip install h5py)") from exc
    notes: List[str] = []
    with h5py.File(path, "r") as f:
        if "Xcap" not in f or "ImageSize" not in f:
            raise ValueError(f"{os.path.basename(path)} has no Xcap/ImageSize: not a Dtect export")
        caps = _points(np.asarray(f["Xcap"][()]).T)            # HDF5 stores the transpose
        image_size = tuple(int(v) for v in _vec(f["ImageSize"][()])[:2])
        fibres: List[np.ndarray] = []
        problems: List[str] = []
        if "Fibers" in f and isinstance(f["Fibers"], h5py.Group) and "x" in f["Fibers"]:
            g = f["Fibers"]
            xr, yr = np.asarray(g["x"][()]).ravel(), np.asarray(g["y"][()]).ravel()
            for rx, ry in zip(xr, yr):
                x, y = _vec(f[rx][()]), _vec(f[ry][()])
                fibres.append(_ring(x, y, problems))
        types = None
        if "FiberTypes" in f:
            t = _vec(f["FiberTypes"][()])
            types = t.astype(int) if t.size else None
    return DtectData(caps, fibres, types, image_size, path, notes, fiber_read_problems=problems)


def _ring(x: np.ndarray, y: np.ndarray, problems: List[str]) -> np.ndarray:
    """One outline; a broken one is kept (empty) and its problem recorded, so
    the data check can name it instead of the whole file failing."""
    if x.size != y.size:
        problems.append(f"x and y lists of different lengths ({x.size} and {y.size})")
        return np.zeros((0, 2))
    problems.append("")
    return np.column_stack([x, y])


def _load_v5(path: str) -> DtectData:
    from scipy.io import loadmat

    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Xcap" not in d or "ImageSize" not in d:
        raise ValueError(f"{os.path.basename(path)} has no Xcap/ImageSize: not a Dtect export")
    caps = _points(d["Xcap"])
    image_size = tuple(int(v) for v in _vec(d["ImageSize"])[:2])
    fibres: List[np.ndarray] = []
    problems: List[str] = []
    F = d.get("Fibers")
    if F is not None:
        for s in np.atleast_1d(F):
            fibres.append(_ring(_vec(getattr(s, "x", [])), _vec(getattr(s, "y", [])), problems))
    t = d.get("FiberTypes")
    types = _vec(t).astype(int) if t is not None and np.size(t) else None
    return DtectData(caps, fibres, types, image_size, path, [], fiber_read_problems=problems)


def load_dtect_mat(path: str) -> DtectData:
    """Read a Dtect ``.mat`` export (MATLAB v5/v7 or v7.3).

    The result feeds :func:`otm_core.geometry.ingest_dtect`::

        d = load_dtect_mat(path)
        ing = ingest_dtect(d.capillaries_px, d.fibers_px, d.image_size, d.fiber_types)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = _load_h5(path) if is_hdf5_mat(path) else _load_v5(path)
    if data.fiber_types is not None and len(data.fiber_types) != len(data.fibers_px):
        data.notes.append(f"{len(data.fiber_types)} fibre types for {len(data.fibers_px)} fibres: types ignored")
        data.type_count_mismatch = (len(data.fiber_types), len(data.fibers_px))
        data.fiber_types = None
    data.fiber_ids = np.arange(1, len(data.fibers_px) + 1)
    data.capillary_ids = np.arange(1, len(data.capillaries_px) + 1)
    return data


def read_parameters(path: str):
    """A ``*.dat``/``*.csv`` parameter file -> :class:`~otm_core.fem.TransportParameters`."""
    from .fem.solve import TransportParameters

    return TransportParameters.from_dat(path)


def write_parameters(params, path: str) -> None:
    """Write the 10 values one per line (the legacy ``.dat`` layout)."""
    from dataclasses import astuple

    with open(path, "w", encoding="utf-8") as fh:
        for v in astuple(params):
            fh.write(f"{float(v):.10g}\n")
