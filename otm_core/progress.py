"""Progress reporting hook shared by the long-running core routines.

Long computations (meshing, the Newton solver, flux lines) accept an optional
``progress`` callable::

    progress(fraction: float, message: str) -> None

``fraction`` runs from 0 to 1 and never decreases within one call of the routine.
The core stays free of any GUI framework: a desktop or web front end passes a
callback that forwards the values to its own progress bar.

To cancel, the callback raises an exception (e.g. ``otm_desktop.workers.WorkerCancelled``).
The routine does not catch it: it cleans up (``finally``) and the exception
reaches the caller. Cancellation therefore takes effect at the next report,
not in the middle of a single gmsh or sparse-solver call.
"""

from __future__ import annotations

from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]

__all__ = ["ProgressCallback", "report", "scaled"]


def report(progress: Optional[ProgressCallback], fraction: float, message: str) -> None:
    """Call ``progress`` (if any) with ``fraction`` clamped to [0, 1]."""
    if progress is not None:
        progress(min(1.0, max(0.0, float(fraction))), message)


def scaled(progress: Optional[ProgressCallback], start: float, stop: float) -> Optional[ProgressCallback]:
    """A callback mapping [0, 1] onto [start, stop] of ``progress`` (for sub-steps)."""
    if progress is None:
        return None

    def sub(fraction: float, message: str) -> None:
        report(progress, start + (stop - start) * fraction, message)

    return sub
