"""Command line: ``python -m otm_core run <files> --width 440 ...``

Examples::

    # every Dtect export of a folder, 440 µm wide, resting, with all exports
    python -m otm_core run "Sample Data/Test Data 2" --width 440

    # several folders, sizes per file from a table, indices only, 4 processes
    python -m otm_core run "Sample Data/**/*_export_cap_*.mat" --dims sizes.csv --steps indices --jobs 4

    # repeat a run exactly
    python -m otm_core run "Sample Data/Test Data 2" --settings OTM_batch_settings.json

    # write a settings file to edit (mesh, flux, retouch, parameters ...)
    python -m otm_core settings my_settings.json

Exit status: 0 when every file succeeded, 1 when at least one had an error,
2 for bad arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence

from .pipeline import (SETTINGS_FILE, STEPS, SUMMARY_FILE, RunResult, RunSettings, find_data_files,
                       read_dimensions_table, run_batch)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m otm_core", description="Oxygen Transport Modeller without the window.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="analyse one or more Dtect exports (.mat)",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description="Load, retouch, size, ROI, supply indices, mesh, PO2, flux lines and exports for "
                                   "each file; one summary table for all of them.")
    r.add_argument("inputs", nargs="+", help="files, folders (their *.mat) or wildcards (** = any subfolder)")
    r.add_argument("--settings", help="settings JSON (from a previous run or 'settings'); options below override it")
    size = r.add_mutually_exclusive_group()
    size.add_argument("--width", type=float, help="tissue width in µm (height follows the image)")
    size.add_argument("--height", type=float, help="tissue height in µm (width follows the image)")
    r.add_argument("--dims", help="CSV with per-file sizes: file, width_um or height_um [, roi_x_min_um, "
                                  "roi_x_max_um, roi_y_min_um, roi_y_max_um]")
    r.add_argument("--roi", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="ROI in µm (default: 20-80 %% of each side)")
    r.add_argument("--tissue", choices=("skeletal", "cardiac"))
    r.add_argument("--no-fibre-types", action="store_true", help="ignore the fibre types in the file")
    r.add_argument("--exercise", choices=("resting", "low", "moderate", "high"))
    r.add_argument("--params", help="biophysical parameters (.dat, legacy layout)")
    r.add_argument("--steps", help=f"comma-separated subset of {','.join(STEPS)} (default: all)")
    r.add_argument("--h-max", type=float, help="largest mesh element (model units)")
    r.add_argument("--seed", type=int, help="random seed of the nearest-neighbour pairings")
    r.add_argument("--no-export", action="store_true", help="summary table only")
    r.add_argument("--no-figures", action="store_true", help="text and spreadsheets only")
    r.add_argument("--dpi", type=int, help="figure resolution (default 600 publication, 300 legacy)")
    r.add_argument("--figure-style", choices=("publication", "legacy"),
                   help="figure layout: journal-ready (default) or the legacy MATLAB look")
    r.add_argument("--figure-width", choices=("single", "onehalf", "double"),
                   help="journal width of the figures: 85, 114 or 175 mm (default double)")
    r.add_argument("--formats", help="figure files to write, comma separated: png,pdf,svg,tiff (default png,pdf)")
    r.add_argument("--colormap", help="colour map of the PO2 profile (default turbo; legacy: jet)")
    r.add_argument("--out", help="write exports to OUT/<file name>/INDICES and PO2 (default: next to each file)")
    r.add_argument("--summary", help=f"summary file stem (default: ./{SUMMARY_FILE}_<date-time>)")
    r.add_argument("--save-session", action="store_true",
                   help="also save each analysis as <name>.otm (open it in the desktop app: File > Open session)")
    r.add_argument("--apply-edits", action="store_true",
                   help="apply the manual corrections saved by the desktop app (<name>.otm next to each data file)")
    r.add_argument("--jobs", type=int, default=1, help="files analysed in parallel (processes)")
    r.add_argument("--quiet", action="store_true")

    st = sub.add_parser("settings", help="write the default settings as JSON (to edit and pass with --settings)")
    st.add_argument("path", nargs="?", default=SETTINGS_FILE)
    st.add_argument("--tissue", choices=("skeletal", "cardiac"), default="skeletal")
    return p


def _settings(a: argparse.Namespace) -> RunSettings:
    s = RunSettings.load(a.settings) if a.settings else RunSettings()
    ch = {}
    if a.width is not None:
        ch.update(width_um=a.width, height_um=None)
    if a.height is not None:
        ch.update(height_um=a.height, width_um=None)
    if a.roi is not None:
        ch["roi_um"] = tuple(a.roi)
    if a.tissue:
        ch["tissue"] = a.tissue
    if a.no_fibre_types:
        ch["use_fibre_types"] = False
    if a.exercise:
        ch["exercise"] = a.exercise
    if a.params:
        ch["parameter_file"] = os.path.abspath(a.params)
    if a.steps:
        ch["steps"] = tuple(x.strip() for x in a.steps.split(",") if x.strip())
    if a.h_max is not None:
        import dataclasses

        ch["mesh"] = dataclasses.replace(s.mesh, h_max=a.h_max)
    if a.seed is not None:
        ch["index_seed"] = a.seed
    if a.no_export:
        ch["export"] = False
    if a.no_figures:
        ch["figures"] = False
    if a.dpi is not None:
        ch["dpi"] = a.dpi
    if a.figure_style:
        ch["figure_layout"] = a.figure_style
    if a.figure_width:
        ch["figure_width"] = a.figure_width
    if a.formats:
        ch["figure_formats"] = tuple(f.strip() for f in a.formats.split(",") if f.strip())
    if a.colormap:
        ch["colormap"] = a.colormap
    return s.with_changes(**ch) if ch else s


def _line(i: int, n: int, r: RunResult) -> str:
    row = r.summary()
    secs = sum(r.timings.values())
    head = f"[{i + 1}/{n}] {os.path.basename(r.path)}  ({secs:.0f} s)"
    if not r.ok:
        return head + "\n    " + "\n    ".join(f"{k} failed: {v}" for k, v in r.errors.items())
    parts = []
    if row.get("LCFR_mean") == row.get("LCFR_mean"):                      # not NaN
        parts.append(f"LCFR {row['LCFR_mean']:.2f}, CD {row['capillary_density_per_mm2']:.0f}/mm²")
    if "po2_tissue_mean_mmHg" in row:
        parts.append(f"PO2 {row['po2_tissue_mean_mmHg']:.2f} mmHg ({row['po2_tissue_hypoxic_pct']:.1f} % hypoxic)")
    if "flux_capillaries_with_lines" in row:
        parts.append(f"flux lines from {row['flux_capillaries_with_lines']} capillaries")
    if r.edits:
        parts.append(f"{len(r.edits)} manual corrections" + (f" ({len(r.edit_warnings)} skipped)"
                                                             if r.edit_warnings else ""))
    if r.files:
        parts.append(f"{len(r.files)} files")
    return head + "\n    " + "; ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = _parser().parse_args(argv)
    if a.command == "settings":
        RunSettings(tissue=a.tissue).save(a.path)
        print(f"Default settings written to {a.path}")
        return 0

    try:
        s = _settings(a)
        dims = read_dimensions_table(a.dims) if a.dims else None
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = find_data_files(a.inputs)
    if not paths:
        print("error: no input files found", file=sys.stderr)
        return 2
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("error: not found: " + ", ".join(missing), file=sys.stderr)
        return 2
    stem = a.summary or os.path.join(os.getcwd(), f"{SUMMARY_FILE}_{time.strftime('%Y%m%d-%H%M%S')}")
    if stem.lower().endswith((".xlsx", ".csv")):
        stem = os.path.splitext(stem)[0]
    say = (lambda *x, **k: None) if a.quiet else print
    say(f"{len(paths)} file(s); steps: {', '.join(s.steps)}; exercise: {s.exercise}; jobs: {a.jobs}")
    t0 = time.perf_counter()
    results, written = run_batch(paths, s, a.out, stem, dims, jobs=max(1, a.jobs), save_sessions=a.save_session, apply_saved_edits=a.apply_edits,
                                 on_result=lambda i, n, r: say(_line(i, n, r), flush=True))
    bad = [r for r in results if not r.ok]
    say(f"\n{len(results) - len(bad)} of {len(results)} file(s) OK in {time.perf_counter() - t0:.0f} s")
    for f in written:
        say(f"  {f}")
    return 1 if bad else 0


if __name__ == "__main__":                                                   # pragma: no cover
    sys.exit(main())
