"""Refresh the generated parts of the OTM Web site.

    python tools/build_site.py                       # rewrite app/python-files.json
    python tools/build_site.py --otm-core "<OTM folder>/otm_core"   # copy otm_core in first

``app/python-files.json`` lists the Python files the browser loads (otm_core and
app/otm_web.py). Run this after adding or removing a Python file; editing a file
needs nothing. ``--otm-core`` copies the package from the desktop app's folder
(tests and caches left out), so both run the same code.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_otm_core(src: Path) -> None:
    dest = ROOT / "otm_core"
    if not (src / "__init__.py").is_file():
        raise SystemExit(f"{src} is not the otm_core package")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    print(f"copied otm_core from {src}")


def write_manifest() -> None:
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "otm_core").rglob("*.py"))
    files.append("app/otm_web.py")
    (ROOT / "app" / "python-files.json").write_text(json.dumps({"files": files}, indent=1) + "\n", encoding="utf-8")
    print(f"app/python-files.json: {len(files)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--otm-core", type=Path, help="otm_core folder of the desktop app to copy in first")
    a = ap.parse_args()
    if a.otm_core:
        copy_otm_core(a.otm_core)
    write_manifest()


if __name__ == "__main__":
    main()
