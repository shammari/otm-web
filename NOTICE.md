# Third-party software in OTM Web

OTM Web ships the following software. Each part keeps its own licence.

| Component | Where | Licence |
| --- | --- | --- |
| Triangle 1.6, by Jonathan Richard Shewchuk | `app/triangle.wasm`, `app/triangle-module.js` | Free for private, research and institutional use; **commercial use needs a licence from the author.** See <https://www.cs.cmu.edu/~quake/triangle.html>. |
| triangle-wasm 1.0.0, by Bruno Imbrizi (JavaScript wrapper) | `app/triangle.mjs`, `app/triangle-module.js` | MIT. The wasm's memory limit was raised for large meshes. |
| Pyodide 314.0.7 | `pyodide/` | MPL-2.0 |
| CPython 3.14 (standard library) | `pyodide/python_stdlib.zip` | PSF License |
| NumPy, SciPy, Shapely, h5py, contourpy, kiwisolver | `pyodide/*.whl` | BSD-3-Clause |
| Matplotlib | `pyodide/matplotlib-*.whl` | Matplotlib License (PSF-based) |
| Pillow | `pyodide/pillow-*.whl` | MIT-CMU (HPND) |
| fonttools, cycler, pyparsing, six, pytz, python-dateutil, packaging, pkgconfig, micropip | `pyodide/*.whl` | MIT / BSD / Apache-2.0 / PSF, as in each wheel |
| scikit-fem 12.0.2 | `wheels/` | BSD-3-Clause |
| openpyxl 3.1.5, et-xmlfile 2.0.0 | `wheels/` | MIT |

Each wheel's own licence file is inside the wheel, under `*.dist-info/`.
