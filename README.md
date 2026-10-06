# OTM Web

OTM Web models oxygen transport in muscle tissue, in a web browser. It runs the same Python code (`otm_core`) as the OTM Lite desktop app. It takes Dtect exports (`.mat`) and gives the supply indices, the PO₂ and O₂-consumption solution, flux lines, and the figures and spreadsheets the desktop app writes.

**Open it:** <https://shammari.github.io/otm-web/>

**Your files stay on your computer.** The analysis runs inside the browser tab, in Python compiled to WebAssembly ([Pyodide](https://pyodide.org)). Nothing is uploaded, and the results last until the tab is closed. Download the ZIP before closing it.

## Using it

1. **Files:** drop one or more `.mat` exports onto the page. Each file is checked on arrival, showing its image size, capillaries, fibres and fibre types. To apply corrections made in the desktop app, also add its `.otm` session with the same name.
2. **Settings:** tissue type, tissue width or height in µm, region of interest, biophysical parameters and exercise level, analysis steps, and figure style.
   - Sizes can be given per file in the file table, or loaded from a dimensions CSV.
   - Settings save and load as the same JSON the desktop batch runner (`python -m otm_core run --settings`) reads.
3. **Run:** samples run one after another, with progress for each. A whole sample with figures takes about 1.5 minutes.
4. **Results:** a row per sample, then for the selected sample its tables (hypoxic compartments in coral) and figures. **Download all results (ZIP)** holds, per sample, the `INDICES` and `PO2` folders exactly as the desktop app writes them, plus `OTM_batch_summary.xlsx`/`.csv` and the settings.

The first visit downloads about 45 MB (Python and its scientific libraries). The browser keeps it for later visits.

**Browsers:** current Chrome, Edge or Firefox on a computer. Safari 17+ should work but is not tested yet. Phones are not supported.

## Differences from the desktop app

- **Mesher:** the mesh comes from [Triangle](https://www.cs.cmu.edu/~quake/triangle.html), compiled to WebAssembly, in place of gmsh, which has no browser build. On the reference sample the results match MATLAB:
  - tissue mean PO₂ within 0.014 %
  - per-fibre PO₂ within 0.006 mmHg
  - LCFR identical
- **No editing:** corrections are made in the desktop app and brought in with its `.otm` file.
- **Figures:** they come out as files. The interactive PO₂ map is planned next.

## Publishing on GitHub Pages

The repository is the website; nothing needs building.

1. Push this folder to `https://github.com/shammari/otm-web`.
2. On GitHub, open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, branch `main`, folder `/ (root)`, and **Save**.
4. After a minute or two the site is at `https://shammari.github.io/otm-web/`.

`.nojekyll` must stay in the root: it stops GitHub from processing the site.

## Updating

- **otm_core changed on the desktop:** `python tools/build_site.py --otm-core "<OTM folder>/otm_core"` copies it in and refreshes `app/python-files.json`. Then commit and push.
- **Test before pushing:** `python tests/browser_test.py --sample <sample.mat> --baseline <baseline_small.json>`. It needs Playwright and runs the whole workflow in headless Chromium, checking the results against the MATLAB baseline.

## What is where

| Path | What |
| --- | --- |
| `index.html`, `app/app.js`, `app/style.css` | The page: files, settings, run, results |
| `app/worker.js` | Background worker that loads Pyodide, otm_core and Triangle and runs the analysis |
| `app/otm_web.py` | Python glue: file checks, runs, tables, the ZIP |
| `app/triangle.*` | Triangle mesher (WebAssembly, from `triangle-wasm`, memory limit raised) |
| `otm_core/` | The OTM analysis package, the same code as the desktop app |
| `pyodide/` | Pyodide 314.0.7 runtime and the packages OTM needs |
| `wheels/` | scikit-fem, openpyxl and et-xmlfile (pure-Python wheels) |
| `tools/build_site.py` | Copies otm_core in and lists the Python files the page loads |
| `tests/browser_test.py` | End-to-end browser test |

## Licences

See [NOTICE.md](NOTICE.md). **Triangle is free for non-commercial use only,** so OTM Web must not be used or offered commercially without a licence from its author.

