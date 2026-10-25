"""End-to-end test of the desktop-parity features of OTM Web in headless Chromium.

    python tests/features_test.py [--out tests/output_features]

Uses the sample bundled with the site (app/sample) and makes a flawed copy of it
(an outline around the whole section exported as a fibre: data-check code DT-F01,
the case a lab reported). Checks, in the page:

* Help: About text with the credits and contact, sample button, diagnostic report;
* data check on arrival, per policy (remove / skip / keep), and its report in the results;
* the corrections editor: tools, undo/redo, the edit list, session (.otm) from the
  editor, corrections used by the run;
* advanced settings (retouch, mesh, flux lines, ROI shares) in the settings JSON,
  which must stay readable by the desktop batch runner (RunSettings.from_dict);
* the tissue view: drawn, hover read-out, layers, fixed colour range and its
  coverage note, VO2 map, PNG export; the per-sample session (.otm).

Needs Playwright (``pip install playwright``) and, for the JSON check, numpy and scipy.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "app" / "sample" / "OTM_sample_Dtect.mat"
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".js": "text/javascript", ".wasm": "application/wasm"}

    def log_message(self, *a):
        pass


def flawed_copy(dst: Path) -> None:
    """The sample plus a 'fibre' drawn around the whole section (DT-F01)."""
    import scipy.io as sio

    m = sio.loadmat(SAMPLE)
    F = m["Fibers"]
    n = F.shape[1]
    H, W = m["ImageSize"].ravel()
    x = np.array([30, 30, H - 30, H - 30, 30], float)          # Dtect's .x runs along the rows
    y = np.array([30, W - 30, W - 30, 30, 30], float)
    G = np.empty((1, n + 1), dtype=F.dtype)
    for k in range(n):
        G[0, k] = F[0, k]
    G[0, n] = (x.reshape(-1, 1), y.reshape(-1, 1))
    sio.savemat(dst, {"Xcap": m["Xcap"], "Fibers": G, "FiberTypes": np.vstack([m["FiberTypes"], [[1.0]]]),
                      "ImageSize": m["ImageSize"]})


def shot(page, path: Path) -> None:
    """A screenshot for looking at afterwards; not part of the checks (software GL can stall it)."""
    try:
        page.screenshot(path=str(path), timeout=60_000)
    except Exception as exc:                          # noqa: BLE001
        print(f"    (screenshot {path.name} skipped: {type(exc).__name__})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "output_features")
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()
    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    flawed = out / "flawed.mat"
    flawed_copy(flawed)
    sys.path.insert(0, str(ROOT))
    from otm_core.pipeline import RunSettings                     # what the desktop batch runner reads

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(_Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    t_all = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = browser.new_context(viewport={"width": 1360, "height": 1000}, accept_downloads=True)
        page = ctx.new_page()
        page.on("pageerror", lambda e: FAILS.append(f"page error: {e}") or print("  page error:", e))
        page.on("console", lambda m: m.type == "error" and print("  console:", m.text[:300]))
        page.goto(f"http://127.0.0.1:{a.port}/index.html")
        page.wait_for_selector("#engine.ready", timeout=240_000)
        ev = page.evaluate

        print("1  help")
        page.click("#help-button")
        txt = page.inner_text("#help-dialog")
        check("Abdullah A. Al-Shammari" in txt and "Kuwait University" in txt and "Roger Kissane" in txt
              and "abdullah.alshammari@ku.edu.kw" in txt and "OTM2026" in txt, "About: author, collaborators, contact")
        check("otm_core 2026" in page.inner_text("#about-version"), "About shows the engine version")
        page.click("#help-sample")
        page.set_input_files("#file-input", [str(flawed)])
        page.wait_for_function("window.__otm.state.files.length == 2 && window.__otm.state.files.every(f => f.status === 'ok' || f.status === 'bad')",
                               timeout=120_000)
        check(page.input_value('#file-table tr[data-name="OTM_sample_Dtect"] input[aria-label^="width"]') == "440",
              "sample added with its width (440 µm)")

        print("2  data check")
        cells = {n: page.inner_text(f'#file-table tr[data-name="{n}"] .dc-cell') for n in ("OTM_sample_Dtect", "flawed")}
        check("passed" in cells["OTM_sample_Dtect"], "sample: data check passed")
        check("1 error" in cells["flawed"] and "will be removed" in cells["flawed"] and "DT-F01" in cells["flawed"],
              f"flawed: DT-F01, flagged items will be removed ({cells['flawed'].splitlines()[0]})")
        page.check('input[name="dcheck"][value="skip"]')
        check("will be skipped" in page.inner_text('#file-table tr[data-name="flawed"] .dc-cell'), "policy 'skip' shown on the row")
        page.check('input[name="dcheck"][value="keep"]')
        check("removed" in page.inner_text('#file-table tr[data-name="flawed"] .dc-cell'),
              "policy 'keep' cannot keep DT-F01 (not overridable): removed")
        page.check('input[name="dcheck"][value="skip"]')
        page.locator('#file-table tr[data-name="OTM_sample_Dtect"] button.apply-all').click()

        print("3  corrections editor")
        page.locator('#file-table tr[data-name="flawed"] button', has_text="Correct").click()
        page.wait_for_function("window.__otm.ed.last && window.__otm.ed.last.ok", timeout=60_000)
        last = ev("window.__otm.ed.last")
        check(len(last["fibres"]) == 90 and last["flags"]["fibres"] and last["flags"]["fibres"][0][1] == "DT-F01",
              "with 'skip' the editor shows the file as it is, the section outline flagged")
        check("not be analysed" in page.inner_text("#ed-check"), "editor says the file would be skipped")
        page.click("#ed-done")
        page.check('input[name="dcheck"][value="fix"]')
        page.locator('#file-table tr[data-name="flawed"] button', has_text="Correct").click()
        page.wait_for_function("window.__otm.ed.last && window.__otm.ed.last.fibres.length == 89", timeout=60_000)
        check("removed fibre 90" in page.inner_text("#ed-check"), "with 'remove' the outline is gone (removed fibre 90)")
        stage = page.locator("#ed-stage canvas").bounding_box()

        def edits() -> int:
            return len(ev("window.__otm.ed.last.edits_text"))

        page.select_option("#ed-type", "22")
        page.click('#ed-tools button[data-tool="fiber_type"]')
        k, sx, sy = ev("""(() => { const e = window.__otm.ed; const k = e.last.types.findIndex(t => t !== 22);
            const r = e.last.fibres[k]; let x = 0, y = 0; for (const p of r) { x += p[0]; y += p[1]; }
            const s = e.editor.toScreen(x / r.length, y / r.length); return [k, s[0], s[1]]; })()""")
        page.mouse.click(stage["x"] + sx, stage["y"] + sy)
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 1", timeout=30_000)
        check(ev(f"window.__otm.ed.last.types[{k}]") == 22, "Set fibre type: clicked fibre becomes IIb")
        n_caps = len(ev("window.__otm.ed.last.caps"))             # (with 'skip' the outline had moved some)
        page.click('#ed-tools button[data-tool="add_capillary"]')
        page.mouse.click(stage["x"] + 220, stage["y"] + 320)
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 2", timeout=30_000)
        check(len(ev("window.__otm.ed.last.caps")) == n_caps + 1, "Add capillary")
        page.click("#ed-undo")
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 1")
        check(len(ev("window.__otm.ed.last.caps")) == n_caps, "Undo removes it again")
        page.click("#ed-redo")
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 2")
        page.click('#ed-tools button[data-tool="draw_fiber"]')
        for dx, dy in ((40, 40), (90, 45), (60, 95)):
            page.mouse.click(stage["x"] + dx, stage["y"] + dy)
            time.sleep(0.15)
        page.mouse.dblclick(stage["x"] + 30, stage["y"] + 90)
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 3", timeout=30_000)
        check(len(ev("window.__otm.ed.last.fibres")) == 90 and ev("window.__otm.ed.last.edits_text[2]").startswith("Add fibre"),
              "Draw fibre (clicks + double-click) adds an outline")
        page.click('#ed-tools button[data-tool="remove_capillary"]')
        cx, cy = ev("(() => { const e = window.__otm.ed; const c = e.last.caps[0]; return e.editor.toScreen(c[0], c[1]); })()")
        page.mouse.click(stage["x"] + cx, stage["y"] + cy)
        page.wait_for_function("window.__otm.ed.last.edits_text.length == 4", timeout=30_000)
        check(len(ev("window.__otm.ed.last.caps")) == n_caps, "Remove capillary")
        check(page.locator("#ed-list li").count() == 4, "edit list has 4 lines")
        shot(page, out / "editor.png")
        with page.expect_download() as d:
            page.click("#ed-save-otm")
        otm_path = out / "flawed_from_editor.otm"
        d.value.save_as(otm_path)
        with zipfile.ZipFile(otm_path) as z:
            meta = json.loads(z.read("session.json"))
        check(len(meta.get("edits") or []) == 4, "session (.otm) from the editor holds the 4 corrections")
        page.click("#ed-done")
        page.wait_for_function("document.querySelector('#file-table tr[data-name=\"flawed\"] .dc-cell').innerText.includes('4 corrections made here')",
                               timeout=10_000)
        check(True, "file row: 4 corrections made here")

        print("4  advanced settings and the settings JSON")
        page.click("#advanced summary")
        page.fill("#flux-lines", "4")
        page.dispatch_event("#flux-lines", "change")
        page.select_option("#mesh-overlap", "larger")
        page.dispatch_event("#mesh-overlap", "change")
        page.uncheck("#rt-reduce")
        page.dispatch_event("#rt-reduce", "change")
        page.locator('label:has(input[name="roi"][value="shares"])').click()
        page.fill("#roi-fx0", "10")
        page.dispatch_event("#roi-fx0", "input")
        check(page.locator("#advanced.changed").count() == 1, "advanced settings marked as changed")
        check(page.locator("#advanced input:invalid").count() == 0, "no advanced field flagged invalid with the defaults")
        with page.expect_download() as d:
            page.click("#save-settings")
        sj = json.loads(Path(d.value.path()).read_text())
        try:
            rs = RunSettings.from_dict(sj)
            check(rs.flux.lines_per_capillary == 4 and rs.mesh.overlap_rule == "larger" and rs.retouch.reduce is False
                  and rs.roi_fraction == (0.1, 0.8, 0.2, 0.8) and rs.data_check == "fix",
                  "settings JSON read by the desktop batch runner with the advanced values")
        except Exception as exc:                     # noqa: BLE001
            check(False, f"settings JSON rejected by RunSettings: {exc}")
        page.select_option("#flux-cap", "fixed")
        page.dispatch_event("#flux-cap", "change")
        page.fill("#flux-cap-um", "33")
        page.dispatch_event("#flux-cap-um", "change")
        with page.expect_download() as d:
            page.click("#save-settings")
        ml = json.loads(Path(d.value.path()).read_text())["flux"]["max_length"]
        check(abs(ml - 33 / 165) < 1e-9, f"fixed cap 33 µm saved in model units ({ml:.4f} = 33 / (330/2))")
        page.select_option("#flux-cap", "auto")
        page.dispatch_event("#flux-cap", "change")
        page.uncheck("#figures")

        print("4b oxygen demand Λ and parallel samples")
        page.fill("#p-M0", "4.71e-4")                         # 3 x the resting M0, exercise level Resting
        page.dispatch_event("#p-M0", "input")
        est = page.inner_text("#dm-estimate")
        check("3.00 × rest" in est and "non-resting" in est and "switched on" in est, f"Λ estimate in the settings ({est})")
        page.uncheck("#dm-enabled")
        page.dispatch_event("#dm-enabled", "input")
        check("only reported" in page.inner_text("#dm-estimate") and page.is_disabled("#dm-uptake"), "legacy switch")
        page.check("#dm-enabled")
        page.dispatch_event("#dm-enabled", "input")
        with page.expect_download() as d:
            page.click("#save-settings")
        sj = json.loads(Path(d.value.path()).read_text())
        check(sj.get("demand") == {"enabled": True, "cutoff": 1.5, "switch_uptake": True, "scale_flux_cap": True}
              and RunSettings.from_dict(sj).demand.enabled, "demand settings in the JSON, read by the batch runner")
        page.select_option("#parallel", "2")
        page.dispatch_event("#parallel", "change")

        print("5  run")
        page.click("#run-button")
        page.wait_for_function("!window.__otm.state.running && window.__otm.state.results.size == 2", timeout=900_000)
        res = ev("Object.fromEntries(window.__otm.state.results)")
        check(all(r["ok"] for r in res.values()), "both samples ran without errors")
        fl = res["flawed"]
        check(fl["corrections"] == 4 and fl["corrections_from"] == "page", "flawed ran with the 4 corrections made here")
        check(fl["check"]["fixed"] and fl["check"]["fixed"][0]["code"] == "DT-F01", "run's data check removed DT-F01")
        check(any(f.endswith("_data_check.txt") for f in fl["files"]), "data-check report written with the results")
        check(all(3 * r["flux"]["seed_capillaries"] < r["flux"]["moving_lines"] <= 4 * r["flux"]["seed_capillaries"]
                  for r in res.values()), "4 flux lines per capillary (" + ", ".join(
                      f"{r['flux']['moving_lines']} moving from {r['flux']['seed_capillaries']}" for r in res.values()) + ")")
        check(all(any(f.endswith(".otm") for f in r["files"]) for r in res.values()), "a session (.otm) per sample")
        for n, r in res.items():
            print(f"    {n}: {r['seconds']:.0f} s on the {r.get('worker')} worker, view {r['view'] and r['view'].get('n_nodes')} nodes")
        check(sorted(r.get("worker") for r in res.values()) == ["extra", "main"], "the two samples ran in two workers at once")
        dm = [r["demand"] for r in res.values()]
        check(all(abs(x["lambda_ratio"] - 3) < 1e-6 and x["uptake_switched_by_lambda"] and x["demand_regime"] == "non-resting"
                  for x in dm), "Λ = 3 × rest at Resting: MM and Mb switched on in both runs")
        check(all(abs(r["flux"]["cap_um"] - 30.2 / 3 ** 0.5) < 1.5 for r in res.values()),
              "automatic flux-line cap shortened by √(1/3) (" + ", ".join(f"{r['flux']['cap_um']:.1f} µm" for r in res.values()) + ")")
        check(all(any(f.endswith("_Lambda.csv") for f in r["files"]) for r in res.values()), "Λ table per fibre written")
        lam_cells = [page.inner_text(f'#overview tr[data-name="{n}"] td:nth-child(5)') for n in res]
        check(lam_cells == ["3.00", "3.00"], f"Λ × rest column in the overview ({lam_cells})")

        print("6  tissue view")
        page.locator('#overview tr[data-name="flawed"]').click()
        page.wait_for_function("window.__otm.view && window.__otm.view.data", timeout=60_000)
        check("removed" in page.inner_text(".dc-result"), "results show what the data check did")
        page.locator(".block-view").scroll_into_view_if_needed()
        time.sleep(0.8)
        st = page.locator(".tv-stage").bounding_box()
        import base64
        from io import BytesIO

        try:
            from PIL import Image

            url = ev("document.querySelector('.tv-gl').toDataURL('image/png')")       # the map itself
            img = np.asarray(Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")).astype(int)
            colours = len(np.unique(img[::4, ::4].reshape(-1, 3) // 16, axis=0))
            check(colours > 40, f"PO2 map drawn ({colours} colour bins)")
        except ImportError:
            print("    (Pillow missing: map pixels not checked)")
        page.mouse.move(st["x"] + st["width"] / 2, st["y"] + st["height"] / 2)
        time.sleep(0.3)
        ro = page.inner_text(".tv-readout")
        check("µm" in ro and "PO₂" in ro and "mmHg" in ro, f"hover read-out ({ro})")
        page.select_option(".tv-units", "ndim")
        page.mouse.move(st["x"] + st["width"] / 2 + 1, st["y"] + st["height"] / 2)
        time.sleep(0.3)
        check("µm" not in page.inner_text(".tv-readout").split("·")[0], "model units in the read-out")
        page.locator(".tv-range input[value=fixed]").check(force=True)
        page.fill(".tv-lo", "28")
        page.fill(".tv-hi", "29")
        page.dispatch_event(".tv-hi", "change")
        time.sleep(0.3)
        cov = page.inner_text(".tv-cover")
        check("outside" in cov and "%" in cov, f"fixed range: coverage note ({cov[:70]}…)")
        check(page.inner_text(".tv-clo") == "28.0" and page.inner_text(".tv-chi") == "29.0", "colour bar shows the fixed range")
        page.check('.tv-layers input[data-layer="voronoi"]')
        page.check('.tv-layers input[data-layer="mesh"]')
        page.wait_for_function("window.__otm.view._gpu.edgeN > 0", timeout=60_000)
        check(True, f"mesh edges layer ({ev('window.__otm.view._gpu.edgeN') // 2:,} edges)")
        page.uncheck('.tv-layers input[data-layer="mesh"]')       # a million lines are slow in software rendering
        page.mouse.wheel(0, -500)
        time.sleep(0.5)
        shot(page, out / "view_po2.png")
        page.select_option(".tv-map", "vo2")
        time.sleep(0.5)
        check("consumption" in page.inner_text(".tv-cunit"), "VO2 map with its unit")
        page.select_option(".tv-map", "lambda")
        time.sleep(0.3)
        check("Λ per fibre" in page.inner_text(".tv-cunit") and "3.00 × rest" in page.inner_text(".tv-cunit"),
              "Λ-per-fibre map with the section's value")
        check("switched on" in page.inner_text(".demand-result"), "results say Λ switched MM and Mb on")
        page.select_option(".tv-map", "types")
        time.sleep(0.3)
        check(page.locator(".tv-legend span").count() >= 3, "fibre-type map with a legend")
        page.select_option(".tv-map", "po2")
        with page.expect_download() as d:
            page.click(".tv-png")
        png = out / "saved_view.png"
        d.value.save_as(png)
        check(png.stat().st_size > 20_000, f"view saved as PNG ({png.stat().st_size // 1000} kB)")
        page.locator('#overview tr[data-name="OTM_sample_Dtect"]').click()
        page.wait_for_function("window.__otm.view && window.__otm.view.data && window.__otm.view.data.head.name == 'OTM_sample_Dtect'", timeout=60_000)
        check(page.locator(".tv-range input[value=fixed]").is_checked(), "the fixed range carries over to the next sample")
        with page.expect_download() as d:
            page.click(".save-otm")
        check(d.value.suggested_filename == "OTM_sample_Dtect.otm", "Save session (.otm) from the results")

        print("7  diagnostic report")
        page.click("#help-button")
        with page.expect_download() as d:
            page.click("#help-report")
        rep = Path(d.value.path()).read_text()
        check(rep.startswith("OTM Web diagnostic report") and "Send to: abdullah.alshammari@ku.edu.kw, subject: OTM2026" in rep
              and "flawed" in rep and "DT-F01" in rep and "otm_core 2026" in rep, "diagnostic report: contact, versions, files, data check")
        browser.close()
    srv.shutdown()
    print(f"\n{'PASSED' if not FAILS else 'FAILED'} in {time.time() - t_all:.0f} s")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
