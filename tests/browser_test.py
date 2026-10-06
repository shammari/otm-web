"""End-to-end test of OTM Web in headless Chromium (the phase 2 gate).

    python tests/browser_test.py --sample "path/to/1 Combined_export_cap_2014_12_1_11_33.mat"
                                 --baseline path/to/baseline_small.json [--out tests/output]

Serves the repository on localhost, then in the page: adds the sample twice
(once with its size in the file table) plus an unreadable file, round-trips the
settings JSON and the .dat parameters, runs the batch, compares the tables with
the MATLAB baseline, opens the figures and checks the ZIP. Needs Playwright
(``pip install playwright``; ``playwright install chromium``).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
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


def serve(port: int) -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "output")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    second = out / "copy_by_height.mat"
    shutil.copy(a.sample, second)
    broken = out / "broken.mat"
    broken.write_bytes(b"this is not a MATLAB file\n" * 20)
    base = json.loads(a.baseline.read_text())
    srv = serve(a.port)
    t_all = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page.on("pageerror", lambda e: FAILS.append(f"page error: {e}") or print("  page error:", e))
        page.on("console", lambda m: m.type == "error" and print("  console:", m.text[:300]))
        page.goto(f"http://127.0.0.1:{a.port}/index.html")

        print("1  engine")
        page.wait_for_selector("#engine.ready", timeout=240_000)
        print("   ", page.inner_text("#engine-text"), "|", page.inner_text("#versions"))
        check(page.locator("#params input").count() == 10, "10 parameter fields")

        print("2  files")
        page.set_input_files("#file-input", [str(a.sample), str(second), str(broken)])
        page.wait_for_function("window.__otm.state.files.every(f => f.status === 'ok' || f.status === 'bad')",
                               timeout=180_000)
        files = page.evaluate("window.__otm.state.files.map(f => ({name: f.name, status: f.status, check: f.check, error: f.error}))")
        for f in files:
            print("   ", f["name"], f["status"], {k: f["check"][k] for k in ("image_cols", "image_rows", "n_capillaries", "n_fibres", "fibre_types")} if f["check"] else f["error"])
        check([f["status"] for f in files] == ["ok", "ok", "bad"], "two readable files, the broken one flagged")
        check("left out" in page.inner_text("#file-table"), "unreadable file says it is left out")
        check(page.is_disabled("#run-button"), "Run disabled until the size is given")
        # second file: its own height (sample is 440 x 330 µm)
        page.locator('#file-table input[aria-label^="height of copy_by_height"]').fill("330")
        w2 = page.locator('#file-table input[aria-label^="width of copy_by_height"]')
        check(w2.input_value() == "440", f"width calculated from the typed height ({w2.input_value()})")
        page.fill("#size-value", "440")
        h1 = page.locator('#file-table input[aria-label^="height of sample"]')
        check(h1.get_attribute("placeholder") == "330", "the settings width shows the calculated height in the table")
        check("sample: 440 × 330 µm" in page.inner_text("#size-derived"), "settings show the calculated size")
        w2.fill("440")
        check(page.locator('#file-table input[aria-label^="height of copy_by_height"]').input_value() == "330",
              "typing the other side recalculates the first instead of clearing it")
        page.locator('#file-table input[aria-label^="height of copy_by_height"]').fill("330")
        check(page.is_enabled("#run-button"), "Run enabled with sizes")

        print("3  settings round trip")
        page.fill("#p-Pcap", "35")
        check(page.input_value("#preset") == "custom", "editing a parameter switches the preset to Custom")
        with page.expect_download() as d:
            page.click("#save-dat")
        dat = d.value.path().read_text().split()
        check(len(dat) == 10 and float(dat[1]) == 35.0, ".dat saved with 10 values")
        page.click("#reset-settings")
        page.fill("#size-value", "440")
        check(page.input_value("#p-Pcap") == "30", "defaults restore P_cap 30")
        page.set_input_files("#dat-input", str(d.value.path()))
        page.wait_for_function("document.querySelector('#p-Pcap').value === '35'")
        check(True, ".dat loaded back (P_cap 35)")
        page.select_option("#preset", "skeletal")
        page.check('input[name="range"][value="fixed"]', force=True)
        page.fill("#range-lo", "0")
        page.fill("#range-hi", "32")
        page.check('#formats input[value="svg"]')
        with page.expect_download() as d:
            page.click("#save-settings")
        saved = json.loads(d.value.path().read_text())
        check(saved["width_um"] == 440 and saved["po2_range_mmHg"] == [0, 32] and "svg" in saved["figure_formats"],
              "settings JSON has the form's values")
        page.check('input[name="tissue"][value="cardiac"]', force=True)
        check(page.input_value("#p-Pcap") == "40", "cardiac preset (P_cap 40)")
        page.set_input_files("#settings-input", str(d.value.path()))
        page.wait_for_function("document.querySelector('input[name=tissue][value=skeletal]').checked")
        check(page.input_value("#p-Pcap") == "30" and page.input_value("#range-hi") == "32",
              "settings JSON loaded back")
        page.uncheck('#formats input[value="svg"]')
        check(set(saved["parameters"]) == {"r_um", "Pcap", "P50_Mb", "P_c", "alpha", "D", "D_Mb", "M0", "c_Mb", "k"},
              "saved parameter names unchanged by the new labels")

        print("3b exercise summary, labels, layout")
        summ = page.inner_text("#exercise-summary")
        check("O₂ demand × 1" in summ and "Michaelis–Menten uptake: off" in summ, f"resting summary ({summ})")
        page.select_option("#exercise", "high")
        summ = page.inner_text("#exercise-summary")
        check("O₂ demand × 6" in summ and "permeability × 51.6" in summ and "Myoglobin facilitation: on" in summ,
              f"high-exercise summary ({summ})")
        check(page.is_hidden("#m0-note"), "no M0 note while M0 is the preset value")
        page.fill("#p-M0", "3e-4")
        note = page.inner_text("#m0-note")
        check(page.is_visible("#m0-note") and "1.8e-3" in note, f"effective-M0 note ({note})")
        page.check('input[name="tissue"][value="cardiac"]', force=True)
        summ = page.inner_text("#exercise-summary")
        check("O₂ demand × 20" in summ, f"cardiac high summary ({summ})")
        page.check('input[name="tissue"][value="skeletal"]', force=True)
        page.select_option("#exercise", "resting")
        page.select_option("#preset", "skeletal")
        check(page.is_hidden("#m0-note") and page.input_value("#p-M0") == "1.57e-4", "back to resting and the preset")
        subs = page.locator("#params sub").count()
        labels = page.inner_text("#params")
        check(subs >= 7 and "O₂ solubility" in labels and "O2" not in labels, f"subscripted labels ({subs} subscripts)")
        geo = page.evaluate("""() => {
            const r = (s) => document.querySelector(s).getBoundingClientRect();
            const ex = r('#exercise'), cb = r('#non-uniform'), cbl = document.querySelector('#non-uniform').closest('label').getBoundingClientRect(), de = r('#diff-extraction').left;
            const del = document.querySelector('#diff-extraction').closest('label').getBoundingClientRect();
            return {dy: Math.abs((ex.top + ex.bottom) / 2 - (cb.top + cb.bottom) / 2),
                    dyText: Math.abs((ex.top + ex.bottom) / 2 - (cbl.top + cbl.bottom) / 2),
                    gap: del.left - cbl.right, sameRow: Math.abs(del.top - cbl.top) < 20}; }""")
        check(geo["dy"] <= 2 and geo["dyText"] <= 2, f"checkbox and its text centred on the exercise level ({geo})")
        check(not geo["sameRow"] or geo["gap"] >= 28, f"space before differential extraction ({geo['gap']:.0f} px)")
        steps = page.evaluate("[...document.querySelectorAll('.step-link')].map(a => a.classList.contains('done'))")
        check(steps == [True, True, False, False], f"step bar before the run {steps}")
        page.locator("fieldset.wide").first.screenshot(path=str(out / "parameters.png"))
        page.screenshot(path=str(out / "page_before_run.png"), full_page=True)

        print("4  run (2 samples)")
        t0 = time.time()
        page.click("#run-button")
        page.wait_for_function("window.__otm.state.running === true", timeout=60_000)
        page.wait_for_function("window.__otm.state.running === false", timeout=1_500_000, polling=2000)
        print(f"    run took {time.time() - t0:.0f} s")
        steps = page.evaluate("[...document.querySelectorAll('.step-link')].map(a => a.classList.contains('done'))")
        check(steps == [True, True, True, True], f"step bar after the run {steps}")
        res = page.evaluate("Object.fromEntries(window.__otm.state.results)")
        check(len(res) == 2 and all(r["ok"] for r in res.values()), "both samples ran without errors")
        for name, r in res.items():
            print(f"    {name}: {r['seconds']:.0f} s, {len(r['files'])} files, timings "
                  + ", ".join(f"{k} {v:.1f}" for k, v in r["timings"].items()))
        r = res[files[0]["name"]]
        po2 = {row[0]: float(row[1]) for row in r["po2"]}
        ref = {"Tissue": base["po2_stats"]["Tissue"]["values"][0], "Interstitial space": base["po2_stats"]["Interstitia"]["values"][0]}
        for k, v in ref.items():
            rel = abs(po2[k] - v) / v
            check(rel < 2e-3, f"{k} mean PO2 {po2[k]:.2f} vs MATLAB {v:.3f} (rel {rel:.1e})")
        lcfr = r["summary"]["LCFR_mean"]
        check(abs(lcfr - base["key_metrics"]["mean_LCFR"]) < 1e-6, f"LCFR {lcfr:.6f} = MATLAB")
        check("exercise Resting: O₂ demand × 1" in page.inner_text(".sample-meta"), "run summary names the exercise level")
        r2 = res["copy_by_height"]
        check(abs(r2["size_um"][0] - 440) < 0.5 and abs(r2["size_um"][1] - 330) < 0.5, "per-file height gives 440 x 330 µm")
        check(r2["po2"] == r["po2"], "same data, same PO2 table")
        check(r["flux"]["capillaries_with_lines"] == r["flux"]["seed_capillaries"], "flux lines from every ROI capillary")
        check(page.locator("#overview tbody tr").count() == 2, "overview has a row per sample")
        rows_ind = page.locator(".block-indices tbody tr").count()
        check(rows_ind == len(r["indices"]) > 10, f"indices table shown ({rows_ind} rows)")
        hyp_rows = page.locator(".block-po2 tr.hypoxic").count()
        check(hyp_rows == len(r.get("po2_hypoxic_rows", [])), f"hypoxic rows marked ({hyp_rows})")

        print("5  figures")
        n_png = sum(1 for f in r["files"] if f.endswith(".png"))
        page.wait_for_function(f"[...document.querySelectorAll('.gallery img')].filter(i => i.naturalWidth > 0).length === {n_png}",
                               timeout=120_000)
        check(True, f"{n_png} figure thumbnails shown")
        pdfs = [f for f in r["files"] if f.endswith(".pdf")]
        check(len(pdfs) == n_png, f"a PDF beside every PNG ({len(pdfs)})")
        page.locator(".fig button.open").first.click()
        page.wait_for_selector("dialog.viewer img")
        page.screenshot(path=str(out / "figure_viewer.png"))
        page.keyboard.press("Escape")
        # sample with the hypoxia tag selected, page screenshot
        page.screenshot(path=str(out / "page_results.png"), full_page=True)

        print("6  ZIP")
        with page.expect_download(timeout=300_000) as d:
            page.click("#zip-button")
        zpath = out / "results.zip"
        d.value.save_as(zpath)
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            top = {n for n in names if "/" not in n}
            check(top == {"OTM_batch_summary.csv", "OTM_batch_summary.xlsx", "OTM_batch_settings.json"},
                  f"summary and settings at the top ({sorted(top)})")
            for name, rr in res.items():
                inside = {n for n in names if n.startswith(name + "/")}
                check(inside == set(rr["files"]), f"{name}: {len(inside)} files, INDICES and PO2 as written")
            summary = z.read("OTM_batch_summary.csv").decode("utf-8-sig").splitlines()
            check(len(summary) == 3, "summary CSV has a row per sample")
            st = json.loads(z.read("OTM_batch_settings.json"))
            check(st["width_um"] == 440 and st["po2_range_mmHg"] == [0, 32], "settings JSON in the ZIP")
        print(f"    ZIP {zpath.stat().st_size / 1e6:.1f} MB, {len(names)} files")

        print("7  hypoxia display (a made-up result)")
        page.evaluate("""() => { const o = window.__otm; const r = structuredClone(o.state.results.get(o.state.order[0]));
            r.name = 'hypoxic_demo'; r.files = []; r.po2_hypoxic_rows = [0, 5];
            r.hypoxia = 'HYPOXIA · 2.3 % of the tissue at or below P_c (Tissue, Type IIb)';
            o.storeResult(r); o.selectSample('hypoxic_demo'); }""")
        check(page.locator(".block-po2 tr.hypoxic").count() == 2, "hypoxic rows in the warning colour")
        check(page.is_visible(".hypoxia") and "HYPOXIA" in page.inner_text(".hypoxia"), "hypoxia tag shown")
        page.locator(".sample").screenshot(path=str(out / "hypoxia_demo.png"))
        browser.close()
    srv.shutdown()
    print(f"\n{'PASSED' if not FAILS else 'FAILED'} in {time.time() - t_all:.0f} s"
          + ("" if not FAILS else ":\n  " + "\n  ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
