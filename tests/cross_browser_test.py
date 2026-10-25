"""OTM Web in Firefox and WebKit (Safari's engine), with Selenium.

    python tests/cross_browser_test.py --browser firefox --firefox <path to firefox> --geckodriver <path>
    python tests/cross_browser_test.py --browser webkit            # WebKitGTK's WebKitWebDriver (needs a display, e.g. Xvfb)
    python tests/cross_browser_test.py --browser chrome            # Chrome or Edge via Selenium Manager

Playwright drives only its own patched browser builds, so this test uses the
browsers' own WebDriver. It checks what depends on the engine: Python in
WebAssembly in a module worker (and a second worker running at the same time),
the data check, the editor's data, a full run against the MATLAB reference,
WebGL in the tissue view, the PNG from the view, the deferred figures and the
ZIP. The page's test hooks (window.__otm) do the clicking where a WebDriver
would be fiddly. The sample is the one bundled with the site.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []
MATLAB = {"Tissue": 28.494, "Interstitial space": 29.039, "LCFR": 1.175439}


def check(ok: bool, what: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".js": "text/javascript", ".wasm": "application/wasm"}

    def log_message(self, *a):
        pass


def driver_for(a):
    from selenium import webdriver

    if a.browser == "firefox":
        from selenium.webdriver.firefox.service import Service

        o = webdriver.FirefoxOptions()
        if a.firefox:
            o.binary_location = a.firefox
        if not a.headed:
            o.add_argument("-headless")
        o.set_preference("webgl.force-enabled", True)            # software WebGL on machines without a GPU (CI)
        o.set_preference("webgl.disabled", False)
        o.set_preference("gfx.webrender.software", True)
        o.set_preference("dom.webdriver.enabled", False)
        return webdriver.Firefox(options=o, service=Service(a.geckodriver) if a.geckodriver else None)
    if a.browser == "webkit":
        from selenium.webdriver.webkitgtk.options import Options
        from selenium.webdriver.webkitgtk.service import Service

        o = Options()
        o.binary_location = a.webkit or "/usr/lib/x86_64-linux-gnu/webkit2gtk-4.1/MiniBrowser"
        o.add_argument("--automation")
        return webdriver.WebKitGTK(options=o, service=Service(executable_path=a.webkitdriver or "WebKitWebDriver"))
    if a.browser == "edge":
        from selenium.webdriver.edge.service import Service

        o = webdriver.EdgeOptions()
        return webdriver.Edge(options=o, service=Service(a.edgedriver) if a.edgedriver else None)
    o = webdriver.ChromeOptions()
    if not a.headed:
        o.add_argument("--headless=new")
    return webdriver.Chrome(options=o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", choices=("firefox", "webkit", "chrome", "edge"), required=True)
    ap.add_argument("--firefox"), ap.add_argument("--geckodriver")
    ap.add_argument("--webkit"), ap.add_argument("--webkitdriver")
    ap.add_argument("--edgedriver", help="msedgedriver.exe (else Selenium Manager downloads one)")
    ap.add_argument("--url", help="test a published site instead of serving this folder")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    url = a.url
    if not url:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(_Quiet, directory=str(ROOT)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{a.port}/index.html"
    t_all = time.time()
    d = driver_for(a)
    d.set_script_timeout(900)
    ev = d.execute_script

    def wait(js: str, timeout: float, what: str) -> bool:
        """Poll a page condition. It must be a plain true/false: returning a big
        object (a view with its mesh arrays) makes some WebDrivers hang."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if ev("return " + js):
                    return True
            except Exception:                      # noqa: BLE001 - page still loading
                pass
            time.sleep(0.5)
        check(False, f"timed out: {what}")
        return False

    def call(js: str):
        """Run async page code and return its result (JSON)."""
        return json.loads(d.execute_async_script(
            "const done = arguments[arguments.length - 1];"
            f"(async () => {{ {js} }})().then((r) => done(JSON.stringify(r ?? null)), (e) => done(JSON.stringify({{error: String(e && e.stack || e)}})));"))

    try:
        print(f"browser: {d.capabilities.get('browserName')} {d.capabilities.get('browserVersion')}")
        d.set_window_size(1360, 1000)
        d.get(url)
        print("1  engine")
        t0 = time.time()
        if not wait("document.querySelector('#engine.ready') !== null", 300, "engine ready"):
            print("   ", ev("return document.querySelector('#engine-text').textContent"))
            return 1
        print(f"    {ev('return document.querySelector(\"#engine-text\").textContent')} ({time.time() - t0:.0f} s) | "
              f"{ev('return document.querySelector(\"#versions\").textContent')}")
        caps = call("""const c = document.createElement('canvas'); const g = c.getContext('webgl2');
            return {webgl2: !!g, renderer: g ? g.getParameter(g.RENDERER) : null, cores: navigator.hardwareConcurrency,
                    mem: navigator.deviceMemory ?? null, dialog: typeof HTMLDialogElement, has: CSS.supports('selector(:has(a))'),
                    colormix: CSS.supports('color', 'color-mix(in srgb, red 50%, blue)')};""")
        print("   ", caps)
        check(caps["webgl2"], "WebGL 2 available")
        check(caps["dialog"] == "function" and caps["has"] and caps["colormix"], "dialog, :has() and color-mix() supported")

        print("2  files, data check, editor data")
        ev("document.querySelector('#try-sample').click()")
        wait("window.__otm.state.files.length === 1 && window.__otm.state.files[0].status === 'ok'", 120, "sample checked")
        f = ev("const f = window.__otm.state.files[0]; return {status: f.status, check: f.check && f.check.check && f.check.check.status, given: f.given, value: f.value}")
        check(f["status"] == "ok" and f["check"] == "ok" and f["value"] == "440", f"sample read, data check passed ({f})")
        p = call("""const f = window.__otm.state.files[0]; const e = window.__otm.engine;
            await e.call('put', {name: f.name, ext: '.mat', buffer: await f.mat.arrayBuffer()});
            const r = await e.call('prepare', {name: f.name, opts: {tissue: 'skeletal'}, edits: [{kind: 'add_capillary', x: 1, y: 2}]});
            return {ok: r.ok, caps: r.caps.length, fibres: r.fibres.length, edits: r.edits_text};""")
        check(p["ok"] and p["caps"] == 128 and p["fibres"] == 89, f"editor data with a correction ({p})")

        print("3  run (two copies, two workers, figures afterwards)")
        call("""const st = window.__otm.state; const f = st.files[0];
            const copy = new File([await f.mat.arrayBuffer()], 'copy_two.mat');
            const before = st.files.length;
            window.__otm.addFiles([copy]);
            return before;""")
        wait("window.__otm.state.files.length === 2 && window.__otm.state.files.every(f => f.status === 'ok')", 120, "copy checked")
        ev("""const st = window.__otm.state; st.files[1].given = 'width'; st.files[1].value = '440';
              for (const i of document.querySelectorAll('#formats input')) i.checked = i.value === 'png';
              document.querySelector('#dpi').value = '300';
              document.querySelector('#parallel').value = '2';
              window.__otm.renderFiles(); window.__otm.validate();""")
        wait("!document.querySelector('#run-button').disabled", 30, "Run enabled")
        t0 = time.time()
        ev("document.querySelector('#run-button').click()")
        wait("window.__otm.state.results.size >= 1", 600, "first result")
        t_first = time.time() - t0
        wait("!window.__otm.state.running && window.__otm.state.results.size === 2", 900, "run finished")
        t_run = time.time() - t0
        res = ev("return Object.fromEntries([...window.__otm.state.results].map(([k, r]) => [k, {ok: r.ok, errors: r.errors, "
                 "po2: r.po2, lcfr: r.summary && r.summary.LCFR_mean, worker: r.worker, files: r.files.length, "
                 "png: r.files.filter(x => x.endsWith('.png')).length, timings: r.timings, view: !!r.view}]))")
        for n, r in res.items():
            print(f"    {n}: {r['worker']} worker, {r['files']} files ({r['png']} PNG), timings "
                  + ", ".join(f"{k} {v:.1f}" for k, v in (r["timings"] or {}).items()))
        print(f"    first result after {t_first:.0f} s, both done after {t_run:.0f} s")
        check(all(r["ok"] for r in res.values()), "both samples without errors " + str([r["errors"] for r in res.values() if not r["ok"]]))
        check(sorted(r["worker"] for r in res.values()) == ["extra", "main"], "the second sample ran in a second worker at the same time")
        check(all(r["png"] >= 17 for r in res.values()), "figures written after the results (17 PNG each)")
        r = next(iter(res.values()))
        po2 = {row[0]: float(row[1]) for row in r["po2"]}
        for k in ("Tissue", "Interstitial space"):
            rel = abs(po2[k] - MATLAB[k]) / MATLAB[k]
            check(rel < 2e-3, f"{k} mean PO2 {po2[k]:.2f} vs MATLAB {MATLAB[k]} (rel {rel:.1e})")
        check(abs(r["lcfr"] - MATLAB["LCFR"]) < 1e-6, f"LCFR {r['lcfr']:.6f} = MATLAB")

        print("4  tissue view (WebGL), PNG, ZIP")
        for name, rr in res.items():
            if rr["worker"] == "extra":
                ev(f"window.__otm.selectSample({json.dumps(name)})")
        ok = wait("!!(window.__otm.view && window.__otm.view.data)", 120, "tissue view of the sample from the second worker")
        print("    view state:", ev("const s = window.__otm.state; return {selected: s.selected, packs: [...s.packs.keys()], "
                                     "view: !!window.__otm.view}"))
        time.sleep(1.5)
        d.set_script_timeout(120)
        px = call("""const c = document.querySelector('.tv-gl'); const v = window.__otm.view; v._draw();
            const g = c.getContext('webgl2'); const w = c.width, h = c.height; const buf = new Uint8Array(4 * 16 * 16);
            g.readPixels(Math.floor(w / 2) - 8, Math.floor(h / 2) - 8, 16, 16, g.RGBA, g.UNSIGNED_BYTE, buf);
            const colours = new Set(); for (let i = 0; i < buf.length; i += 4) colours.add(buf[i] + ',' + buf[i + 1] + ',' + buf[i + 2]);
            return {w, h, colours: colours.size, sample: [...colours].slice(0, 3)};""")
        check(px.get("colours", 0) > 3, f"PO2 map drawn with WebGL ({px})")
        ro = call("""const v = window.__otm.view; const st = v.el.stage.getBoundingClientRect();
            v._hover(st.width / 2, st.height / 2); return v.el.readout.textContent;""")
        check("PO₂" in ro and "mmHg" in ro, f"hover read-out ({ro})")
        png = call("""const v = window.__otm.view; v._draw();
            const c = document.createElement('canvas'); c.width = v.el.gl.width; c.height = v.el.gl.height;
            const x = c.getContext('2d'); x.drawImage(v.el.gl, 0, 0); x.drawImage(v.el.ov, 0, 0);
            const url = c.toDataURL('image/png'); return url.length;""")
        check(png > 20000, f"view composed into a PNG ({png // 1000} kB as data URL)")
        z = call("""const st = window.__otm.state; const b = await window.__otm.engine.call('zip', {names: st.order, settings: st.lastSettings});
            return {bytes: b.byteLength, head: String.fromCharCode(b[0], b[1])};""")
        check(z["head"] == "PK" and z["bytes"] > 5e6, f"ZIP of both samples ({z['bytes'] / 1e6:.1f} MB)")
        check(not ev("return window.__otm.state.log.filter(e => e.text.startsWith('page error')).map(e => e.text).join('; ')"),
              "no page errors logged")
    finally:
        try:
            d.quit()
        except Exception:                          # noqa: BLE001
            pass
    print(f"\n{'PASSED' if not FAILS else 'FAILED'} in {time.time() - t_all:.0f} s")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
