// OTM Web analysis worker: Pyodide (Python in WebAssembly) + otm_core + the Triangle mesher.
// Runs in the background so the page stays responsive. The page sends {id, cmd, args};
// the worker answers {id, ok, result | error} and, while running, {type: "progress" | "status"}.
// Commands run one at a time, in the order they arrive.

import { loadPyodide } from "../pyodide/pyodide.mjs";
import { Triangle } from "./triangle.mjs";

const here = (p) => new URL(p, import.meta.url).href;
const WHEELS = ["scikit_fem-12.0.2-py3-none-any.whl", "openpyxl-3.1.5-py2.py3-none-any.whl",
                "et_xmlfile-2.0.0-py3-none-any.whl"];
const PACKAGES = ["numpy", "scipy", "shapely", "matplotlib", "h5py", "micropip"];

let py = null;        // the Pyodide instance
let web = null;       // the otm_web Python module
const status = (text) => postMessage({ type: "status", text });

// ---- Triangle bridge: otm_core.fem.mesh_triangle calls this through set_triangulator ----------
function triangulate(switches, d) {
  const order = ["pointlist", "trianglelist", "triangleattributelist", "trianglearealist", "segmentlist",
                 "segmentmarkerlist", "holelist", "regionlist"];
  const data = {};
  for (const k of order) { const v = d[k]; if (v !== undefined && v !== null && v.length) data[k] = v; }
  const input = Triangle.makeIO(data);
  if (data.trianglelist) input.arr[10] = 3;              // numberofcorners, needed for refinement
  const out = Triangle.makeIO();
  Triangle.triangulate(switches, input, out);
  const cp = (a, T) => (a ? T.from(a) : new T(0));
  const res = {
    pointlist: cp(out.pointlist, Float64Array), trianglelist: cp(out.trianglelist, Int32Array),
    triangleattributelist: cp(out.triangleattributelist, Float64Array),
    segmentlist: cp(out.segmentlist, Int32Array), segmentmarkerlist: cp(out.segmentmarkerlist, Int32Array),
  };
  Triangle.freeIO(input, true);
  Triangle.freeIO(out);
  return res;
}

async function writePythonFiles() {
  const list = await (await fetch(here("python-files.json"))).json();
  const root = "/home/pyodide/lib";
  await Promise.all(list.files.map(async (rel) => {
    const resp = await fetch(here("../" + rel));
    if (!resp.ok) throw new Error(`could not load ${rel} (${resp.status})`);
    const text = await resp.text();
    const dest = `${root}/${rel.startsWith("app/") ? rel.slice(4) : rel}`;
    py.FS.mkdirTree(dest.slice(0, dest.lastIndexOf("/")));
    py.FS.writeFile(dest, text);
  }));
  return root;
}

async function init() {
  const t0 = performance.now();
  status("Loading Python (first visit: about 45 MB, then cached)…");
  py = await loadPyodide({ indexURL: here("../pyodide/") });
  status("Loading the mesher…");
  await Triangle.init(here("triangle.wasm"));
  status("Loading numpy, scipy, matplotlib…");
  await py.loadPackage(PACKAGES);
  status("Loading scikit-fem and openpyxl…");
  const micropip = py.pyimport("micropip");
  await micropip.install(WHEELS.map((w) => here("../wheels/" + w)), { deps: false });
  status("Loading OTM…");
  const root = await writePythonFiles();
  py.registerJsModule("otm_triangle_js", { triangulate });
  await py.runPythonAsync(`
import sys, numpy as np
sys.path.insert(0, ${JSON.stringify(root)})
from pyodide.ffi import to_js
from js import Object
import otm_triangle_js
from otm_core.fem import mesh_triangle as _mt

def _tri_js(switches, data):
    d = {k: to_js(np.ascontiguousarray(v).ravel()) for k, v in data.items() if v is not None and np.size(v)}
    out = otm_triangle_js.triangulate(switches, to_js(d, dict_converter=Object.fromEntries))
    return {k: np.asarray(getattr(out, k).to_py()) for k in
            ("pointlist", "trianglelist", "triangleattributelist", "segmentlist", "segmentmarkerlist")}

_mt.set_triangulator(_tri_js)
import matplotlib
matplotlib.use("Agg")
import otm_web
`);
  web = py.pyimport("otm_web");
  const info = JSON.parse(web.info());
  info.pyodide = py.version;
  info.load_seconds = (performance.now() - t0) / 1000;
  return info;
}

// ---- commands ---------------------------------------------------------------------------------
const commands = {
  init,
  put({ name, ext, buffer }) {
    py.FS.writeFile(`/data/${name}${ext}`, new Uint8Array(buffer));
    return true;
  },
  inspect: ({ name }) => JSON.parse(web.inspect(name)),
  corrections: ({ name }) => JSON.parse(web.corrections(name)),
  drop: ({ name }) => { web.drop(name); return true; },
  forget: ({ name }) => { web.forget(name); return true; },
  check: ({ settings }) => JSON.parse(web.check_settings(JSON.stringify(settings))),
  parseDat: ({ text }) => JSON.parse(web.parse_parameters(text)),
  parseDims: ({ text }) => JSON.parse(web.parse_dimensions(text)),
  run({ name, settings }) {
    let last = 0;
    const progress = (f, msg) => {
      const now = performance.now();
      if (now - last > 120 || f >= 1) { last = now; postMessage({ type: "progress", name, f, msg }); }
    };
    return JSON.parse(web.run(name, JSON.stringify(settings), progress));
  },
  zip({ names, settings }) {
    const path = web.make_zip(JSON.stringify(names), JSON.stringify(settings));
    const bytes = py.FS.readFile(path);
    py.FS.unlink(path);
    return transfer(bytes);
  },
  file({ path }) {
    if (path.includes("..")) throw new Error("bad path");
    return transfer(py.FS.readFile("/out/" + path));
  },
};

const TRANSFER = Symbol("transfer");
function transfer(u8) { return { [TRANSFER]: true, value: u8 }; }

let queue = Promise.resolve();
onmessage = ({ data }) => {
  const { id, cmd, args } = data;
  queue = queue.then(async () => {
    try {
      if (!commands[cmd]) throw new Error(`unknown command ${cmd}`);
      if (cmd !== "init" && !web) throw new Error("Python is not loaded");
      let result = await commands[cmd](args || {});
      if (result && result[TRANSFER]) {
        const u8 = result.value;
        postMessage({ id, ok: true, result: u8 }, [u8.buffer]);
        return;
      }
      postMessage({ id, ok: true, result });
    } catch (e) {
      postMessage({ id, ok: false, error: cleanError(e) });
    }
  });
};

function cleanError(e) {
  const text = String(e && e.message ? e.message : e);
  // Python tracebacks: keep the last line (the error itself) and the full text for the details box
  const lines = text.trim().split("\n");
  return { message: lines[lines.length - 1], details: text };
}
