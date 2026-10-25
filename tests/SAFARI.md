# Testing OTM Web in Safari (macOS)

Safari's engine (WebKit) passes the automated test on Linux (WebKitGTK 2.52), but Safari itself has not been run yet. Either of the two routes below does that; the first takes about 10 minutes, most of it waiting.

Note the macOS and Safari versions (Apple menu › About This Mac; Safari › About Safari) and whether the Mac has Apple silicon or Intel.

## Route A: the automated test (recommended)

1. **Allow automation, once.** Safari › Settings › Advanced › tick "Show features for web developers". Then Develop › "Allow Remote Automation". In Terminal: `safaridriver --enable` (asks for your password).
2. **Get the code.** Download the branch as a ZIP from GitHub (Code › Download ZIP) and unzip it, or `git clone https://github.com/shammari/otm-web`.
3. **Install Selenium.** In Terminal, in the unzipped folder: `python3 -m pip install --user selenium`.
4. **Run.** `python3 tests/cross_browser_test.py --browser safari`. A Safari window with an orange address bar opens and works by itself. Do not click in it until the Terminal prints `PASSED` or `FAILED`.
5. **Send back** the whole Terminal output (copy and paste, or `... > safari_log.txt 2>&1`).

To test the published site instead of the local copy, add `--url https://<published address>/index.html`.

## Route B: by hand (no Terminal)

Open the site in Safari and tick each step. Note any step that fails, with a screenshot, and anything in Develop › Show JavaScript Console.

| # | Do | Expect |
|---|----|--------|
| 1 | Open the site and wait | The top-right corner counts to "ready" (first visit about 45 MB) |
| 2 | Click "Try the sample section" | One row; the data check passes |
| 3 | Click "Correct…" (or open the editor) | Fibres and capillaries drawn, the counts shown |
| 4 | Biophysical parameters › Oxygen demand Λ, with Exercise level "Resting", then "Low" | Resting: a resting estimate. Low: about 2 × the resting Λ, non-resting |
| 5 | Back to "Resting". If you have a Dtect export, add it too and set "Samples at a time" to 2. Click Run | Each sample shows its numbers first and its figures afterwards; with two files both run at once |
| 6 | The results table | Tissue PO₂ 28.50 mmHg for the sample at the default settings; "Λ × rest" 1.00 |
| 7 | Tissue view: the PO₂ map, then "Oxygen demand Λ per fibre" | Colour maps drawn (WebGL); hovering shows PO₂ and Λ |
| 8 | "Save PNG" in the tissue view | A PNG downloads |
| 9 | "Download all results (ZIP)" | The ZIP opens in Finder, with INDICES and PO2 folders per sample |
| 10 | Reload the page | Ready again in about 10 seconds (cached) |

## Known WebKit points

- On Linux WebKit was the slowest engine (first result 75 s, two samples in 142 s, against 29 s and about 60 s in Chrome). Safari on Apple silicon should be faster; please note the times in step 5 (the Time column).
- Safari limits the memory of a tab. If it reloads the page during step 5, set "Samples at a time" to 1 and say so.
