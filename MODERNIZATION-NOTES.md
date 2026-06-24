# Modernization Notes — `makeabilitylab/signals` v2.0

Report back to the `physcomp` textbook side (tracking issue **#116**). Living document;
updated as the lesson-by-lesson pass proceeds.

**Status:** Pass 1 (repo infrastructure), Pass 2 (lesson-by-lesson modernization), the
Colab-bootstrap sweep, and Pass 3 (quality/pedagogy review) are **complete**.
**Pass 4 (accessibility, engagement/`ipywidgets` sliders, and textbook-style lesson openers)
is COMPLETE** — see the **Pass 4** section at the bottom of this file. **Date:** 2026-06-24.

---

## Environment

- **Interpreter:** Python **3.12** (verified on 3.12.10).
- **Setup one-liner (venv):** `python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt`
- **Setup one-liner (conda):** `conda env create -f environment.yml && conda activate signals`
- Both install the local helper packages editable (`pip install -e .`), so:
  - `import makelab.signal` / `import makelab.audio`
  - `import gesturerec.data` (etc.)
  work from **any** directory with no `sys.path` hacks.
- Pinned stack: numpy 2.4.6 · scipy 1.18.0 · matplotlib 3.11.0 · pandas 3.0.3 ·
  scikit-learn 1.9.0 · seaborn 0.13.2 · librosa 0.11.0 · jupyterlab 4.6.0.

## Packaging / path changes (contract-relevant)

- **No notebook files were renamed or moved.** All 10 published paths are unchanged.
- `makelab` and `gesturerec` were left **in place** and made importable via a root
  `pyproject.toml` (`tool.setuptools.package-dir` maps each to a top-level import name).
  Added empty `__init__.py` to each. No imports in the notebooks needed to change.
- **Open recommendation (your call):** the four notebooks with **spaces in their
  filenames** (`Signals - *.ipynb`, `Feature Selection and Hyperparameter Tuning.ipynb`)
  work fine, but de-spacing would simplify Colab/Binder URLs. This is a **breaking change
  to the textbook's 11 wrapper links**, so it is NOT done unilaterally — decide and I'll
  do it + list old→new paths here.

## Colab / Binder badges

- Badge URLs for all 10 notebooks are in `README.md`, pointing at branch `master`.
- **Binder** builds from `environment.yml`/`requirements.txt` and checks out the full
  repo, so data + helper imports work there once these changes are pushed.
- **Colab** loads only the single notebook — it does **not** get the repo's data or the
  `makelab`/`gesturerec` packages. Each data/helper-dependent notebook needs a small "Colab
  setup" bootstrap cell (clone repo + `pip install -e .` + `cd` into the lesson folder).
  **Decision (with Jon, 2026-06-23): do this as a dedicated cross-cutting sweep AFTER the
  code-modernization pass on all 10**, not interleaved — because (a) the bootstrap clones from
  GitHub `master`, so it can only be truly validated post-push; (b) one sweep keeps the near-
  identical cells uniform; (c) clean separation of "runs on modern stack" vs "runs in Colab".
  Covers **NB4, 5, 6, 7, 8, 9, 10** (NB1–3 need no data; NB3 already has its own ipympl/widget
  Colab guard). Proposed guarded, idempotent pattern (no-op outside Colab), as the first code cell:
  ```python
  import sys, os
  if 'google.colab' in sys.modules:
      if not os.path.exists('signals'):
          !git clone -q https://github.com/makeabilitylab/signals.git
      %pip install -q -e signals
      os.chdir('signals/<lesson folder>')   # e.g. Tutorials  or  Projects/StepTracker
  ```
  **Status: ✅ done (2026-06-23).** Added a guarded markdown+code bootstrap (idempotent clone +
  `pip install -e signals` + `os.chdir` into the lesson folder) before the first code cell of all
  8 data/helper notebooks: NB4/5/6 (→ `Tutorials`), NB7 Exercises **and** WithExampleSolution
  (→ `Projects/StepTracker`), NB8/9/10 (→ `Projects/GestureRecognizer`). Inserted as no-output
  cells (no re-exec needed); verified locally as a clean no-op (the `if 'google.colab'` guard is
  False off-Colab). **Still to validate post-push:** the actual Colab launch (the clone pulls from
  GitHub `master`, so it only works once this branch is merged) — confirm each badge end-to-end then.
- **Not yet verifiable:** badges can only be confirmed to actually launch *after* these
  commits are pushed to `master` on GitHub. Flagging so we verify post-push, not assume.

## Cruft removed

- Deleted `Untitled.ipynb` and all tracked `.ipynb_checkpoints/` entries.
- Rewrote `.gitignore` (clean patterns; preserves the intentional ignores for
  `*-Private.ipynb` solution keys, the `-NanDu` student variant, and `GestureLogsTemp/`).
- Added `.venv/` to `.gitignore`.

---

## Baseline execution sweep (the Pass 2 work list)

Every notebook was run headless ("Restart & Run All" equivalent) on the new 3.12 stack.
Originals were not modified. **3/10 already pass clean; 7/10 stop at a first error.**
The failures split into *intentional teaching cells* (just need a cell tag) vs *real
breakages* (need code fixes).

| # | Notebook | Sweep | First blocker | Pass 2 action |
|---|----------|-------|---------------|---------------|
| 1 | IntroToPython | ❌ | `SyntaxError`: `print 'x' in animals` — **Python-2 print statements** (~4) | Convert to `print(...)`. Real bug. |
| 2 | IntroToNumPy | ❌ | `np.array(1,2,3,4)  # WRONG, will throw` — **intentional** error demo | Tag cell `raises-exception`. (A one-off zmq kernel-death also appeared; non-reproducible flake.) |
| 3 | IntroToMatplotlib | ✅ | — | Passes despite a `%matplotlib notebook` magic; still switch to `inline`/`widget` for Colab. |
| 4 | Quantization & Sampling | ❌ | `AttributeError: module 'librosa.display' has no attribute 'waveplot'` | `librosa.display.waveplot` → `waveshow` (renamed in librosa ≥0.10). Real bug. |
| 5 | Comparing Signals | ❌ | `distance.euclidean` dimension-mismatch — **intentional** error demo | Tag cell `raises-exception`. |
| 6 | Frequency Analysis | ✅ | — | Clean. |
| 7 | StepTracker (Exercises) | ✅ | — | Clean (note: still has the Colab `Logs/` manual-folder step to fix). |
| 8 | GestureRec: Shape-Based | ❌ | `AttributeError: 'NoneType' object has no attribute 'map_gestures_to_trials'` | Gesture-set lookup/load returns `None`; investigate data-dir name match + load order. Real bug. |
| 9 | GestureRec: Feature-Based | ❌ | `AttributeError: 'DataFrame' object has no attribute 'append'` | `df.append()` removed in pandas 2.0 → `pd.concat`. Also fix `np.bool`→`bool` and `%matplotlib notebook`. Real bug. |
| 10 | Feature Selection & Hyperparam | ❌ | `TypeError: unhashable type: 'StringArray'` | pandas/seaborn/sklearn churn around string-dtype columns. Also `np.bool`→`bool`. Real bug. |

### Cross-cutting fixes confirmed present in the corpus
- `np.bool` — **5 occurrences** (notebooks 9 and 10). ⚠️ **Update:** `np.bool` was removed in
  NumPy 1.24 but **re-added as an alias in NumPy 2.0**, so on our 2.4 stack it is NOT broken.
  Changed to builtin `bool` anyway for clarity/consistency (a style fix, not a bug fix).
- `%matplotlib notebook` — **6 occurrences** (notebooks 3 and 9). Broken in Notebook 7 /
  JupyterLab / Colab → `%matplotlib inline` (or `widget` where interaction is needed).
- `sklearn.externals` — searched, **not** present (no action).
- Intentional `raises-exception` cells (notebooks 2, 5) need tagging so Restart-&-Run-All
  completes for the rendered reading copies.

---

## Community PRs

- **PR #2** (Kartik Patil / @ARcKP98, 2024-03-10) — "Add instructions on using import" to
  `IntroToPython.ipynb`. *Content is on-point* (fulfills the notebook's own import TODO) but the
  PR itself is **superseded** by the v2 rewrite (conflicts after our branch lands) and has minor
  quality issues (typos "importinhg"/"randomn_number"; bumps metadata to Python 3.10, behind our
  3.12). **Disposition:** carried the idea forward as a fresh, modernized "Importing libraries"
  section in our v2 notebook; credit Kartik via `Co-authored-by:` on the commit; then close PR #2
  with a warm thank-you. _Awaiting Jon's go-ahead to post the close (acts under his GitHub account)._

## Per-notebook status (Pass 2)

### 1. IntroToPython — ✅ done (2026-06-23)
- **Real bug fixed:** converted the remaining Python-2 `print` statements to `print(...)`
  calls — 6 in the string-methods cell, plus 2 each in the three set-operations cells
  (10 statements across 4 cells). These were the `SyntaxError` blocker; ironic given the
  notebook's own changelist already claims "Updated all examples to Python 3."
- **Intentional-error cells tagged:** the `KeyError: 'monkey'` dict-lookup cell and the
  `t[0] = 1` tuple-immutability cell were tagged `raises-exception` so Restart-&-Run-All
  completes for the rendered reading copy (same treatment notebooks 2 and 5 need).
- **Re-executed in place** on the 3.12 stack — clean exit 0, every cell's output now
  matches its Python-3 source. No data loading / helper imports, so **no Colab bootstrap
  cell is needed** for this notebook.
- **Added a modernized "Importing libraries" section** (2026-06-23) fulfilling the notebook's
  own "Future TODOs" item (and removed that TODO line). Crisp: `import` + dot notation,
  `from x import y`, `import numpy as np` aliasing, plus the PEP-8 "imports at top" convention;
  f-strings, official-doc links, no bloat. This **carries forward community PR #2** (Kartik
  Patil / @ARcKP98) which targeted the same TODO — see PR disposition below. Credit will be a
  `Co-authored-by:` trailer on the commit, not an in-notebook note.

### 2. IntroToNumPy — ✅ done (2026-06-23)
- **No code bugs** — the blocker was purely the intentional `np.array(1,2,3,4)` error demo.
- **Intentional-error cells tagged:** *two* cells, not one — the `np.array(1,2,3,4)` cell
  **and** the shape-mismatch `x - y` (shapes `(5,)` vs `(3,)`) cell further down. The second
  would have become the next blocker once the first was tagged. Both tagged `raises-exception`.
- **numpy-2.4 behavior note:** `np.array(1,2,3,4)` now raises **`TypeError`** ("takes from 1
  to 2 positional arguments…"), not the old `ValueError: only 2 non-keyword arguments accepted`.
  Pedagogy is unaffected — the prose above the cell just says "will throw an exception."
- The one-off zmq kernel-death the baseline sweep saw here did **not** reproduce (confirmed flake).
- **Re-executed in place** — clean exit 0. Imports only numpy; no data/helper imports, so **no
  Colab bootstrap cell needed.**

### 3. IntroToMatplotlib — ✅ done (2026-06-23)
- **Interactive backend modernized:** `%matplotlib notebook` (the dead nbagg backend, removed
  in Notebook 7 / JupyterLab) → **`%matplotlib widget`** (ipympl, already pinned at 0.10.0 in
  Pass 1). Rewrote the "Making the plots interactive using magic!" markdown to explain the
  modern backend story (`inline` vs `widget`), call out that `notebook` is gone, and note the
  Colab `enable_custom_widget_manager()` step.
- **Static-render safety:** re-executed headless (clean exit 0). ipympl saves a mime-bundle of
  `widget-view+json` + `text/html` + `image/png`. **Committed only the `image/png` fallback**
  (stripped `widget-view+json` and `text/html`, and removed the nb-level `metadata.widgets`
  state store). Reason: a widget-aware static renderer (nbviewer) would otherwise pick the saved
  widget-view and show a **broken "widget state not found" placeholder** with no live kernel.
  With png-only, every static surface shows a clean image; interactivity regenerates when a
  reader actually runs the cell in Binder/Colab/local.
- **Size:** 0.6 MB → 1.9 MB after naive execution → **0.9 MB** after the strip (21 consistent
  static PNGs; the old version had one 274 KB nbagg JS blob).
- No other code bugs; imports numpy/matplotlib/random only — no data/helper imports.

### 4. Signals - Quantization and Sampling — ✅ done (2026-06-23)
- **Real bug fixed (the blocker):** `librosa.display.waveplot` → **`waveshow`** (renamed/removed
  in librosa ≥0.10). Cell 73.
- **Latent bug fixed:** cell 5 called a bare `convert_to_mono(...)` that is **undefined** in the
  notebook namespace (it lives in `makelab.audio`). Qualified to `makelab.audio.convert_to_mono(...)`,
  matching the `makelab.signal.*` style used throughout. The default file is mono so the branch is
  never taken with stock data — but it's a `NameError` waiting for anyone who swaps in a stereo file
  (and the comment invites exactly that).
- **Non-issue checked & dismissed:** `sp.io.wavfile` resolves fine on scipy 1.18 (lazy submodule
  loading) — no explicit `import scipy.io` needed.
- **Verified clean:** full headless Restart-&-Run-All → exit 0.
- **Committed as a minimal diff (see policy below):** 11 insertions / 15 deletions; file stays ~12.9 MB
  (orig 12.8 MB) instead of ballooning to 19 MB from a full-output rewrite.

### 5. Signals - Comparing Signals — ✅ done (2026-06-23)
The baseline sweep stopped at the intentional error (cell 11) and **masked several real
breakages after it**. Found and fixed all of them:
- **Intentional-error cell tagged** `raises-exception` (cell 11: `distance.euclidean` on
  mismatched-length arrays — the documented blocker).
- **Real bug — matplotlib:** `axes.stem(..., use_line_collection=True)` ×2. That kwarg was
  **removed in matplotlib 3.9** → `TypeError` on 3.11. Removed it (it's now the default behavior).
- **Real bug — wrong module:** cell 41 called `makelab.audio.plot_signal`, but `plot_signal`
  lives only in `makelab.signal` → `AttributeError`. Fixed (cell 40 already did it right).
- **Real bug — unmaintained/broken dep:** `from fastdtw import fastdtw`. fastdtw isn't pinned,
  and even when installed it's **broken on the modern stack** — its `dist=distance.euclidean`
  passes scalars to scipy's `euclidean`, which now rejects 0-d input (`AxisError`). **Decision
  (with Jon): migrate to `librosa.sequence.dtw`** (already a pinned dep — drops the abandoned
  library with *zero* new dependencies). Rewrote the code cell + the stale "## FastDTW" markdown
  section (now "## DTW with librosa") + the "we use fastdtw below" line in the intro. **NB8
  (GestureRec ShapeBased) mentions fastdtw too — handle there.** *(Resolved: NB8 only had a
  markdown recommendation + a student stub, no fastdtw code to migrate — see NB8 entry.)*
- **Correctness/modernization:** 9× `bshift_method is 'all'` (and friends) — `is` with string
  literals (raises `SyntaxWarning`, semantically wrong identity-vs-equality) → `==`.
- **Readability:** cell 48 had `&lt;`/`&gt;` HTML entities in a code comment → real `<`/`>`
  operators, and split a run-on comment onto two lines.
- **Readability:** de-collided imports — `from scipy import signal` + `from makelab import
  signal` rebind the same name; switched to `import scipy.signal` / `import makelab.signal` /
  `import makelab.audio` (submodule imports, no shadowing; all call sites are fully-qualified).
- **Verified:** full headless run → exit 0 (only error is the tagged cell 11). Committed the
  fresh re-exec (1.6 → 2.3 MB; substantive multi-cell change on a small file justifies it, vs
  NB4's minimal-diff).

### 6. Signals - Frequency Analysis — ✅ done (2026-06-23)
Genuinely close to clean (baseline ✅). Big notebook (106 cells, audio + many plots, ~6 MB).
- **Pre-existing uncommitted churn** (present at session start) was benign metadata only — a
  Jupyter open/save had bumped the py-version string / kernel name / `nbformat_minor` and dropped
  some `"scrolled"` cell metadata. No content to preserve.
- **Real bug:** cell 5 `if i + 1 is not len(freqs):` — `is not` comparing **integers** (works only
  by small-int caching; raises `SyntaxWarning`) → `!=`.
- **Readability / dead imports:** de-collided the scipy/makelab `signal` shadowing (→
  `import scipy.signal` / `import makelab.signal`) and removed two **unused** imports
  (`scipy.spatial.distance`, `makelab.audio` — verified 0 call sites) plus a "matplot lib" typo.
- **Widget note:** uses `ipywidgets.FloatProgress` (a progress bar during the brute-force
  cross-correlation DFT) — works fine on ipywidgets 8.x; no change needed.
- **Verified + committed minimal:** full headless re-exec → exit 0 (confirmed on the 3.12 stack),
  but since both edits touch only a function def / the import cell (**no output changes**), I
  committed HEAD outputs + the 2 source edits + modern kernel metadata. Diff is 6/9 lines, file
  stays 5.9 MB (a full re-exec would have churned it to 9.5 MB / 2,100 lines for nothing).

### 7. StepTracker — Exercises + WithExampleSolution — ✅ code-complete (2026-06-23)
- **No code bugs.** Both the Exercises and the example-solution notebook are small (27/24
  cells) and run clean on the 3.12 stack (verified Exercises headless → exit 0).
- Checked and dismissed: the `.append(` is a **list** append (`peak_locations.append(...)`),
  not pandas `df.append()`; `from scipy import signal` is genuinely used (6 bare `signal.`
  calls, no makelab collision here) so it stays.
- **Colab data-loading pending the sweep** (below): both load `Logs/` via a repo-relative path.
- No file committed — left at HEAD (no source changes); the verification re-exec was discarded
  to avoid churning plot PNGs for nothing.

### 8. GestureRecognizer-ShapeBased — ✅ done (2026-06-23)
The baseline's diagnosis was slightly off: the notebook gets **past** cell 23 (it uses
`selected_gesture_set`, which resolves to Jon's set fine). Running it surfaced the real chain:
- **Real blocker:** cell 32 hard-coded `get_gesture_set_with_str(map_gesture_sets, "Easy")`, but
  there is **no "Easy" gesture set** in the repo (data is participant-named: `JonGestures`, …) →
  returned `None` → `NoneType … map_gestures_to_trials` in `generate_kfolds_scikit`. Fixed to reuse
  `selected_gesture_set` (already chosen in cell 7), which is always valid + matches the narrative.
- **Real bug — sklearn:** cell 49 `confusion_matrix(y_true, y_pred, labels)` → modern sklearn's
  `@validate_params` makes `labels` **keyword-only** → `TypeError: too many positional arguments`.
  Fixed to `labels=labels`.
- **Stale advice (NOT a code dep):** `fastdtw` appears only in a **markdown** cell recommending
  students `pip install fastdtw` for DTW. `find_closest_match_dtw` is just a student stub
  (`print("Implement this")`) — **there is no fastdtw import here.** Repointed the advice to
  `librosa.sequence.dtw` (consistent with NB5; works on the modern stack). *(Correction to an
  earlier note: NB8 did NOT need a fastdtw→librosa code migration — only the prose.)*
- No `is`-identity antipatterns in code (all such hits were comments/docstrings).
- **Verified:** used `--allow-errors` to surface ALL failures in one pass (only those two), then a
  final clean run → exit 0, valid, 0 errors. Runs **end-to-end for the first time** on the new
  stack; reproducible (Jon's set resolves deterministically, k-folds `seed=5`).
- **Size flag:** 6.7 MB → **14 MB** (13 MB of plot PNGs — the per-gesture/per-trial viz + experiment
  confusion matrices). The old 6.7 MB was a *broken partial* state. Committed the full working
  version; **candidate for a later media-slimming pass** (e.g. lower inline DPI, or cap displayed
  trials) if repo size matters.

### 9. GestureRecognizer-FeatureBased — ✅ done (2026-06-23)
Big notebook (162 cells). All the baseline-predicted bugs were real, plus the widget treatment:
- **pandas 2.0 — `df.append()` removed (2×):** modernized to the recommended idiom rather than a
  literal `pd.concat` swap:
  - cell 64 (builds the feature table row-by-row from `features` dicts) → collect into a `rows`
    list, then `df = pd.DataFrame(rows)` once (drops the old `if df is None` special-case).
  - cell 154 (accumulates 1-row test DataFrames) → `test_rows` list + a single `pd.concat`
    (also avoids the empty-DataFrame-concat `FutureWarning` and the O(n²) repeated-append).
  - (All other `.append(` in the notebook are list appends — left alone.)
- **numpy ≥1.24 — `np.bool` removed (2×):** cells 73, 75 (`dtype=np.bool`, `.astype(np.bool)`) → `bool`.
- **`%matplotlib notebook` → `widget` (2×):** cells 56, 58 are the **interactive 3D feature-combo
  plots** — exactly where rotate/zoom matters. Applied the NB3 pattern (guarded Colab ipympl +
  `enable_custom_widget_manager` on cell 56, `%matplotlib widget` on cell 58). Cell 61 already
  resets to `inline` afterward. Committed **png-only** for the widget figures (same static-render
  safety as NB3) and dropped a leftover transient `IntProgress` progress-bar widget output (cell 38)
  that had no png fallback (would've shown a broken widget placeholder).
- **Checked & clean:** all `confusion_matrix(...)` calls already pass `labels=` as a keyword (no
  NB8-style positional bug); `get_gesture_set_with_str` uses real names (Jon/Justin), no "Easy" issue.
- **Verified:** `--allow-errors` full run → **0 errors** across all 162 cells; final clean run → exit 0,
  valid. Runs **end-to-end for the first time** (the `df.append` at the feature table previously
  halted it). Size 8.8 → 14 MB (plot-heavy incl. a 1500×6600 110-axes figure); old size was a broken
  partial. Same media-slimming candidate note as NB8.

### 10. Feature Selection and Hyperparameter Tuning — ✅ done (2026-06-23)
113 cells. The baseline's `StringArray` blocker was real, but in a different spot than the
`np.bool`/`df.corr()` I first suspected — found the true cause by running `--allow-errors`:
- **Real blocker (`TypeError: unhashable type: 'StringArray'`, cells 36 & 41):** `df[filtered_df.columns.values] = scaler.fit_transform(...)`. In pandas 2.x the column index has **string dtype**, so `.columns.values` is a (StringArray) extension array, which is unhashable and fails as a multi-column setitem key. Fixed to `.columns.tolist()` (plain list of str).
- **Correctness bug (cell 41):** the MinMax-scaler section displayed `normalized_df_standard_scaler` (copy-paste from the StandardScaler cell) → now displays `normalized_df_minmax_scaler`.
- **`np.bool` (cells 4, 29, 31):** changed to builtin `bool` for clarity/consistency — **but note it was NOT actually broken**: numpy 2.0 *re-added* `np.bool` as an alias, so it works on our 2.4 stack. (Corrects the baseline's "np.bool removed" assumption — see cross-cutting note.)
- **Checked & clean:** `df.corr()` (cells 4, 25, 31) runs fine (the df is numeric there); `get_gesture_set` uses "Jon"; cell-13 appends are list appends.
- **Verified:** `--allow-errors` → 0 errors across 113 cells; clean run → exit 0, valid. Runs
  end-to-end (was blocked at cell 36 before). 1.4 → 2.5 MB (modest; seaborn heatmaps).

**🎉 All 10 notebooks are code-complete.** Remaining: the Colab-bootstrap sweep (NB4–10).

---

## Cross-cutting: minimal-diff policy for media-heavy notebooks (decided 2026-06-23)

The audio notebooks (4 Quantization, and likely 5 Comparing Signals, 6 Frequency Analysis,
7 StepTracker) embed **large base64 audio players** (`IPython.display.Audio` → `text/html`) and
dozens of plot PNGs — NB4 alone is ~13 MB. A full Restart-&-Run-All **verifies** correctness but,
if committed wholesale, churns 40+ binary PNG outputs and inflates the file (NB4: 12.8 → 19 MB).

**Policy:** for these heavy notebooks, *verify* with a full headless run, but *commit* a minimal
diff — restore the pristine outputs and refresh only the cells whose code actually changed. Keeps
diffs reviewable and repo size flat. (For the small tutorial notebooks 1–3 a full re-exec is fine.)

---

## Cross-cutting: interactive-plot strategy (decided 2026-06-23, applies to NBs 3 & 9)

`%matplotlib notebook` is **removed** in modern Jupyter — every occurrence becomes
`%matplotlib widget` (ipympl). Rationale and the rendering contract:
- **No static page can be truly interactive** (no live kernel) — interactivity is delivered only
  via the reader opening Binder/Colab/local Jupyter. `widget` is the only maintained backend that
  serves that path; the dead `notebook` magic gives nothing.
- **Commit png-only** for widget cells (strip `widget-view+json` + `text/html`, drop nb-level
  `metadata.widgets`). Static renderers → clean PNG; live re-run → full interactivity. Also keeps
  file size sane.
- **Colab** needs `from google.colab import output; output.enable_custom_widget_manager()` (and
  ipympl installed) for live interactivity. **Decided with Jon (option A):** bake a *guarded*
  Colab-enable block into the interactive notebooks so Colab "just works." Pattern (no-op off
  Colab), placed immediately before the `%matplotlib widget` magic:
  ```python
  import sys
  if 'google.colab' in sys.modules:
      %pip install -q ipympl
      from google.colab import output
      output.enable_custom_widget_manager()

  %matplotlib widget
  ```
  ✅ Implemented in NB 3. **TODO: apply the same block to NB 9** when it's modernized.

---

## Pass 3 — Quality / Pedagogy review (2026-06-23)

A deeper, lesson-by-lesson pass *beyond* "does it run": correctness of **explanations**
(not just code), readability/prose, code commenting + docstrings, and notebook-design best
practices (learning objectives, clean heading hierarchy, no instructor leftovers). Calibrated
to "polish + crisp additions, no bloat." Each notebook verified headless on the 3.12 venv and
committed separately. Most edits are **source-only** (preserving existing outputs); the one
exception re-executed because a real bug was fixed.

**Cross-cutting themes**
- **Built-in auto-TOC:** replaced stale `toc2`-extension advice (ShapeBased) — JupyterLab /
  Notebook 7 / Colab all auto-generate a table of contents from markdown headings. Fixed
  `#`→`###` heading skips (NB4/5/6: `# Main imports` → `## Main imports`) so that auto-TOC
  renders cleanly.
- **Removed instructor leftovers:** "Jon Sandbox / Outline / Notes" + visible all-done `# TODO`
  checklists (NB4 Quantization, NB6 Frequency, NB9 FeatureBased, NB10 FeatureSelection) and a
  dead `# Sandbox` block (NB5). Kept the *student-facing* sandbox in NB8 (explicitly "for you
  to play").
- **Learning objectives:** added a crisp "What you'll learn" list to the tutorials that lacked
  one (NB1–6, both StepTrackers). The gesture notebooks already had "Your TODOs"/overview goals.
- **Stale links refreshed:** numpy-1.12/1.18 and matplotlib-2.0.2/3.1.1/3.2.1 pinned doc links →
  `/stable/`; dead `docs.scipy.org/doc/numpy` → `numpy.org/doc/stable`; removed matplotlib
  `usage.html` (deleted upstream); fixed the anatomy.png image URL.
- **Import hygiene:** dropped unused imports (`random` in NB2/5/6; `from scipy import signal`
  in NB4); standardized the numpy import comment ("numerical/array library").

**Real bugs found & fixed (correctness, not just style)**
- **StepTracker-WithExampleSolution:** `NameError` — the pairing filter used
  `max_distance_between_peaks` but its definition was commented out. Restored it; the Solution
  now runs end-to-end (Pass 2 had only headless-verified the Exercises). Re-executed in place.
- **IntroToPython:** two Python-2 leftovers in teaching comments — `type(...)` prints
  `<class '...'>`, not `<type '...'>`, in Python 3.
- **IntroToNumPy:** wrong comment — `np.eye(4)` is a 4×4 (not 4×3) identity matrix.
- **IntroToMatplotlib:** discouraged `import matplotlib.pylab as plt` → `matplotlib.pyplot`;
  removed a confusing custom-marker demo that abused a list-comprehension for side effects.
- **FeatureBased:** `plot_feature_3d` z-axis defaulted to "Feature 2" and its docstring said
  "two…2-dimensional" (copy-paste from `plot_feature_2d`).
- **gesturerec/vis.py:** `plot_signals_aligned` used `math` and `shift_array`, neither imported
  or defined in the module (latent `NameError`). Added `import math` + a documented
  `shift_array` helper.

**Structural / de-duplication (Jon-approved scope)**
- **ShapeBased:** dropped the inline `plot_confusion_matrix` (a verbatim copy of
  `gesturerec.vis.plot_confusion_matrix`) and delegated to the shared helper.
- **Comparing Signals:** removed ~40 lines of dead code (the never-called
  `plot_signals_with_alignment`, superseded by `compare_and_plot_signals_with_alignment`).
- **Helper packages:** added module docstrings to every module; rewrote `utility.py` comments
  as docstrings; documented the public functions that lacked docstrings
  (`vis.plot_confusion_matrix`, `gesturestream.load`,
  `makelab.signal.plot_signal_and_magnitude_spectrum`, `makelab.audio.convert_to_mono`).

**Pedagogy deepened (crisp, no bloat)**
- ShapeBased: explained *why* DTW (warps the time axis to align gestures at different
  speeds/offsets) vs. point-by-point Euclidean; window-size guidance for the mean filter.
- FeatureBased: Nyquist-limit note on the "top frequency" feature; "why stratified k-fold"
  (imbalanced data); "StandardScaler vs MinMaxScaler — when to use which".
- FeatureSelection: feature-reduction trade-off caveat (cutting too far *lowers* accuracy);
  "GridSearchCV vs RandomizedSearchCV"; flagged that a polynomial-kernel `degree` is normally
  small (2–5).
- StepTracker: a "The approach" rationale for the Solution's smooth→find_peaks→filter pipeline.

**Deliberately NOT done** (judgment calls, flagged): did *not* over-comment the IntroToPython
quicksort (its prose sells it as concise); did *not* collapse `plot_feature_1d/2d/3d` into one
variadic helper (three clearly-named functions read better); did *not* alter GridSearch
`param_grid` values (avoids churning the results narrative — flagged via comment instead).

**Commits:** one per notebook/group on `signals-v2-refresh`
(`5647575` IntroToPython … `e825472` helper packages), preceded by `63a7c62`
(cosmetic notebook-metadata normalization, committed standalone to keep Pass 3 diffs clean).

---

## Pass 4 — Accessibility, engagement & up-front content (2026-06-24)

Closes the two issue-#116 dimensions Passes 1–3 left under-served — **Accessibility**
and **Presentation/engagement** — plus textbook-style lesson openers. Branch
`signals-v2-pass4` (off `master`); one commit per notebook/group.

**Scope decisions (with Jon):** do all of accessibility + engagement + operational
close-out; **keep the three intro tutorials as reference** (no new intro exercises);
**yes** to `ipywidgets` live sliders; lesson openers use a **blockquote note block**
matching the v2.0 textbook (`> **In this lesson, you will learn:**`).

### Accessibility
- **Alt text on every image.** Added descriptive `alt`/`![…]` text to all images that
  lacked it (Matplotlib anatomy, Wikipedia FFT + sampling + aliasing figures, the SVM
  hyperplane, the cross-correlation animation, …) and upgraded weak alt text
  ("Image of three-part split" → a real description). A repo-wide audit now reports
  **0 images missing/empty alt**. Markdown-only — no output churn.
- **Colorblind-safe colors (surgical).** Audited all `color=`/`cmap=` usage: the only
  genuine color-vision failure was the **blue + green** Correct/Incorrect bar pair in
  ShapeBased's `plot_bar_graph_average_scores_by_gesture` → recolored to the **Okabe-Ito
  blue + orange** pair. Everything else is single-hue (lone red spectra/annotations,
  lone blue bars) or red/blue, all already distinguishable, so they were left as-is to
  avoid churning megabytes of plots for no real benefit. Added a documented, reusable
  `COLORBLIND_SAFE` (Okabe-Ito) palette to `gesturerec/vis.py`.

### Engagement — live `ipywidgets` sliders
Added `@interact` sliders **additively** (the existing step-by-step walkthroughs are
themselves pedagogy and were kept). Pinned `ipywidgets==8.1.*` in `requirements.txt` /
`environment.yml`.
- **Quantization & Sampling:** a quantization **bit-depth** slider (re-quantize → replot →
  replay audio) and an **aliasing frequency** slider (fixed 50 Hz sampling).
- **Comparing Signals:** a cross-correlation **shift** slider ("slide b until it lines up
  with a; the dot product peaks at the best alignment").
- **Frequency Analysis:** a **frequency → magnitude-spectrum** slider after the worked 10 Hz
  FFT example (time- and frequency-domain views side by side).

**Static-render contract for `@interact` (important):** an `interact` cell stores its plot
in *widget state*, not the normal output bundle, so stripping widget state (needed for clean
GitHub/nbviewer rendering) would leave it blank. Fix: each slider cell **also calls its
function once at a representative default**, committing a normal static png (+ audio) as the
fallback; then `interact` is wired up for live use. Widget-view + nb-level `metadata.widgets`
are stripped (png-only), and a guarded Colab block
(`if 'google.colab' … enable_custom_widget_manager()`) precedes the first widget. Verified by
full headless `nbconvert --execute` (exit 0; only intentional `raises-exception` cells error),
then committed **minimal-diff** (pristine outputs on original cells; fresh widget-stripped
output only on the new slider cells).

### Up-front content (learning objectives + TOC)
Every lesson now opens with the **"In this lesson, you will learn:"** blockquote (restyling
the Pass-3 "What you'll learn" lists, or replacing redundant inline objectives). Longer
lessons also got an inline **Contents** TOC (anchor links to the existing headings) — the
live apps auto-generate a TOC sidebar, but static renders don't, so this helps GitHub/Colab
readers. **Correctness fix found along the way:** Comparing Signals' "## Dependencies" still
told students to `pip install fastdtw`, but Pass 2 migrated it to `librosa.sequence.dtw` —
corrected the prose.

### Operational
- **Colab badge validation** can only be confirmed after this branch merges to `master`
  (the bootstrap clones `master`). Deferred to post-merge — validate each of the 10 badges
  then.
- **Media-slimming** of the ~14 MB gesture notebooks: still optional/deferred.

**Commits:** `8b730ea` (foundation: ipywidgets pin + palette) → `97aa365` (ShapeBased),
one per notebook/group on `signals-v2-pass4`.
**Note:** an unrelated stale-stat refresh to FeatureBased (scikit-learn star/commit counts)
appeared in the working tree from outside this pass and was reverted — flag for Jon if a
stats refresh is wanted.

---

## Pass 5 — Test infrastructure (2026-06-24)

Tracking issue **#8**. The Pass 2–4 "does it run" verification was done by *manually*
running each notebook headless; this pass automates that and adds real unit coverage for
the helper packages. **Branch `signals-v2-tests` (off `master`). Zero changes inside the
notebooks** — all test code lives in `tests/` + `.github/workflows/`.

**Decisions (with Jon):** both layers (unit + notebook smoke); GitHub Actions CI; tiered
notebook execution (fast on PR, heavy gesture notebooks nightly); **execute-only** notebook
checks via **nbmake** (not output-diffing — random amplitudes / `random` xzoom / timing make
strict output comparison flaky).

**Added**
- **Unit tests (`tests/`, pytest):** `test_makelab_signal.py` (wave gen, `shift_array`,
  `calc_zero_crossings`, `map`/`remap`, top-N indices), `test_gesturerec_signalproc.py`
  (`compute_fft` half-spectrum + peak bin + scaling, `get_top_n_frequency_peaks`),
  `test_gesturerec_utility.py` (the `fulldatastream` exclusion guard, `extract_gesture_name`,
  `path_leaf`, subdirs), `test_gesturerec_data.py` (`SensorData` mag/rate + int64 cast, `Trial`
  CSV parse, `GestureSet.load` ordering **and the Windows `__` double-underscore quirk**),
  `test_gesturerec_experiments.py` (`TrialClassificationResult` n-best sort + `is_correct`,
  via stub trials). **26 tests, green locally.** Tiny synthetic fixture corpus under
  `tests/fixtures/TestGestures/` (avoids depending on the large real `GestureLogs/`).
  - *Caught a real contract detail while writing them:* `create_sine_wave(return_time=True)`
    returns `(time, sine_wave)` — order matters.
- **nbmake smoke tests:** verified locally that nbmake honors the `raises-exception` tags
  (NB2, NB5) and executes notebooks in their own dir (CWD-relative `./Logs` loads work).
- **`pyproject.toml`:** `[project.optional-dependencies].test` (pytest, nbmake, pytest-xdist,
  ipykernel) + `[tool.pytest.ini_options].testpaths = ["tests"]` (keeps bare `pytest` fast;
  notebook sweeps invoked explicitly by path).
- **CI:** `.github/workflows/ci.yml` (push/PR → `unit` job + `notebooks-fast` job over
  Tutorials + StepTracker) and `notebooks-nightly.yml` (`schedule` + `workflow_dispatch` →
  the slow GestureRecognizer notebooks). Both install pinned `requirements.txt` + `.[test]`
  on Python 3.12 and `libsndfile1` for librosa.
- **Docs:** updated `CLAUDE.md` ("no test suite, no CI" line + a Testing section) and
  `README.md` (Testing section + layout).

**Still to validate post-push:** the GitHub Actions runs themselves (Linux + pinned stack) —
CI can only be confirmed green after the branch lands and a PR triggers it.
