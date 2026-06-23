# Modernization Notes — `makeabilitylab/signals` v2.0

Report back to the `physcomp` textbook side (tracking issue **#116**). Living document;
updated as the lesson-by-lesson pass proceeds.

**Status:** Pass 1 (repo infrastructure) complete. Pass 2 (lesson-by-lesson) not started.
**Date:** 2026-06-23.

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
  `makelab`/`gesturerec` packages. Each notebook needs a small "Colab setup" bootstrap
  cell (clone repo + `pip install -e .` + `cd` into the lesson folder). That is a **Pass 2**
  per-notebook edit.
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
- `np.bool` — **5 occurrences** (notebooks 9 and 10). Removed in NumPy ≥1.24 → `bool`.
- `%matplotlib notebook` — **6 occurrences** (notebooks 3 and 9). Broken in Notebook 7 /
  JupyterLab / Colab → `%matplotlib inline` (or `widget` where interaction is needed).
- `sklearn.externals` — searched, **not** present (no action).
- Intentional `raises-exception` cells (notebooks 2, 5) need tagging so Restart-&-Run-All
  completes for the rendered reading copies.

---

## Per-notebook status (Pass 2)

_Not started — to be filled in as each notebook is modernized (1 → 10)._
