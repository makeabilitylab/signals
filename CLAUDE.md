# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Educational materials for applied signal processing and time-series classification in HCI / ubiquitous computing (a University of Washington course, part of the Makeability Lab "physcomp" curriculum). The artifacts are **Jupyter notebooks** backed by small supporting Python packages. There is no build, no test suite, no CI — work is done interactively in notebooks.

Top-level layout:
- `Tutorials/` — standalone teaching notebooks (NumPy, Matplotlib, Python, and signals: sampling/quantization, frequency analysis, comparing signals). Supported by the `makelab/` package.
- `Projects/GestureRecognizer/` — accelerometer gesture classification. The most substantial project; supported by the `gesturerec/` package.
- `Projects/StepTracker/` — step-counting from accelerometer data (exercise + example-solution notebooks).
- Many notebooks have a `*-Private.ipynb`, `*-Exercises.ipynb` / `*-WithExampleSolution.ipynb` split: the public/exercise version has cells for students to fill in; the private/solution version is the instructor answer key. `*-Private.ipynb` files are gitignored.

## Running

Setup (Python 3.12; see `README.md`): `python -m venv .venv` → activate → `pip install -r requirements.txt`. This also installs the repo itself editable (`pip install -e .`), which is what makes the helper packages importable.

The two helper packages (`makelab` in `Tutorials/makelab/`, `gesturerec` in `Projects/GestureRecognizer/gesturerec/`) are mapped to top-level import names by the root `pyproject.toml`, so after the editable install they import **from any working directory**, no `sys.path` hacks:

```
import gesturerec.data as grdata
import makelab.signal
```

**However, data loading is still CWD-relative.** Notebooks use paths like `./GestureLogs`, `./Logs`, and `data/audio/...`, so a notebook's kernel must run with its own folder as the working directory (JupyterLab/VS Code do this automatically; `nbconvert --execute` does too). Don't "fix" those into absolute paths — they must stay repo-relative to work in Colab/Binder (textbook contract).

Pinned dependencies live in `requirements.txt` / `environment.yml`: NumPy, SciPy, scikit-learn, pandas, Matplotlib, seaborn, librosa, JupyterLab. The `.venv/` and a registered `signals` Jupyter kernel are local-only (gitignored).

Notebook files contain non-ASCII characters (arrows, curly quotes). When parsing a `.ipynb` with a script, open as UTF-8 and run Python with `PYTHONUTF8=1` — the default Windows cp1252 codec will raise `UnicodeDecodeError`.

## gesturerec architecture (Projects/GestureRecognizer)

This package abstracts the data loading and experiment bookkeeping so notebook code can focus on the classification algorithm itself. The data flow is:

**Raw logs → data structures → classifier → results.**

Data structures (`gesturerec/data.py`):
- `SensorData` — one sensor's signal as parallel NumPy arrays (`time`, `sensor_time`, `x`, `y`, `z`, plus computed `mag`). Holds `*_p` placeholders for **processed** (filtered) versions produced by notebook preprocessing. Also exposes `sampling_rate` and `length_in_secs`.
- `Trial` — one recorded gesture (a single CSV). **The constructor parses the CSV** (`np.genfromtxt`) and builds the `SensorData`. Each trial knows its ground-truth `gesture_name`.
- `GestureSet` — all trials for one participant, loaded from a directory of CSVs. Usage is always **construct → `load()` → preprocess in the notebook**. Filenames encode `GestureName_endTimeMs_numRows.csv`; trials are ordered by end-timestamp. Note the Windows quirk: Windows replaced `'` in gesture names with `_`, producing `__` double-underscore filenames, which the parser special-cases.
- Module-level `get_gesture_set*` / `get_all_gesture_sets*` helpers operate on a `map` of gesture-set-name → `GestureSet` (one entry per participant directory under `GestureLogs/`).

Experiment bookkeeping (`gesturerec/experiments.py`):
- `TrialClassificationResult` — result of classifying one test trial: holds the **n-best list** of (template_trial, score) tuples sorted by score, the chosen `closest_trial`, and `is_correct`.
- `ClassificationResults` — aggregates results for a whole experiment: accuracy, per-gesture correctness, confusion matrix (via scikit-learn), n-best-list performance curves, timing. This is the object a matching algorithm returns.
- `Experiments` — a collection of `ClassificationResults`, sortable by accuracy / compute time, for comparing approaches.

Two classification paradigms are taught (each in its own notebook), both running over `gesturerec` data structures:
- **Shape-based** (`GestureRecognizer-ShapeBased.ipynb`) — template/similarity matching between time-series (e.g. Euclidean / DTW-style distance) on pre-segmented trials.
- **Feature-based** (`GestureRecognizer-FeatureBased.ipynb`, `Feature Selection and Hyperparameter Tuning.ipynb`) — extract features (mean, variance, dominant frequency, …) and train a scikit-learn model (SVM in a `StandardScaler`→classifier `Pipeline`), evaluated with `StratifiedKFold` cross-validation.

Other modules:
- `gesturerec/signalproc.py` — `compute_fft` and `get_top_n_frequency_peaks`, the shared FFT/feature primitives.
- `gesturerec/gesturestream.py` — `GestureStream` (the full continuous `fulldatastream.csv` for *offline* segmentation work, vs. the pre-segmented per-trial CSVs) and `Event` (a segment cut from that stream). `GestureStream.load()` auto-cleans malformed rows, renaming the original to `*_old_<ms>.csv` and rewriting a cleaned file.
- `gesturerec/vis.py` — plotting helpers (`plot_confusion_matrix`, signal plots).
- `gesturerec/utility.py` — file-handling helpers; `find_csv_filenames` deliberately **excludes** files containing `fulldatastream` so trial loading and stream loading don't collide.

Two data corpora exist: `GestureLogs/` (per-participant `*Gestures/` dirs, the primary set) and `ADXL335GestureLogs/` (an alternate sensor). Notebooks switch between them by changing `root_gesture_log_path`.

## makelab (Tutorials)

`makelab/signal.py` and `makelab/audio.py` provide signal-generation and analysis helpers (sine/cosine generators, FFT, distance metrics, audio via librosa) used by the tutorial notebooks. Same import rule: run notebooks from `Tutorials/`.

## Conventions

- Code is written to be **read by students** — verbose docstrings, inline citations to Stack Overflow / library docs, and explanatory comments are intentional. Match that style; don't strip the pedagogy.
- Loaders are tolerant of messy real-world logs (anomalous timestamps, empty columns, OS-specific filename mangling). Preserve those guards when touching parsing code.
- `int64` casts in `SensorData` are deliberate (Windows `long` is 32-bit) — keep them.
