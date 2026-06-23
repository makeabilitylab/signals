# Signals: Applied Signal Processing & ML Notebooks

Hands-on Jupyter notebooks for **applied signal processing and time-series
classification** in human–computer interaction and ubiquitous computing — covering
sampling and quantization, comparing signals in the time domain, frequency analysis
(DFT/FFT/STFT), and two complete accelerometer projects (step tracking and gesture
recognition).

These notebooks are the **source of truth** for the *Signals* module of the
[Physical Computing interactive textbook](https://makeabilitylab.github.io/physcomp/signals/).
The textbook links to and renders them; this repository is where they are maintained,
mirroring how the textbook treats [`makeabilitylab/arduino`](https://github.com/makeabilitylab/arduino).
They were developed for, and refined across several offerings of, a graduate
ubiquitous-computing course at the University of Washington.

## Lessons

Run the notebooks in this order — each builds on the previous one. Launch in the
cloud (no install) with the badges, or run locally (see [Setup](#setup)).

| # | Lesson | Notebook | Cloud |
|---|--------|----------|-------|
| 1 | Intro to Python | [`Tutorials/IntroToPython.ipynb`](Tutorials/IntroToPython.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/IntroToPython.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/IntroToPython.ipynb) |
| 2 | Intro to NumPy | [`Tutorials/IntroToNumPy.ipynb`](Tutorials/IntroToNumPy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/IntroToNumPy.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/IntroToNumPy.ipynb) |
| 3 | Intro to Matplotlib | [`Tutorials/IntroToMatplotlib.ipynb`](Tutorials/IntroToMatplotlib.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/IntroToMatplotlib.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/IntroToMatplotlib.ipynb) |
| 4 | Quantization & Sampling | [`Tutorials/Signals - Quantization and Sampling.ipynb`](Tutorials/Signals%20-%20Quantization%20and%20Sampling.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/Signals%20-%20Quantization%20and%20Sampling.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/Signals%20-%20Quantization%20and%20Sampling.ipynb) |
| 5 | Comparing Signals (time domain) | [`Tutorials/Signals - Comparing Signals.ipynb`](Tutorials/Signals%20-%20Comparing%20Signals.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/Signals%20-%20Comparing%20Signals.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/Signals%20-%20Comparing%20Signals.ipynb) |
| 6 | Frequency Analysis (DFT/FFT/STFT) | [`Tutorials/Signals - Frequency Analysis.ipynb`](Tutorials/Signals%20-%20Frequency%20Analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/Signals%20-%20Frequency%20Analysis.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Tutorials/Signals%20-%20Frequency%20Analysis.ipynb) |
| 7 | Step Tracker | [`Projects/StepTracker/StepTracker-Exercises.ipynb`](Projects/StepTracker/StepTracker-Exercises.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Projects/StepTracker/StepTracker-Exercises.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Projects/StepTracker/StepTracker-Exercises.ipynb) |
| 8 | Gesture Recognition: Shape-Based | [`Projects/GestureRecognizer/GestureRecognizer-ShapeBased.ipynb`](Projects/GestureRecognizer/GestureRecognizer-ShapeBased.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Projects/GestureRecognizer/GestureRecognizer-ShapeBased.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Projects/GestureRecognizer/GestureRecognizer-ShapeBased.ipynb) |
| 9 | Gesture Recognition: Feature-Based | [`Projects/GestureRecognizer/GestureRecognizer-FeatureBased.ipynb`](Projects/GestureRecognizer/GestureRecognizer-FeatureBased.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Projects/GestureRecognizer/GestureRecognizer-FeatureBased.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Projects/GestureRecognizer/GestureRecognizer-FeatureBased.ipynb) |
| 10 | Feature Selection & Hyperparameter Tuning | [`Projects/GestureRecognizer/Feature Selection and Hyperparameter Tuning.ipynb`](Projects/GestureRecognizer/Feature%20Selection%20and%20Hyperparameter%20Tuning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Projects/GestureRecognizer/Feature%20Selection%20and%20Hyperparameter%20Tuning.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/makeabilitylab/signals/master?labpath=Projects/GestureRecognizer/Feature%20Selection%20and%20Hyperparameter%20Tuning.ipynb) |

> **Note:** the Colab badges load a notebook directly from GitHub but do **not** clone
> the repo's data or helper packages. A small "Colab setup" bootstrap cell (added per
> notebook in the lesson-modernization pass) handles that. Binder builds a full
> environment from `environment.yml`/`requirements.txt` and checks out the whole repo,
> so data and imports work there without extra steps.

## Setup

Requires **Python 3.12**. Two supported paths — pick one.

### Option A — venv + pip (no conda needed)

```bash
git clone https://github.com/makeabilitylab/signals.git
cd signals
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
jupyter lab
```

### Option B — conda

```bash
git clone https://github.com/makeabilitylab/signals.git
cd signals
conda env create -f environment.yml
conda activate signals
jupyter lab
```

Both paths also install the local helper packages (`makelab`, `gesturerec`) in
*editable* mode via `pip install -e .`, so the notebooks import them from anywhere
with no `sys.path` hacks:

```python
import makelab.signal          # used by the Tutorials notebooks
import gesturerec.data         # used by the GestureRecognizer notebooks
```

## Repository layout

```
.
├── Tutorials/                    # Lessons 1–6 (Python/NumPy/Matplotlib + signals)
│   ├── makelab/                  # shared signal + audio helpers (importable package)
│   └── data/audio/               # audio clips used by Quantization & Sampling
├── Projects/
│   ├── StepTracker/              # Lesson 7  (+ Logs/ accelerometer step data)
│   └── GestureRecognizer/        # Lessons 8–10
│       ├── gesturerec/           # data structures + experiment scaffolding (package)
│       ├── GestureLogs/          # per-participant gesture training data
│       └── ADXL335GestureLogs/   # alternate-sensor gesture data
├── pyproject.toml                # packaging for makelab + gesturerec
├── requirements.txt              # pinned pip environment
└── environment.yml               # pinned conda environment
```

Some notebooks ship as an exercise/solution pair (`*-Exercises.ipynb` for students,
`*-WithExampleSolution.ipynb` as the answer key). Instructor-only solution keys
(`*-Private.ipynb`) are intentionally not published.

## License

See [LICENSE](LICENSE).
