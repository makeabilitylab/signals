# makelab

Shared signal-processing and audio helpers for the Makeability Lab **signals tutorial
notebooks** (`Tutorials/`). This package keeps the notebooks focused on the *concepts* by
handling the repetitive bits — generating test signals, finding zero crossings, and building
the multi-axis time/frequency plots used throughout the tutorials.

> **Teaching code, on purpose.** This package is meant to be *read* by students. The
> implementations favor clarity over cleverness, cite their sources, and frequently show the
> alternative way a result could be computed. Where a function duplicates something a production
> library already does (e.g. `numpy.interp`, `scipy.ndimage.shift`, `librosa.to_mono`), a comment
> in the source points to that equivalent and notes any behavioral difference. Keep that style if
> you edit it.

## Modules

- **`makelab.signal`** — signal generation, analysis, and plotting.
- **`makelab.audio`** — audio post-processing (currently just stereo→mono). Audio *loading*
  (librosa/soundfile) happens in the notebooks; this module only works on the arrays they produce.

## Importing

The repo is installed editable (`pip install -e .`) and `pyproject.toml` maps this folder to the
top-level `makelab` import name, so it imports **from any working directory** — no `sys.path`
hacks:

```python
import makelab.signal
import makelab.audio
```

Note that **data/audio loading in the notebooks is still CWD-relative** (paths like
`data/audio/...`), so a notebook's kernel must run from its own folder. That's a notebook
concern, not a `makelab` one — these helpers take in-memory arrays.

## Public API (quick reference)

### Signal generation (`makelab.signal`)
| Function | Purpose |
| --- | --- |
| `create_sine_wave(freq, sampling_rate, total_time_in_secs=None, return_time=False)` | One sine wave (one period if no length given). |
| `create_cos_wave(...)` | Same, but cosine (starts at amplitude 1). |
| `create_sine_waves(freqs, ...)` | A list of `(freq, wave)` tuples, one per frequency. |
| `create_composite_sine_wave(freqs, ..., amplitudes=None)` | Sum of sine waves (i.e. a chord / complex tone). |
| `create_sine_wave_sequence(freqs, ..., starting/ending_amplitudes)` | Notes played back-to-back with amplitude ramps. |

### Analysis (`makelab.signal`)
| Function | Purpose |
| --- | --- |
| `shift_array(arr, shift_amount, fill_value=np.nan)` | Shift left/right, filling the vacated end. |
| `calc_zero_crossings(s, min_gap=None)` | Sample indices where `s` crosses zero (`min_gap` thins them). |
| `get_top_n_frequency_indices_sorted(n, freqs, amplitudes)` | Indices of the `n` largest amplitudes, high→low. |
| `remap(val, start1, stop1, start2, stop2)` / `map(...)` | Linear range remap (Arduino-style; `map` is an alias). |
| `get_random_xzoom(signal_length, fraction_of_length)` | A random `(start, end)` sample window for zoom plots. |

### Plotting (`makelab.signal`)
All plotting helpers return their matplotlib `(fig, axes, …)` objects and add a secondary
time-based x-axis on top of the sample-based axis.

| Function | Purpose |
| --- | --- |
| `plot_signal(s, sampling_rate, title=None, xlim_zoom=None)` | Waveform; adds a zoomed panel if `xlim_zoom` given. |
| `plot_audio(s, sampling_rate, quantization_bits=16, ...)` | `plot_signal` with a bit-depth-aware default title. |
| `plot_signal_to_axes(ax, s, sampling_rate, ...)` | Plot onto an existing axes (building block). |
| `plot_sampling_demonstration(total_time_in_secs, real_world_freqs, ...)` | Stem-plot demo of sampling/aliasing. |
| `plot_signal_and_magnitude_spectrum(t, s, sampling_rate, ...)` | Time domain beside its magnitude spectrum. |
| `plot_spectrogram(s, sampling_rate, ...)` / `plot_spectrogram_to_axes(...)` | Spectrogram (full + zoom). |
| `plot_signal_and_spectrogram(s, sampling_rate, quantization_bits, xlim_zoom, ...)` | Waveform + spectrogram, both with zoom. |

### Audio (`makelab.audio`)
| Function | Purpose |
| --- | --- |
| `convert_to_mono(audio_data)` | Average a `(samples, channels)` array to mono; 1-D input passes through. |

## Tests

Pure (non-plotting) helpers are unit-tested under the repo's top-level `tests/`
(`test_makelab_signal.py`, `test_makelab_audio.py`); plotting helpers are exercised by the
`nbmake` notebook smoke tests. Run the unit tests with:

```bash
pip install -e ".[test]"
pytest tests/
```

If you add or change a helper here, add/adjust a unit test to match (see the repo's **Testing**
section in `CLAUDE.md`).
