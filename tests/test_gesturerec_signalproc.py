"""Unit tests for gesturerec.signalproc -- the shared FFT/peak primitives used by
the feature-based gesture notebooks.
"""
import numpy as np

import gesturerec.signalproc as sp


def _sine(freq, fs, secs):
    t = np.arange(fs * secs) / fs
    return np.sin(2 * np.pi * freq * t)


def test_compute_fft_returns_half_spectrum():
    fs = 100
    s = _sine(10, fs, 1)  # 100 samples
    freqs, amps = sp.compute_fft(s, fs)
    # Only the positive half of the spectrum is returned.
    assert len(freqs) == len(s) // 2
    assert len(amps) == len(s) // 2


def test_compute_fft_peak_sits_at_input_frequency():
    fs = 100
    s = _sine(10, fs, 1)  # bin spacing = fs/n = 1 Hz, so 10 Hz is an exact bin
    freqs, amps = sp.compute_fft(s, fs)
    peak_freq = freqs[np.argmax(amps)]
    assert peak_freq == 10


def test_compute_fft_amplitude_scaling():
    fs = 100
    s = _sine(10, fs, 1)
    _, amps_scaled = sp.compute_fft(s, fs, scale_amplitudes=True)
    _, amps_raw = sp.compute_fft(s, fs, scale_amplitudes=False)
    # Unit-amplitude sine -> scaled peak ~1.0; raw peak ~ N/2 = 50.
    assert np.isclose(np.max(amps_scaled), 1.0, atol=1e-2)
    assert np.isclose(np.max(amps_raw), len(s) / 2, atol=1.0)


def test_get_top_n_frequency_peaks_sorted_by_amplitude():
    fs = 200
    t = np.arange(fs) / fs  # 1 second
    # Two tones: 10 Hz at amplitude 1.0, 20 Hz at amplitude 0.5.
    s = 1.0 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    freqs, amps = sp.compute_fft(s, fs)
    peaks = sp.get_top_n_frequency_peaks(2, freqs, amps)
    assert len(peaks) == 2
    # Sorted by amplitude descending -> the 10 Hz tone (stronger) comes first.
    assert peaks[0][0] == 10
    assert peaks[1][0] == 20
    assert peaks[0][1] > peaks[1][1]
