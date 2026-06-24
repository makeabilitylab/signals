"""Unit tests for the pure signal-generation/analysis helpers in makelab.signal.

These pin the numeric contracts the tutorial notebooks rely on (wave generation,
array shifting, zero-crossing counting, value remapping). Plotting helpers are not
tested here -- they render figures rather than return data.
"""
import numpy as np
import pytest

import makelab.signal as ms


def test_create_sine_wave_default_length_is_one_period():
    # With total_time_in_secs=None the helper returns exactly one period:
    # one period = 1/freq seconds -> (1/freq) * sampling_rate samples.
    s = ms.create_sine_wave(freq=2, sampling_rate=8)  # 0.5s * 8Hz = 4 samples
    assert len(s) == 4


def test_create_sine_wave_length_and_values():
    fs = 10
    # return_time=True yields (time, sine_wave), in that order.
    t, s = ms.create_sine_wave(freq=1, sampling_rate=fs, total_time_in_secs=2,
                               return_time=True)
    assert len(s) == 2 * fs
    # Values must match sin(2*pi*f*t) at each sample time.
    np.testing.assert_allclose(s, np.sin(2 * np.pi * 1 * t), atol=1e-12)


def test_create_cos_wave_starts_at_one():
    s = ms.create_cos_wave(freq=1, sampling_rate=100, total_time_in_secs=1)
    assert s[0] == pytest.approx(1.0)


def test_shift_array_positive_shifts_right_and_fills_front():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ms.shift_array(arr, 2, fill_value=np.nan)
    np.testing.assert_array_equal(result, np.array([np.nan, np.nan, 1.0, 2.0, 3.0]))


def test_shift_array_negative_shifts_left_and_fills_back():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ms.shift_array(arr, -2, fill_value=np.nan)
    np.testing.assert_array_equal(result, np.array([3.0, 4.0, 5.0, np.nan, np.nan]))


def test_shift_array_zero_is_identity():
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ms.shift_array(arr, 0), arr)


def test_map_and_remap_linear_interpolation():
    assert ms.map(5, 0, 10, 0, 100) == pytest.approx(50)
    assert ms.remap(0, 0, 10, 0, 100) == pytest.approx(0)
    assert ms.remap(10, 0, 10, 0, 100) == pytest.approx(100)


def test_get_top_n_frequency_indices_sorted_returns_largest_descending():
    amplitudes = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    freqs = np.arange(len(amplitudes))
    # Top 2 amplitudes are 8 (idx 3) then 5 (idx 1), in descending order.
    ind = ms.get_top_n_frequency_indices_sorted(2, freqs, amplitudes)
    assert list(ind) == [3, 1]


def test_calc_zero_crossings_alternating_signal():
    # +1,-1,+1,-1 changes sign at indices 1, 2, 3 -> three crossings.
    s = np.array([1.0, -1.0, 1.0, -1.0])
    assert ms.calc_zero_crossings(s) == [1, 2, 3]


def test_calc_zero_crossings_min_gap_thins_results():
    s = np.array([1.0, -1.0, 1.0, -1.0])
    # A min_gap larger than the spacing should drop the closely-spaced crossings.
    assert ms.calc_zero_crossings(s, min_gap=5) == [1]
