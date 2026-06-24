"""Unit tests for makelab.audio.

Only convert_to_mono lives here so far. The key contract: a 2-D (samples, channels)
array is averaged across channels; a 1-D array is already mono and passes through.
"""
import numpy as np

import makelab.audio as ma


def test_convert_to_mono_averages_two_channels():
    # shape (num_samples, num_channels): each row is one sample's [left, right].
    stereo = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(ma.convert_to_mono(stereo), np.array([3.0, 7.0]))


def test_convert_to_mono_passes_through_mono():
    mono = np.array([1.0, 2.0, 3.0])
    # 1-D input is already mono and must be returned unchanged.
    np.testing.assert_array_equal(ma.convert_to_mono(mono), mono)


def test_convert_to_mono_averages_more_than_two_channels():
    # Guards the bug fix: averaging (not the old `sum / 2`) must be correct for any
    # channel count. Three channels of [3,6,9] average to 6, not (3+6+9)/2 = 9.
    three_channel = np.array([[3.0, 6.0, 9.0], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(ma.convert_to_mono(three_channel), np.array([6.0, 1.0]))
