"""Unit tests for gesturerec.data -- SensorData, Trial, and GestureSet.

Uses a tiny synthetic fixture corpus under tests/fixtures/TestGestures/ rather than
the large real GestureLogs/, so the suite stays fast and self-contained. The fixture
deliberately includes:
  - two "Shake" trials (to verify chronological trial ordering by end-time),
  - a "Midair Zorro _Z_" file exercising the Windows double-underscore filename quirk,
  - a *_fulldatastream_* file that must be excluded from per-trial loading.
"""
from pathlib import Path

import numpy as np
import pytest

import gesturerec.data as grdata

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "TestGestures"


def test_sensordata_magnitude_and_rate():
    time = np.array([1000, 1100, 1200])
    sensor_time = np.array([10, 20, 30])
    x = np.array([3, 0, 0])
    y = np.array([4, 0, 0])
    z = np.array([0, 0, 0])
    sd = grdata.SensorData("Accelerometer", time, sensor_time, x, y, z)

    # mag = sqrt(x^2 + y^2 + z^2); first row 3,4,0 -> 5.
    assert sd.mag[0] == pytest.approx(5.0)
    assert sd.length() == 3
    # length_in_secs = (1200-1000)/1000 = 0.2s; sampling_rate = 3 / 0.2 = 15 Hz.
    assert sd.length_in_secs == pytest.approx(0.2)
    assert sd.sampling_rate == pytest.approx(15.0)


def test_sensordata_casts_time_to_int64():
    sd = grdata.SensorData("Accelerometer",
                           np.array([1000, 2000]), np.array([1, 2]),
                           np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))
    # int64 cast is deliberate (Windows long is 32-bit) -- keep it.
    assert sd.time.dtype == np.int64


def test_trial_parses_csv_in_constructor():
    trial = grdata.Trial("Shake", 0, str(FIXTURE_DIR / "Shake_1000_3.csv"))
    assert trial.gesture_name == "Shake"
    assert trial.length() == 3
    assert trial.get_start_time() == 1000
    assert trial.get_end_time() == 1200
    # First data row is 3,4,0 -> magnitude 5.
    assert trial.accel.mag[0] == pytest.approx(5.0)


def test_gestureset_load_orders_trials_and_handles_windows_quirk():
    gs = grdata.GestureSet(str(FIXTURE_DIR))
    gs.load()

    # The *_fulldatastream_* file is excluded -> exactly two gestures.
    names = gs.get_gesture_names_sorted()
    assert "Shake" in names
    # The "Midair Zorro _Z_" filename (Windows replaced ' with _) must decode back
    # to the apostrophe form.
    assert "Midair Zorro 'Z'" in names
    assert gs.get_num_gestures() == 2

    # Two Shake trials, ordered chronologically by end-time (1000 then 2000).
    shake_trials = gs.get_trials("Shake")
    assert len(shake_trials) == 2
    assert shake_trials[0].get_end_time() == 1200
    assert shake_trials[1].get_end_time() == 2200
