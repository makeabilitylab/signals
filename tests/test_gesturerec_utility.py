"""Unit tests for gesturerec.utility -- file-handling helpers for loading gesture logs."""
import gesturerec.utility as util


def test_find_csv_filenames_excludes_fulldatastream(tmp_path):
    # The deliberate guard: per-trial loading must not pick up the continuous
    # full-stream file (see utility.find_csv_filenames docstring).
    (tmp_path / "Shake_1000_3.csv").write_text("x")
    (tmp_path / "Wave_2000_4.csv").write_text("x")
    (tmp_path / "armData_fulldatastream_9_9.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    found = util.find_csv_filenames(str(tmp_path))

    assert set(found) == {"Shake_1000_3.csv", "Wave_2000_4.csv"}


def test_extract_gesture_name_takes_text_before_first_underscore():
    assert util.extract_gesture_name("Shake_1556730840228_206.csv") == "Shake"


def test_path_leaf_returns_final_component():
    assert util.path_leaf("/a/b/c.csv") == "c.csv"
    # Trailing slash -> the leaf is the final directory name.
    assert util.path_leaf("/a/b/") == "b"


def test_get_immediate_subdirectories(tmp_path):
    (tmp_path / "JonGestures").mkdir()
    (tmp_path / "JustinGestures").mkdir()
    (tmp_path / "afile.csv").write_text("x")

    subdirs = util.get_immediate_subdirectories(str(tmp_path))

    assert set(subdirs) == {"JonGestures", "JustinGestures"}
