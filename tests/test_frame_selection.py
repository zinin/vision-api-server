import numpy as np
import pytest

from frame_selection import (
    PTS_TOLERANCE,
    SelectedFrame,
    SelectionParams,
    blob_area,
    median_blob,
    prepare_frame,
    select_frames,
)


def _static(duration: float, fps: float = 10.0):
    """pts and zero blob for a static clip of `duration` seconds."""
    n = int(round(duration * fps))
    pts = [i / fps for i in range(n)]
    return pts, [0.0] * n


def _indices(selected):
    return [f.index for f in selected]


def _reasons(selected):
    return [f.reason for f in selected]


class TestSelectionParams:
    def test_max_frames_below_one_raises(self):
        with pytest.raises(ValueError):
            SelectionParams(max_frames=0)

    def test_zero_min_interval_raises(self):
        with pytest.raises(ValueError):
            SelectionParams(min_interval=0)


class TestFirstAndGrid:
    def test_single_frame(self):
        assert select_frames([0.0], [0.0], SelectionParams()) == [SelectedFrame(0, "first")]

    def test_static_clip_gives_first_and_grid(self):
        pts, blob = _static(10.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0))
        assert _indices(selected) == [0, 40, 80]
        assert _reasons(selected) == ["first", "grid", "grid"]

    def test_grid_step_is_max_of_gap_and_interval(self):
        pts, blob = _static(10.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, min_interval=5.0))
        assert _indices(selected) == [0, 50]

    def test_pts_jitter_within_tolerance_keeps_grid(self):
        pts, blob = _static(10.0)
        pts[40] = 4.0 - PTS_TOLERANCE / 2  # camera jitter: 3.9995 s
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0))
        assert _indices(selected) == [0, 40, 80]

    def test_empty_input_gives_empty_result(self):
        assert select_frames([], [], SelectionParams()) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            select_frames([0.0, 0.1], [0.0], SelectionParams())


class TestThinning:
    def test_long_clip_thins_grid_keeping_first_and_last(self):
        pts, blob = _static(40.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _indices(selected) == [0, 80, 160, 200, 280, 360]
        assert _reasons(selected) == ["first", "grid", "grid", "grid", "grid", "grid"]

    def test_max_frames_one_keeps_only_first(self):
        pts, blob = _static(40.0)
        assert select_frames(pts, blob, SelectionParams(max_frames=1)) == [SelectedFrame(0, "first")]


class TestMotionReserve:
    """The grid is held one frame short of `max_frames` so a peak always fits."""

    def test_long_clip_keeps_one_motion_slot(self):
        pts, blob = _static(40.0)
        blob[100] = 0.05
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert SelectedFrame(100, "motion") in selected
        assert len(selected) == 6

    def test_grid_filling_the_budget_exactly_still_keeps_a_motion_slot(self):
        # 20 s: the grid is exactly max_frames frames, so nothing is thinned
        pts, blob = _static(20.1)
        blob[70] = 0.05
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert SelectedFrame(70, "motion") in selected
        assert len(selected) == 6

    def test_only_one_slot_is_held_back(self):
        pts, blob = _static(40.0)
        for i in (50, 100, 150, 250, 350):
            blob[i] = 0.05
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _reasons(selected).count("motion") == 1
        assert len(selected) == 6

    def test_quiet_long_clip_keeps_the_full_grid(self):
        pts, blob = _static(40.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _indices(selected) == [0, 80, 160, 200, 280, 360]
        assert _reasons(selected) == ["first"] + ["grid"] * 5

    def test_unusable_peak_gives_the_full_grid_back(self):
        # the only candidate sits 0.5 s from a grid frame, closer than min_interval
        pts, blob = _static(40.0)
        blob[85] = 0.05
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _indices(selected) == [0, 80, 160, 200, 280, 360]
        assert "motion" not in _reasons(selected)

    def test_storm_on_a_long_clip_keeps_the_full_grid(self):
        pts, _ = _static(40.0)
        blob = [0.0] + [0.05] * (len(pts) - 1)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _indices(selected) == [0, 80, 160, 200, 280, 360]
        assert "motion" not in _reasons(selected)


class TestStorm:
    def test_storm_returns_grid_only(self):
        pts, _ = _static(10.0)
        blob = [0.0] + [0.05] * (len(pts) - 1)  # rain: every frame changes
        blob[25] = 0.5
        selected = select_frames(pts, blob, SelectionParams())
        assert _indices(selected) == [0, 40, 80]

    def test_below_storm_median_peaks_are_used(self):
        pts, blob = _static(10.0)
        blob[25] = 0.5
        selected = select_frames(pts, blob, SelectionParams())
        assert SelectedFrame(25, "motion") in selected


class TestMedianBlob:
    def test_empty_is_zero(self):
        assert median_blob([]) == 0.0

    def test_single_frame_is_zero(self):
        assert median_blob([0.0]) == 0.0

    def test_ignores_the_first_element(self):
        # median of [0.01, 0.03, 0.05]; including blob[0] would give 0.02
        assert median_blob([0.0, 0.01, 0.03, 0.05]) == pytest.approx(0.03)


class TestPeaks:
    def test_larger_peak_wins_the_last_slot(self):
        pts, blob = _static(10.0)
        blob[15] = 0.01
        blob[25] = 0.02
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 25, 40, 80]
        assert selected[1].reason == "motion"

    def test_peak_at_threshold_is_ignored(self):
        pts, blob = _static(10.0)
        blob[25] = 0.001  # equal to the threshold: not strictly above
        selected = select_frames(pts, blob, SelectionParams(motion_threshold=0.001))
        assert _indices(selected) == [0, 40, 80]

    def test_peak_too_close_to_grid_frame_rejected(self):
        pts, blob = _static(10.0)
        blob[41] = 0.02  # 0.1 s after grid frame 40
        blob[25] = 0.01
        selected = select_frames(pts, blob, SelectionParams(min_interval=1.0))
        assert _indices(selected) == [0, 25, 40, 80]

    def test_peaks_respect_min_interval_between_themselves(self):
        pts, blob = _static(10.0)
        blob[20] = 0.03
        blob[25] = 0.02  # 0.5 s after frame 20
        blob[30] = 0.01  # exactly 1.0 s after frame 20
        selected = select_frames(pts, blob, SelectionParams(min_interval=1.0))
        assert _indices(selected) == [0, 20, 30, 40, 80]

    def test_budget_limits_peaks(self):
        pts, blob = _static(10.0)
        blob[10] = 0.01
        blob[20] = 0.02
        blob[30] = 0.03
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 30, 40, 80]

    def test_equal_metric_prefers_lower_index(self):
        pts, blob = _static(10.0)
        blob[30] = 0.01
        blob[60] = 0.01
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 30, 40, 80]

    def test_result_sorted_by_index(self):
        pts, blob = _static(10.0)
        blob[60] = 0.02
        blob[20] = 0.01
        selected = select_frames(pts, blob, SelectionParams())
        assert _indices(selected) == sorted(_indices(selected))


class TestBlobArea:
    def test_identical_frames_give_zero(self):
        frame = prepare_frame(np.full((360, 640), 128, np.uint8))
        assert blob_area(frame, frame) == 0.0

    def test_square_gives_its_dilated_area(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[100:140, 100:140] = 255  # 40×40 square
        value = blob_area(prev, prepare_frame(cur_raw))
        # 40×40 = 0.0069 of the frame before blur and dilation, at most 50×50 = 0.0109 after
        assert 0.0069 <= value <= 0.0109

    def test_noise_below_threshold_gives_zero(self):
        rng = np.random.default_rng(0)
        prev = prepare_frame(np.full((360, 640), 128, np.uint8))
        cur = prepare_frame(rng.integers(120, 137, size=(360, 640), dtype=np.uint8))
        assert blob_area(prev, cur) == 0.0

    def test_largest_component_only(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[10:20, 10:20] = 255      # small blob, 10×10
        cur_raw[200:260, 300:360] = 255  # big blob, 60×60
        value = blob_area(prev, prepare_frame(cur_raw))
        assert 60 * 60 / (640 * 360) <= value <= 70 * 70 / (640 * 360)

    def test_returns_python_float(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[50:90, 50:90] = 255
        assert type(blob_area(prev, prepare_frame(cur_raw))) is float
