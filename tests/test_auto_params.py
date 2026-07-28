"""The numbers behind the settings that read 'Auto'.

A control left on 'Auto' carries no value of its own - the pyramid picks its
depth from the frame size while it runs - so the value it settles on is
recorded through `utils/auto_params.py`, logged, and written into the saved
result's XMP group. What is tested here is that the record reaches the caller:
that the pyramid reports the depth it used, that a tiled run reporting several
depths keeps all of them, and that a method run on its own with nobody
listening still works.

Run with:  python -m pytest tests/test_auto_params.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.pyramid import DEFAULT_LEVELS, pyramid_impl
from utils import auto_params


def _stack(height=96, width=96, frames=2):
    """A tiny stack; the depth resolves from its size, not from its content."""
    rng = np.random.default_rng(7)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for _ in range(frames)]


class TestRecording:
    def test_the_pyramid_reports_the_depth_auto_resolved_to(self):
        with auto_params.recording() as resolved:
            pyramid_impl(_stack(), levels=None)

        # 96 px halves past DEFAULT_LEVELS without reaching the 2 px floor, so
        # what ran is the method's own default rather than a size-limited depth.
        assert resolved[auto_params.PYRAMID_LEVELS] == [DEFAULT_LEVELS]

    def test_a_depth_the_frame_cannot_carry_is_reported_as_it_ran(self):
        with auto_params.recording() as resolved:
            pyramid_impl(_stack(16, 16), levels=8)

        # 16 px allows three halvings before a side would drop below 2.
        assert resolved[auto_params.PYRAMID_LEVELS] == [3]

    def test_a_method_run_with_nobody_listening_still_runs(self):
        result = pyramid_impl(_stack(), levels=2)
        assert result.shape == (96, 96, 3)


class TestWording:
    def test_one_value_reads_as_itself_and_several_as_a_span(self):
        assert auto_params.describe_value([5]) == "5"
        # Tiles are recorded as they finish, so the ends are found rather than
        # taken from the order they arrived in.
        assert auto_params.describe_value([5, 4]) == "4-5"
        assert auto_params.describe_value([]) == ""

    def test_repeated_values_are_kept_once(self):
        with auto_params.recording() as resolved:
            for _ in range(3):
                auto_params.record(auto_params.PYRAMID_LEVELS, 5)

        assert resolved[auto_params.PYRAMID_LEVELS] == [5]

    def test_the_log_line_names_the_setting_it_reports(self):
        assert auto_params.summarize(
            {auto_params.PYRAMID_LEVELS: [5]}) == "pyramid levels = 5"
        assert auto_params.summarize({}) == ""
        assert auto_params.summarize(None) == ""

    def test_a_value_that_went_unrecorded_reads_as_nothing(self):
        assert auto_params.value_of(None, auto_params.PYRAMID_LEVELS) is None
        assert auto_params.value_of({}, auto_params.PYRAMID_LEVELS) is None
        assert auto_params.value_of(
            {auto_params.PYRAMID_LEVELS: [6]}, auto_params.PYRAMID_LEVELS) == "6"
