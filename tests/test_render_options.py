"""The vocabulary a saved result uses to describe its own render.

`core/render_options.describe` is the single place both render paths - the
interactive one and the batch worker - turn their settings into the properties
of the XMP OpenFocus group. Two things matter and are tested here:

1. A parameter is only quoted when the method it belongs to actually reads it,
   so nothing in the file suggests a knob mattered when it did not
   (`TestParameterRelevance`).
2. A stage that did not run says so rather than going missing, and the wording
   of what did run is stable (`TestWording`).

Run with:  python -m pytest tests/test_render_options.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import render_options
from utils import auto_params, bitdepth


@pytest.fixture(autouse=True)
def restore_mode():
    """Every test leaves the global depth mode as it found it."""
    previous = bitdepth.get_mode()
    yield
    bitdepth.set_mode(previous)


class TestParameterRelevance:
    @pytest.mark.parametrize("algorithm", sorted(render_options.KERNEL_METHODS))
    def test_kernel_driven_methods_quote_the_kernel(self, algorithm):
        options = render_options.describe(algorithm=algorithm, kernel_size=31)
        assert options["KernelSize"] == "31 px"

    @pytest.mark.parametrize("algorithm", ["dtcwt", "stackmffv4"])
    def test_methods_that_ignore_the_kernel_do_not_quote_it(self, algorithm):
        options = render_options.describe(algorithm=algorithm, kernel_size=31)
        assert "KernelSize" not in options

    def test_halo_is_only_reported_for_depth_map(self):
        assert render_options.describe(
            algorithm="depthmap_max", halo_radius=12)["HaloRadius"] == "12 px"
        # Off is stated rather than omitted: the method reads it either way.
        assert render_options.describe(algorithm="depthmap_average")["HaloRadius"] == "Off"
        assert "HaloRadius" not in render_options.describe(algorithm="dct")

    def test_batch_size_is_only_reported_for_stackmff(self):
        assert render_options.describe(
            algorithm="stackmffv4", stackmffv4_batch_size=4)["StackMffBatchSize"] == "4"
        assert "StackMffBatchSize" not in render_options.describe(
            algorithm="pyramid", stackmffv4_batch_size=4)

    def test_a_batch_the_card_could_not_hold_says_what_ran(self):
        options = render_options.describe(
            algorithm="stackmffv4", stackmffv4_batch_size=4,
            resolved_auto={auto_params.STACKMFF_BATCH_SIZE: [2, 1]},
        )
        assert options["StackMffBatchSize"] == "4 (1-2 applied)"

    def test_tile_workers_are_only_reported_by_a_render_that_tiled(self):
        options = render_options.describe(
            tile_enabled=True, tile_block_size=512, tile_overlap=64,
            resolved_auto={auto_params.TILE_WORKERS: [3]},
        )
        assert options["TilingWorkers"] == "3 tiles at once"
        # Tiling switched on but never reached: the frame stayed in one piece,
        # so there is no worker count to report.
        assert "TilingWorkers" not in render_options.describe(
            tile_enabled=True, tile_block_size=512, tile_overlap=64)

    def test_pyramid_tuning_is_only_reported_for_the_pyramid(self):
        options = render_options.describe(algorithm="pyramid")
        # Quoted even when nothing was chosen: what ran is the method default,
        # and an absent property would read as "this build had no such control".
        assert set(options) >= {
            "PyramidLevels", "PyramidSelectivity", "PyramidCoherence",
            "PyramidBaseWeighting", "PyramidNoiseGate", "PyramidEnvelopeClip",
        }
        assert not any(name.startswith("Pyramid") for name in
                       render_options.describe(algorithm="dct", pyramid_levels=4))

    def test_registration_details_are_dropped_when_nothing_was_aligned(self):
        options = render_options.describe(reference_mode="middle", reg_downscale_width=1024)
        assert options["Registration"] == "Off"
        assert "ReferenceFrame" not in options
        assert "RegistrationDownscale" not in options

    def test_ecc_parallel_is_reported_only_when_ecc_ran(self):
        with_ecc = render_options.describe(align_ecc=True, ecc_parallel=True)
        assert with_ecc["EccParallel"] == "On"
        assert "EccParallel" not in render_options.describe(
            align_homography=True, ecc_parallel=True)


class TestWording:
    def test_registration_stages_read_in_pipeline_order(self):
        options = render_options.describe(
            align_scale=True, align_homography=True, align_ecc=True, reference_mode="middle"
        )
        assert options["Registration"] == "Scale + Homography + ECC"
        assert options["ReferenceFrame"] == "Middle"
        assert options["RegistrationDownscale"] == "Full resolution"

    def test_a_subset_render_says_how_many_frames_took_part(self):
        assert render_options.describe(
            source_count=12, total_count=20)["SourceImages"] == "12 of 20 frames"
        assert render_options.describe(
            source_count=20, total_count=20)["SourceImages"] == "20 frames"

    def test_registration_only_render_names_no_method(self):
        options = render_options.describe(algorithm=None, align_ecc=True)
        assert options["FusionMethod"] == "None (registration only)"
        # A refinement stage cannot run without fusion, so it is not claimed.
        assert "IfcnnRefinement" not in options

    def test_contrast_wording_covers_both_methods_and_off(self):
        assert render_options.describe_contrast("auto", 50) == "Auto, strength 50%"
        assert render_options.describe_contrast("clahe", 30) == "Local (CLAHE), strength 30%"
        assert render_options.describe_contrast("off", 50) == "Off"
        # Strength 0 is off however the method reads.
        assert render_options.describe_contrast("auto", 0) == "Off"

    def test_region_reports_how_a_roi_render_was_saved(self):
        assert render_options.describe()["Region"] == "Full frame"
        assert render_options.describe(roi_mode="crop")["Region"] == \
            "Selected region, saved cropped"
        assert render_options.describe(roi_mode="paste", roi_base_index=3)["Region"] == \
            "Selected region, blended onto frame 4"

    def test_tiling_reports_its_geometry(self):
        options = render_options.describe(
            tile_enabled=True, tile_block_size=512, tile_overlap=64, tile_threshold=2048
        )
        assert options["Tiling"] == "512 px blocks, 64 px overlap, above 2048 px"
        assert render_options.describe(tile_enabled=False)["Tiling"] == "Off"

    def test_bit_depth_reports_the_mode_and_the_result(self):
        bitdepth.set_mode(bitdepth.MODE_AUTO)
        options = render_options.describe(result_dtype=np.uint16)
        assert options["BitDepth"] == "Auto (16-bit result)"

        bitdepth.set_mode(bitdepth.MODE_8)
        assert render_options.describe()["BitDepth"] == "Forced 8-bit"

    def test_pyramid_presets_read_as_a_name_and_the_number_behind_it(self):
        options = render_options.describe(
            algorithm="pyramid", kernel_size=5, pyramid_levels=6,
            pyramid_selectivity=32.0, pyramid_coherence=0.25, pyramid_base=0.0,
            pyramid_noise_gate=False, pyramid_envelope=True,
        )
        assert options["KernelSize"] == "5 px"
        assert options["PyramidLevels"] == "6"
        assert options["PyramidSelectivity"] == "Strict (32)"
        assert options["PyramidCoherence"] == "Light (0.25)"
        assert options["PyramidBaseWeighting"] == "Mean (0)"
        assert options["PyramidNoiseGate"] == "Off"
        assert options["PyramidEnvelopeClip"] == "On"

    def test_pyramid_defaults_are_named_rather_than_left_blank(self):
        options = render_options.describe(algorithm="pyramid")
        # Depth 0 is the UI's "let the method decide" and resolves at render
        # time against the image size; nothing was recorded here, so there is
        # no number to quote alongside it.
        assert options["PyramidLevels"] == "Auto"
        assert options["PyramidSelectivity"] == "Balanced (8)"
        assert options["PyramidCoherence"] == "Off (0)"
        assert options["PyramidBaseWeighting"] == "Balanced (3)"
        assert options["PyramidNoiseGate"] == "On"
        assert options["PyramidEnvelopeClip"] == "On"

    def test_auto_depth_quotes_what_the_render_resolved_it_to(self):
        """'Auto' says who chose; the bracket says what they chose."""
        options = render_options.describe(
            algorithm="pyramid", pyramid_levels=0,
            resolved_auto={auto_params.PYRAMID_LEVELS: [5]},
        )
        assert options["PyramidLevels"] == "Auto (5)"

    def test_auto_depth_spans_the_values_a_tiled_render_used(self):
        # Edge tiles are smaller than the full blocks and can resolve
        # shallower, so both ends are stated rather than one of them.
        options = render_options.describe(
            algorithm="pyramid",
            resolved_auto={auto_params.PYRAMID_LEVELS: [5, 4]},
        )
        assert options["PyramidLevels"] == "Auto (4-5)"

    def test_a_depth_the_method_could_not_honour_says_what_ran(self):
        options = render_options.describe(
            algorithm="pyramid", pyramid_levels=8,
            resolved_auto={auto_params.PYRAMID_LEVELS: [6]},
        )
        assert options["PyramidLevels"] == "8 (6 applied)"
        # A request the method honoured is not restated in brackets.
        assert render_options.describe(
            algorithm="pyramid", pyramid_levels=6,
            resolved_auto={auto_params.PYRAMID_LEVELS: [6]},
        )["PyramidLevels"] == "6"

    def test_choose_max_selectivity_is_spelled_out(self):
        """inf is the published choose-max rule; 'inf' would name nothing."""
        assert render_options.describe(
            algorithm="pyramid",
            pyramid_selectivity=float("inf"))["PyramidSelectivity"] == "Winner (choose-max)"

    def test_a_pyramid_value_no_preset_holds_is_quoted_on_its_own(self):
        assert render_options.describe(
            algorithm="pyramid", pyramid_selectivity=12.5)["PyramidSelectivity"] == "12.5"

    def test_every_value_is_a_string(self):
        """The XMP writer escapes strings; a stray int would slip through it."""
        options = render_options.describe(
            algorithm="guided_filter", kernel_size=31, source_count=5, total_count=5,
            align_ecc=True, thread_count=8, device_name="CUDA (RTX 4090)",
        )
        assert options
        assert all(isinstance(value, str) for value in options.values())
