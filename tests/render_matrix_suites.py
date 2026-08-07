"""Named render matrices, so a question asked once can be asked again.

Each entry in SUITES is a function returning a Plan (see tests/render_matrix.py)
- what to render, at what settings, and where to put it. A suite is a question
someone had about a capture, written down: keeping it here rather than in a
shell history is what makes a second run a week later comparable with the first.

    python -m tests.render_matrix --list
    python -m tests.render_matrix --suite depthmap_average
    python -m tests.render_matrix --suite depthmap_average --step 8

Everything a suite sets can be overridden from the command line, `--step` above
all: a suite is written for the full stack, and `--step 8` is how it gets tried
on a tenth of it first.
"""

import itertools
import os
from typing import List, Sequence

from tests.render_matrix import Plan, RegistrationSpec, StackSpec, Suite, Variant

# The working capture: an electronics board with an ant on it, cropped, as DNG.
# 333 frames, shot on a Laowa 5x. Deep enough that a blend which does not select
# hard collapses into the mean of the stack, which is why it is the fixture the
# depth-map average is tuned on.
ELECTRONICS_ANT = r"F:\Media\MacroTest07\_frames\electronics_ant\framesDNG_c"
WORK = r"F:\Media\MacroTest07\work"

# Helicon Focus method A - its own contrast-weighted average, the like-for-like
# counterpart of depthmap_average - at radius 30, smoothing 1. The smoothest of
# the three Method A settings shipped with the capture, and the render this
# method is being asked to come closer to.
#
# It is a reference and not an answer. Helicon aligned the stack its own way, so
# the two renders sit several pixels apart locally and nothing rigid relates
# them; see samples/captures.py. What it is worth is that the no-reference
# metrics cannot rank grain against texture - `focus_retention` rewards whichever
# render kept the most local contrast, and on this capture that is the grainiest
# one - while an independent implementation of the same algorithm has already
# made that trade in a way somebody shipped.
HELICON_A30 = (r"F:\Media\MacroTest07\_frames\electronics_ant"
               r"\electronics_ant-HF-A-30-1.png")

# The app's shipped registration, and what every suite here uses unless it is
# asking a question about registration itself: homography then ECC, measured on
# 1024 px reductions, against the middle frame so chain drift is halved.
ECC_HOMOGRAPHY = RegistrationSpec(
    method="homography+ecc",
    downscale_width=1024,
    reference_mode="middle",
    ecc_parallel=True,
)


def _grid(fusion: str, **axes) -> List[Variant]:
    """Cross product of the named dials, in the order they were given."""
    names = list(axes)
    return [Variant(params=dict(zip(names, combination)))
            for combination in itertools.product(*(axes[name] for name in names))]


def depthmap_average() -> Plan:
    """Depth Map (Average): kernel x selectivity, then halo at the useful corners.

    The average blends the stack in proportion to each frame's local contrast,
    so on a 333-frame capture its failure mode is haze: a defocused frame still
    measures a fraction of the peak, and three hundred of those fractions add up
    until the blend is the mean of everything. Selectivity is the dial that
    stops it, kernel size is what decides how large a neighbourhood the contrast
    is judged over, and the two interact - a large kernel pools a defocused
    neighbour's energy into the decision, which a high selectivity then trusts.

    So the grid is those two together (25 renders) rather than one at a time.
    Halo suppression is a third axis that mostly matters at the depth
    boundaries, and it multiplies the grid rather than adding to it, so it is
    swept separately at the settings the grid is expected to land near.
    """
    variants = _grid(
        "depthmap_average",
        # 3 is finer than the grain; 51 is the app's maximum, and pools a
        # neighbourhood wide enough to blur the focus decision itself.
        kernel_size=[5, 9, 15, 25, 51],
        # 0 is the plain linear contrast weighting - the haze this is about.
        # 100 is as close to a hard select as the average gets.
        average_selectivity=[0, 25, 50, 75, 100],
    )
    # A halo pass at the two kernels most likely to win, at the selectivity
    # where the dial has something to act on. Duplicates of halo_radius=0 are
    # not repeated - the grid above already rendered those.
    variants += _grid(
        "depthmap_average",
        kernel_size=[9, 15],
        average_selectivity=[75],
        halo_radius=[4, 8, 15],
    )

    return Plan(
        stack=StackSpec(source=ELECTRONICS_ANT),
        destination=WORK,
        reference=HELICON_A30,
        suites=[Suite(
            name="Depth Map (Average) - kernel x selectivity x halo",
            fusion="depthmap_average",
            registration=ECC_HOMOGRAPHY,
            variants=variants,
            note="Halo rows are appended after the kernel x selectivity grid; "
                 "rows without a `_h` tag ran at halo_radius 0.",
        )],
    )


def depthmap_average_coherence() -> Plan:
    """Depth Map (Average): the weight-coherence radius, against kernel and selectivity.

    The suite the `coherence_radius` dial was added from. `depthmap_average`
    found the shape of the problem and could not fix it: every setting in that
    grid trades haze against grain along one line, and the only clean corner of
    it (kernel 51) is clean because a 51 px pool blurs the focus decision itself.
    The kernel was doing two jobs - resolving the detail and making the decision
    coherent - and the second one is what the new stage takes over.

    So the axes are the two the kernel was overloaded with, plus the dial that
    unloads it. The expectation being tested is specifically that the radius
    removes the kernel's second job: if it does, the small kernels should catch
    the large ones up rather than merely improving alongside them.

    Radius 0 rows are the controls, and they are the previous suite's grid
    exactly, so the two runs can be read against each other.
    """
    variants = _grid(
        "depthmap_average",
        kernel_size=[5, 9, 25, 51],
        average_selectivity=[50, 100],
        coherence_radius=[0, 8, 16, 24],
    )

    return Plan(
        stack=StackSpec(source=ELECTRONICS_ANT),
        destination=WORK,
        reference=HELICON_A30,
        suites=[Suite(
            name="Depth Map (Average) - weight coherence x kernel x selectivity",
            fusion="depthmap_average",
            registration=ECC_HOMOGRAPHY,
            variants=variants,
            note="`coherence_radius` 0 is the unfiltered blend, byte for byte, "
                 "and costs one pass over the stack; anything else costs two. "
                 "Scored against Helicon Focus method A at radius 30 - see "
                 "RefGap and RefAgree in the table, and read them against "
                 "Retention rather than instead of it.",
        )],
    )


def depthmap_average_halo() -> Plan:
    """Depth Map (Average): halo radius against kernel, at high selectivity.

    The follow-up to `depthmap_average`, which found the dial that matters. That
    grid left dark blobs scattered over the smooth parts of the picture at every
    small kernel - a defocused frame's glow measuring as detail and winning
    ground it cannot resolve - and two things removed them independently: a
    kernel wide enough to pool the glow away (51), and halo suppression, which
    is aimed at exactly that mechanism. h15 was the widest radius that grid
    tried and it was the best of {0, 4, 8, 15} by a visible margin, so the
    trend had not turned over and the range needs extending.

    Radii are kept at or below the kernel size here, which is the ceiling the UI
    enforces (constants.halo_radius_ceiling) - the engine is permissive, but a
    setting that cannot be dialled in the app is not an answer to "what should I
    render this at". Halo 0 rows are the controls.
    """
    variants = []
    for kernel, radii in ((15, [0, 8, 15]), (25, [0, 15, 25]), (51, [0, 15, 30])):
        variants += _grid(
            "depthmap_average",
            kernel_size=[kernel],
            average_selectivity=[75, 100],
            halo_radius=radii,
        )

    return Plan(
        stack=StackSpec(source=ELECTRONICS_ANT),
        destination=WORK,
        suites=[Suite(
            name="Depth Map (Average) - halo x kernel at high selectivity",
            fusion="depthmap_average",
            registration=ECC_HOMOGRAPHY,
            variants=variants,
            note="Radii stay within the UI ceiling of min(kernel, 30). "
                 "The `_h0` rows are the controls.",
        )],
    )


def depthmap_max() -> Plan:
    """Depth Map (Max): kernel x depth smoothing, then halo.

    The hard select's dials are the same kernel, the coherence that stops a
    region no frame resolves from tearing into patches, and the halo radius.
    Kept alongside the average suite so the two modes can be rendered from one
    load of the stack and compared directly.
    """
    variants = _grid(
        "depthmap_max",
        kernel_size=[5, 9, 15, 25, 51],
        depth_smoothing=[0, 25, 50, 75, 100],
    )
    variants += _grid(
        "depthmap_max",
        kernel_size=[9, 15],
        depth_smoothing=[50],
        halo_radius=[4, 8, 15],
    )

    return Plan(
        stack=StackSpec(source=ELECTRONICS_ANT),
        destination=WORK,
        suites=[Suite(
            name="Depth Map (Max) - kernel x smoothing x halo",
            fusion="depthmap_max",
            registration=ECC_HOMOGRAPHY,
            variants=variants,
        )],
    )


def depthmap_modes() -> Plan:
    """Both depth-map modes at their defaults, from one registration.

    A short run: what the two modes do to the same aligned stack, before any
    tuning. Useful as a rig check and as the baseline the sweeps are read
    against.
    """
    registration = ECC_HOMOGRAPHY
    stack = StackSpec(source=ELECTRONICS_ANT)
    return Plan(
        stack=stack,
        destination=WORK,
        suites=[
            Suite(name="Depth Map (Max) - defaults", fusion="depthmap_max",
                  registration=registration,
                  variants=_grid("depthmap_max", kernel_size=[9, 25],
                                 depth_smoothing=[50])),
            Suite(name="Depth Map (Average) - defaults", fusion="depthmap_average",
                  registration=registration,
                  variants=_grid("depthmap_average", kernel_size=[9, 25],
                                 average_selectivity=[50, 90])),
        ],
    )


SUITES = {
    "depthmap_average": depthmap_average,
    "depthmap_average_coherence": depthmap_average_coherence,
    "depthmap_average_halo": depthmap_average_halo,
    "depthmap_max": depthmap_max,
    "depthmap_modes": depthmap_modes,
}


def build(names: Sequence[str]) -> Plan:
    """One Plan from one or more named suites.

    Several suites can share a run only if they share a stack: the frames are
    loaded once and every suite renders from them, so a second source folder
    would have to be a second invocation.
    """
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        raise SystemExit(f"Unknown suite(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(sorted(SUITES))}")

    plans = [SUITES[name]() for name in names]
    sources = {os.path.normcase(plan.stack.source) for plan in plans}
    if len(sources) > 1:
        raise SystemExit("Suites run together must share one source folder; "
                         f"got {', '.join(sorted(sources))}")

    head = plans[0]
    suites: List[Suite] = []
    for plan in plans:
        suites.extend(plan.suites)
    return Plan(
        stack=head.stack,
        destination=head.destination,
        suites=suites,
        reference=head.reference,
        thread_count=head.thread_count,
        output_format=head.output_format,
        preview_format=head.preview_format,
        crop_count=head.crop_count,
        crop_size=head.crop_size,
        metric_source_limit=head.metric_source_limit,
        save_aligned=head.save_aligned,
    )
