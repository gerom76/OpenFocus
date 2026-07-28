"""What a setting left on 'Auto' actually resolved to during a render.

Some controls carry no number of their own: the pyramid's level count reads
'Auto', and the method picks a depth from the frame size at the moment it runs.
'Auto' on its own does not say what produced the pixels, which is exactly what
somebody rereading a saved file months later wants to know, so the value the
method settled on is recorded here as it is chosen and read back out when the
render is logged and described (see `core/render_options.py`).

Recording is opened around one fusion run by `recording()`. Tiled fusion calls
the method once per tile from a pool of worker threads, so the sink is shared
between threads rather than thread-local, and every distinct value a run
produced is kept: edge tiles are smaller than the rest and can resolve
shallower than the full blocks, which is worth showing rather than hiding
behind whichever tile happened to finish first.
"""

from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

# The names values are recorded under. They end up in a log line and in the
# saved metadata, so they are spelled out once here rather than at each call
# site, along with the wording the log uses for them.
PYRAMID_LEVELS = "pyramid_levels"
STACKMFF_BATCH_SIZE = "stackmffv4_batch_size"
TILE_WORKERS = "tile_workers"

_LABELS = {
    PYRAMID_LEVELS: "pyramid levels",
    STACKMFF_BATCH_SIZE: "StackMFF batch size",
    TILE_WORKERS: "tiles at once",
}

_lock = Lock()
_sinks: List[Dict[str, List[Any]]] = []

Resolved = Mapping[str, Sequence[Any]]


@contextmanager
def recording() -> Iterator[Dict[str, List[Any]]]:
    """Collect the auto values resolved inside the block.

    The dict fills as the block runs and is complete once it returns; a run
    that resolved nothing yields an empty one.
    """
    sink: Dict[str, List[Any]] = {}
    with _lock:
        _sinks.append(sink)
    try:
        yield sink
    finally:
        with _lock:
            _sinks.remove(sink)


def record(name: str, value: Any) -> None:
    """Note the value an auto setting resolved to.

    Doing nothing when no recording is open is deliberate: the fusion methods
    are importable and runnable on their own, and none of them should have to
    care whether anybody is listening.
    """
    with _lock:
        for sink in _sinks:
            seen = sink.setdefault(name, [])
            if value not in seen:
                seen.append(value)


def describe_value(values: Sequence[Any]) -> str:
    """One resolved value as text, or 'low-high' when a tiled run disagreed."""
    if not values:
        return ""
    try:
        lowest, highest = min(values), max(values)
    except TypeError:                      # values that do not order; quote them
        return ", ".join(str(value) for value in values)
    return f"{lowest}" if lowest == highest else f"{lowest}-{highest}"


def value_of(resolved: Optional[Resolved], name: str) -> Optional[str]:
    """The value recorded under `name`, as text, or None when nothing was."""
    if not resolved:
        return None
    return describe_value(resolved.get(name) or ()) or None


def summarize(resolved: Optional[Resolved]) -> str:
    """One line naming what each auto setting resolved to, for the log."""
    if not resolved:
        return ""
    return ", ".join(
        f"{_LABELS.get(name, name)} = {describe_value(values)}"
        for name, values in resolved.items() if values
    )
