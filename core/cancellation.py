"""Shared cancellation primitive for the render pipeline.

Kept in its own dependency-free module so both the worker thread
(core.workers) and the fusion engine (core.multi_focus_fusion) can raise and
propagate the same exception without importing each other.
"""


class RenderCancelled(Exception):
    """Raised to unwind the render pipeline when the user requests a stop.

    The worker passes a ``cancel_check`` callable into the heavy stages; that
    callable raises this exception at the next checkpoint, letting the normal
    stack unwind free any resources instead of hard-killing the thread.
    """
    pass
