"""PyTorch availability checks that stay correct inside a frozen build.

A PyInstaller build made with ``--exclude-module torch`` can still ship a
``torch`` *directory*: CuPy's extension modules link against the CUDA DLLs that
live in ``site-packages/torch/lib``, so PyInstaller's dependency scan copies
those DLLs into ``torch/lib/`` inside the bundle. The directory has no
``__init__.py``, which makes ``import torch`` succeed and hand back an empty
implicit namespace package. Every ``except ImportError`` fallback in the
codebase is then bypassed and the first attribute access (``torch.cuda``)
raises ``AttributeError`` mid-render instead.

Importing this module installs the guard, so that such a stub fails the import
the same way a genuinely absent PyTorch does and the existing CPU fallbacks take
over. ``utils/__init__`` imports it first for that reason: the block has to be
in place before any module reaches for PyTorch, which is earlier than an entry
point could arrange by calling a function.
"""

import importlib.util
import sys
from typing import Iterable, List, Optional, Tuple

# Packages that are useless without their Python code: a bundle that carries
# only their shared libraries must not look importable.
_GUARDED_MODULES: Tuple[str, ...] = ("torch", "torchvision")


class _BlockedModuleFinder:
    """Meta path finder that fails the import of the modules it is given.

    A finder cannot veto later entries in ``sys.meta_path`` by returning a
    value, so the block is expressed as the exception the caller would have
    seen if the module had never been on the path at all.
    """

    def __init__(self, blocked: Iterable[str]):
        self.blocked = frozenset(blocked)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in self.blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None  # Not ours; let the regular finders answer.


def _find_spec_quietly(name: str):
    """Return the spec for ``name``, or None when it cannot be resolved."""
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        # ValueError: the module is already in sys.modules without a spec.
        return None


def is_stub_package(name: str) -> bool:
    """Return True when ``name`` resolves only to a directory without code.

    An implicit namespace package has no ``origin``, so importing it yields a
    module with no attributes - the frozen-build failure mode described above.
    """
    spec = _find_spec_quietly(name)
    return spec is not None and spec.origin is None


def block_stub_torch(module_names: Iterable[str] = _GUARDED_MODULES) -> List[str]:
    """Make code-less ``torch`` / ``torchvision`` bundles unimportable.

    Returns the names that were blocked (empty when the environment is sane),
    so callers can log what happened. Safe to call more than once.
    """
    stubs = [name for name in module_names if name not in sys.modules and is_stub_package(name)]
    if not stubs:
        return []

    for finder in sys.meta_path:
        if isinstance(finder, _BlockedModuleFinder):
            finder.blocked = finder.blocked.union(stubs)
            return stubs

    sys.meta_path.insert(0, _BlockedModuleFinder(stubs))
    return stubs


def is_torch_available() -> bool:
    """Return True when a usable (importable, complete) PyTorch is present."""
    if "torch" in sys.modules:
        return hasattr(sys.modules["torch"], "cuda")

    if is_stub_package("torch") or _find_spec_quietly("torch") is None:
        return False

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False

    # A partially collected torch imports but is missing its submodules.
    return hasattr(torch, "cuda")


def import_torch() -> Optional[object]:
    """Import and return PyTorch, or None when it is unavailable/unusable."""
    if not is_torch_available():
        return None
    return sys.modules.get("torch") or importlib.import_module("torch")


def gpu_device_name() -> Optional[str]:
    """Return 'CUDA' / 'MPS' for the backend PyTorch can use, else None."""
    torch = import_torch()
    if torch is None:
        return None

    try:
        if torch.cuda.is_available():
            return "CUDA"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "MPS"
    except Exception:
        # A driver-level failure means no usable device, not a crash.
        pass
    return None


def has_gpu_device() -> bool:
    """Return True when PyTorch reports a usable CUDA or MPS device."""
    return gpu_device_name() is not None


# Installed on import; see the module docstring. Empty on a healthy environment,
# so it doubles as the diagnostic shown in the environment info dialog.
BLOCKED_STUB_MODULES: Tuple[str, ...] = tuple(block_stub_torch())
