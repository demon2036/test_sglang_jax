"""Runtime overrides loaded via PYTHONPATH.

Python automatically imports sitecustomize when it is importable. We keep the
logic small and optional to avoid breaking upstream code.
"""

from __future__ import annotations

import importlib
import os


def _log(message: str) -> None:
    if os.environ.get("EASYDEL_OVERRIDES_VERBOSE") == "1":
        print(message)


module_name = os.environ.get("EASYDEL_OVERRIDES_MODULE", "easydel_overrides_patch")
try:
    importlib.import_module(module_name)
    _log(f"easydel overrides loaded: {module_name}")
except ModuleNotFoundError:
    _log(f"easydel overrides missing: {module_name}")
