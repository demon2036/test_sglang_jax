"""Placeholder module for EasyDeL overrides.

Add monkey patches here while keeping imports optional and defensive.
"""

from __future__ import annotations

import os


def _log(message: str) -> None:
    if os.environ.get("EASYDEL_OVERRIDES_VERBOSE") == "1":
        print(message)


_log("easydel overrides patch module loaded")
