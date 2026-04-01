from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_training_path() -> None:
    """Allow scripts/ entrypoints to import sibling training packages directly."""

    training_root = Path(__file__).resolve().parents[1]
    training_root_str = str(training_root)
    if training_root_str not in sys.path:
        sys.path.insert(0, training_root_str)
