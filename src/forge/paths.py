"""storage locations, resolved at runtime so no machine specific path is ever written in code or config.

the storage root holds data and runs. it is chosen in this order.
1. the FORGE_STORAGE environment variable, when set.
2. the project folder on Google Drive, when running on Colab.
3. the drive folder inside the repository, a local mirror of the Drive project folder that git ignores.
"""

import os
import sys
from pathlib import Path

PROJECT_NAME = "forge-nids"
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive") / PROJECT_NAME


def repo_root() -> Path:
    # src/forge/paths.py sits two levels below the repository root
    return Path(__file__).resolve().parents[2]


def on_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def storage_root() -> Path:
    env = os.environ.get("FORGE_STORAGE")
    if env:
        return Path(env).expanduser().resolve()
    if on_colab():
        return COLAB_DRIVE_ROOT
    return repo_root() / "drive"


def data_dir(kind: str = "raw") -> Path:
    # kind is raw or processed
    return storage_root() / "data" / kind


def runs_dir() -> Path:
    return storage_root() / "runs"


def configs_dir() -> Path:
    return repo_root() / "configs"
