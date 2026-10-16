"""Pinned robot identity shared by host calibration and Docker simulation."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "source/so101_typing/tasks/SO101/data/workshop_asset_sources.json"
WORKSHOP_USD_SHA256 = json.loads(MANIFEST.read_text())["expected_asset_sha256"]
TYPING_USD_ENV = "SO101_TYPING_USD"
DEFAULT_TYPING_USD = "/opt/so101/assets/workshop_so101.usdc"


def typing_usd_path() -> Path:
    return Path(os.environ.get(TYPING_USD_ENV, DEFAULT_TYPING_USD))
