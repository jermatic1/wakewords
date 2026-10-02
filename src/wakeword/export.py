"""Copies the trained model and writes the ESPHome v2 manifest."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from wakeword.config import FEATURE_STEP_MS, WakewordConfig
from wakeword.paths import Paths

log = logging.getLogger(__name__)


def build_manifest(cfg: WakewordConfig, model_filename: str) -> dict[str, Any]:
    return {
        "type": "micro",
        "wake_word": cfg.phrase,
        "author": cfg.manifest.author,
        "website": cfg.manifest.website,
        "model": model_filename,
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": cfg.manifest.probability_cutoff,
            "sliding_window_size": cfg.manifest.sliding_window_size,
            "feature_step_size": FEATURE_STEP_MS,
            "tensor_arena_size": cfg.manifest.tensor_arena_size,
            "minimum_esphome_version": cfg.manifest.minimum_esphome_version,
        },
    }


def export(cfg: WakewordConfig, paths: Paths) -> Path:
    ww = paths.wakeword(cfg.name)
    if not ww.tflite.exists():
        raise FileNotFoundError(f"{ww.tflite} not found; run train")
    ww.output.mkdir(parents=True, exist_ok=True)
    model_name = f"{cfg.name}.tflite"
    shutil.copy(ww.tflite, ww.output / model_name)
    manifest_path = ww.output / f"{cfg.name}.json"
    with open(manifest_path, "w") as f:
        json.dump(build_manifest(cfg, model_name), f, indent=2)
        f.write("\n")
    log.info("wrote %s and %s", ww.output / model_name, manifest_path)
    return manifest_path
