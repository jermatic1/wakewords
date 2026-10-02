"""Generates synthetic wake word samples with piper-sample-generator.

The generator runs from its checkout under external/, as upstream does,
because the piper_train code it needs is not part of its installable package.
"""

import logging
import subprocess
import sys

from wakeword.audio import normalize_dir
from wakeword.config import WakewordConfig
from wakeword.data_setup import setup_piper_model
from wakeword.paths import Paths, piper_generator_dir, remove

log = logging.getLogger(__name__)


def generate(cfg: WakewordConfig, paths: Paths) -> None:
    model = setup_piper_model(paths, cfg.piper.model)
    out_dir = paths.wakeword(cfg.name).positives
    remove(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        cfg.phrase,
        "--model",
        str(model.resolve()),
        "--max-samples",
        str(cfg.piper.samples),
        "--batch-size",
        str(cfg.piper.batch_size),
        "--output-dir",
        str(out_dir.resolve()),
    ]
    if cfg.piper.max_speakers is not None:
        cmd += ["--max-speakers", str(cfg.piper.max_speakers)]
    log.info("generating %d samples of %r", cfg.piper.samples, cfg.phrase)
    subprocess.run(cmd, check=True, cwd=piper_generator_dir())
    normalize_dir(out_dir)
