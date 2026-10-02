"""Runs microwakeword training and streaming quantized export."""

import logging
import subprocess
import sys

from wakeword import data_setup, train_config
from wakeword.config import WakewordConfig
from wakeword.paths import Paths, remove

log = logging.getLogger(__name__)


def train(cfg: WakewordConfig, paths: Paths, resume: bool = False) -> None:
    ww = paths.wakeword(cfg.name)
    if not ww.positive_features.is_dir():
        raise FileNotFoundError(f"no features in {ww.positive_features}; run features")
    data_setup.setup_negatives(paths)
    if not resume:
        remove(ww.train_dir)
    config_path = train_config.write(cfg, paths)
    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train",
        "1",
        "--restore_checkpoint",
        "1" if resume else "0",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        cfg.train.pointwise_filters,
        "--repeat_in_block",
        "1,1,1,1",
        "--mixconv_kernel_sizes",
        "[5], [7,11], [9,15], [23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        str(cfg.train.stride),
    ]
    log.info("training %s for %d steps", cfg.name, cfg.train.steps)
    subprocess.run(cmd, check=True)
