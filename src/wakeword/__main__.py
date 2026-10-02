"""Command line interface: wakeword <command> <config.yaml>."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from wakeword import export, features, gen_piper, train
from wakeword.config import ConfigError, load_config
from wakeword.paths import Paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wakeword")
    parser.set_defaults(resume=False, skip_background=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("generate", "synthesize positive samples with Piper"),
        ("features", "build spectrogram feature sets"),
        ("train", "train the model"),
        ("export", "write the tflite model and ESPHome manifest"),
        ("all", "generate, features, train and export"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", type=Path)
        if name in ("features", "all"):
            p.add_argument("--skip-background", action="store_true")
        if name == "train":
            p.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    paths = Paths(Path("data"), Path("output"))
    cfg = load_config(args.config)
    if args.command in ("generate", "all"):
        gen_piper.generate(cfg, paths)
    if args.command in ("features", "all"):
        features.build_features(cfg, paths, skip_background=args.skip_background)
    if args.command in ("train", "all"):
        train.train(cfg, paths, resume=args.resume)
    if args.command in ("export", "all"):
        export.export(cfg, paths)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        run(build_parser().parse_args(argv))
    except (ConfigError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: {Path(e.cmd[0]).name} {' '.join(e.cmd[1:3])} failed")


if __name__ == "__main__":
    main()
