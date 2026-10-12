"""downloads and prepares the dataset a data config names.

    python scripts/prepare.py configs/data/litnet2020.yaml

raw files go to data/raw under the storage root and are fetched only when missing, each checked against its published
size and sha256. the processed version the config names is written once and never overwritten.
"""

import argparse
from datetime import datetime, timezone
from importlib import import_module

from forge.config import load_config


def log(message: str) -> None:
    # flushed at once, so a notebook that streams this process shows each line as it happens
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    module = import_module(f"forge.data.{cfg['dataset']['name']}")
    module.download(cfg, log=log)
    module.prepare(cfg, log=log)


if __name__ == "__main__":
    main()
