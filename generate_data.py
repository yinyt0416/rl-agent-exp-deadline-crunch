"""Generate the fixed train/val/test scenario dataset."""

import argparse
import json
from pathlib import Path

from config import EnvConfig
from scenario_generator import generate_dataset

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(BASE_DIR / "data"))
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--config", default=None, help="optional EnvConfig JSON path")
    args = parser.parse_args()

    config = EnvConfig.from_json(args.config) if args.config else EnvConfig()
    manifest = generate_dataset(args.output, args.n_train, args.n_val, args.n_test, config)
    summary = {split: info["count"] for split, info in manifest["splits"].items()}
    print("dataset written to", args.output, json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
