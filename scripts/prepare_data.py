from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ecommerce_intelligence.data_processing import prepare_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw retail data for the project.")
    parser.add_argument("--sample", action="store_true", help="Use the small beginner sample dataset.")
    args = parser.parse_args()

    outputs = prepare_all(use_sample=args.sample)
    print("Data preparation complete.")
    for name, frame in outputs.items():
        print(f"{name}: {frame.shape[0]} rows, {frame.shape[1]} columns")


if __name__ == "__main__":
    main()
