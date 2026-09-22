"""End-to-end: ingest -> classify -> validate.

    python run.py                 # uses config.yaml (provider defaults to mock)
    python run.py --sample 100    # classify fewer incidents

Set classify.provider to "anthropic" or "openai" in config.yaml (and the
matching API key in your environment) for a real run. "mock" only tests wiring.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src import ingest, classify, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    print("== step 1: ingest ==")
    ingest.main()
    print("\n== step 2: classify ==")
    classify.main(sample=args.sample)
    print("\n== step 3: validate ==")
    validate.main()


if __name__ == "__main__":
    main()