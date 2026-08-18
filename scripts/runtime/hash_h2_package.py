#!/usr/bin/env python3
"""Compute/verify the canonical content hash of an H2 package."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from inner.runtime.package_hash import h2_sha256  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect")
    args = parser.parse_args()
    observed = h2_sha256(args.path)
    if args.expect and observed != args.expect:
        raise SystemExit(
            f"fixed H2 hash mismatch: expected {args.expect}, observed {observed}"
        )
    print(observed)


if __name__ == "__main__":
    main()
