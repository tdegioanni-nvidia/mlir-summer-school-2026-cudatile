"""Run a zero-argument CUDA Tile IR bytecode entry point."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from cuda_tile_runner import launch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the sole entry point in CUDA Tile IR bytecode."
    )
    parser.add_argument(
        "bytecode",
        type=os.fsdecode,
        help="path to a CUDA Tile IR bytecode file",
    )
    parser.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        default=(1, 1, 1),
        help="logical launch grid (default: 1 1 1)",
    )
    args = parser.parse_args(argv)

    launch(args.bytecode, grid=tuple(args.grid))
    print(f"CUDA Tile launch completed: {args.bytecode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
