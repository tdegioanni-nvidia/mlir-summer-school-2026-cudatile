"""Run and check a 256x256 f16-input/f32-output Tile matrix multiplication."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from cuda_tile_runner import launch


_MATRIX_SIZE = 256
_TILE_SIZE = 16


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a CUDA Tile matrix-multiplication kernel and check its result."
        )
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

    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch is required: {exc}")

    if not torch.cuda.is_available():
        parser.error("PyTorch does not have access to a CUDA GPU")

    device = torch.device("cuda:0")
    element_count = _MATRIX_SIZE * _MATRIX_SIZE
    tiles_per_dimension = _MATRIX_SIZE // _TILE_SIZE
    base = torch.arange(
        element_count, dtype=torch.float32, device=device
    ).reshape(
        tiles_per_dimension,
        tiles_per_dimension,
        _TILE_SIZE,
        _TILE_SIZE,
    )
    base = base.permute(0, 2, 1, 3).reshape(_MATRIX_SIZE, _MATRIX_SIZE)
    input_a = (base / (element_count - 1)).to(torch.float16)
    input_b = (base.flip((0, 1)) / (element_count - 1)).to(torch.float16)
    output = torch.zeros(
        (_MATRIX_SIZE, _MATRIX_SIZE), dtype=torch.float32, device=device
    )

    stream = torch.cuda.current_stream(device)
    launch(
        args.bytecode,
        input_a,
        input_b,
        output,
        grid=tuple(args.grid),
        stream=stream,
        device=device.index,
    )

    actual = output.cpu()
    expected = torch.matmul(input_a.cpu().float(), input_b.cpu().float())
    matches = torch.allclose(actual, expected, rtol=1e-3, atol=1e-2)

    if matches:
        print("Matmul result is correct.")
        return 0

    max_absolute_error = torch.max(torch.abs(actual - expected)).item()
    print(f"Matmul result is incorrect (maximum absolute error: {max_absolute_error}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
