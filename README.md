# CUDA Tile bytecode runner

`cuda_tile_runner.py` loads CUDA Tile IR bytecode through the CUDA Driver API
and launches its entry kernel with device-buffer pointers.

## Requirements

- A CUDA driver capable of loading Tile IR bytecode
- Python 3.10 or newer
- `cuda-python` (`python -m pip install cuda-python`)
- `pytest` for running the test suite
- PyTorch with CUDA support for the PyTorch-buffer integration test

Install the Python dependencies with:

```console
python -m pip install -r requirements.txt
```

The bytecode version must be supported by the installed driver. For example,
a driver exposing CUDA 13.3 can load Tile IR bytecode emitted with version
13.3, but bytecode emitted for a newer CUDA release may fail during driver JIT.

## Usage

```python
import torch

from cuda_tile_runner import launch

a = torch.empty(1024, device="cuda")
b = torch.empty_like(a)

launch("kernel.tileirbc", a, b, grid=(8, 1, 1))
```

To launch a bytecode file whose entry point takes no buffers:

```console
python run_cuda_tile.py path/to/kernel.tileirbc --grid 8 1 1
```

`--grid` defaults to `1 1 1` when omitted.

To run and validate a 256x256 matrix-multiplication kernel with float16 inputs
and a float32 output:

```console
python run_matmul.py path/to/matmul.tileirbc --grid 1 1 1
```

The script passes two float16 CUDA tensors derived from a tile-major normalized
`arange(256 * 256)`; the second input is reversed. This preserves 256 distinct
16x16 tile patterns after float16 conversion. The zero-filled output tensor is
float32. The script compares the copied-back result with PyTorch's float32 CPU
matrix multiplication and exits with status 1 on a mismatch.

Buffers are passed in kernel-argument order. PyTorch tensors (`data_ptr()`),
CUDA Array Interface objects such as CuPy arrays, DkgDSL-style objects with a
`pointer` attribute, and raw integer device addresses are accepted.

The bytecode entry kernel is discovered automatically. The runner intentionally
requires bytecode with exactly one entry point.

An integer CUDA stream handle or a `torch.cuda.Stream` may be passed with
`stream=`. The selected stream is synchronized before `launch` returns.

The logical `grid` is required from the caller because it is not encoded in
Tile IR bytecode. The physical block shape passed to the Driver API is
`(1, 1, 1)`; CUDA Tile kernel metadata tells the driver the actual CTA width
and cluster shape.

Run the complete suite, including the real-driver GPU integration test, with:

```console
pytest -v
```

The integration tests launch the CUDA Tile 13.3 fixture in `tests/testdata/`
through the real driver. One uses a raw Driver API allocation; the other passes
a CUDA `torch.Tensor` directly. Both verify that the kernel writes `42`. Run
only the isolated tests with `pytest -v -m "not integration"`.
