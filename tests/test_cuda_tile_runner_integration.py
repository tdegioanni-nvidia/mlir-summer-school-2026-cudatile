from __future__ import annotations

import base64
import ctypes
from pathlib import Path

import pytest
from cuda.bindings import driver

from cuda_tile_runner import launch


_FIXTURE = Path(__file__).parent / "testdata" / "store_answer_13_3.tileirbc.b64"


def _check_cuda(result: tuple[object, ...], operation: str) -> object | None:
    error = result[0]
    if int(error) != 0:
        _, name = driver.cuGetErrorName(error)
        _, description = driver.cuGetErrorString(error)
        pytest.fail(
            f"{operation} failed: {name.decode()}: {description.decode()}",
            pytrace=False,
        )

    values = result[1:]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _write_bytecode_fixture(tmp_path: Path) -> Path:
    bytecode_path = tmp_path / "store_answer_13_3.tileirbc"
    bytecode_path.write_bytes(base64.b64decode(_FIXTURE.read_bytes()))
    return bytecode_path


@pytest.mark.integration
def test_real_driver_launches_tile_bytecode_with_device_pointer(
    tmp_path: Path,
) -> None:
    """Launch a real Tile kernel and verify its device-memory side effect."""
    bytecode_path = _write_bytecode_fixture(tmp_path)

    _check_cuda(driver.cuInit(0), "cuInit")
    previous_context = _check_cuda(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
    device = _check_cuda(driver.cuDeviceGet(0), "cuDeviceGet")
    context = _check_cuda(
        driver.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain"
    )
    _check_cuda(driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    device_pointer = None
    try:
        device_pointer = _check_cuda(driver.cuMemAlloc(4), "cuMemAlloc")
        _check_cuda(driver.cuMemsetD32(device_pointer, 0, 1), "cuMemsetD32")

        launch(bytecode_path, int(device_pointer), device=0)

        output = ctypes.c_int32()
        _check_cuda(
            driver.cuMemcpyDtoH(ctypes.addressof(output), device_pointer, 4),
            "cuMemcpyDtoH",
        )
        assert output.value == 42
    finally:
        if device_pointer is not None:
            _check_cuda(driver.cuMemFree(device_pointer), "cuMemFree")
        _check_cuda(
            driver.cuCtxSetCurrent(previous_context), "cuCtxSetCurrent(restore)"
        )
        _check_cuda(
            driver.cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease"
        )


@pytest.mark.integration
def test_real_driver_accepts_pytorch_cuda_tensor(tmp_path: Path) -> None:
    """Pass a CUDA tensor directly and observe the Tile kernel's write."""
    try:
        import torch
    except ImportError:
        pytest.fail("PyTorch is required for the CUDA tensor integration test")

    assert torch.cuda.is_available(), "PyTorch does not have access to a CUDA GPU"

    bytecode_path = _write_bytecode_fixture(tmp_path)
    output = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    stream = torch.cuda.current_stream(output.device)

    launch(bytecode_path, output, device=output.device.index, stream=stream)

    assert output.cpu().item() == 42
