from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

import cuda_tile_runner


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _append_section(
    output: bytearray, section_id: int, payload: bytes, alignment: int
) -> None:
    output.append(section_id | (0x80 if alignment > 1 else 0))
    output.extend(_varint(len(payload)))
    if alignment > 1:
        output.extend(_varint(alignment))
        output.extend(b"\xcb" * (-len(output) % alignment))
    output.extend(payload)


def _test_bytecode(entry_name: str = "test_kernel") -> bytes:
    output = bytearray(b"\x7fTileIR\x00\x0d\x04\x00\x00")

    # One public kernel: name index, signature index, kernel flag, location,
    # and an empty body. The function payload itself is padded to 8 bytes.
    function_payload = bytearray(b"\x01\x00\x00\x02\x00\x00")
    function_payload.extend(b"\xcb" * (-len(function_payload) % 8))
    _append_section(output, 2, bytes(function_payload), 8)

    encoded_name = entry_name.encode("utf-8")
    string_payload = bytearray(b"\x01")
    string_payload.extend(b"\xcb" * (-len(string_payload) % 4))
    string_payload.extend((0).to_bytes(4, "little"))
    string_payload.extend(encoded_name)
    _append_section(output, 1, bytes(string_payload), 4)
    output.append(0)
    return bytes(output)


class _DataPtrBuffer:
    def __init__(self, pointer: int) -> None:
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer


class _CudaArrayBuffer:
    def __init__(self, pointer: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "<f4",
            "data": (pointer, False),
            "version": 3,
        }


class _PointerBuffer:
    def __init__(self, pointer: int) -> None:
        self.pointer = pointer


class _TorchStreamLike:
    cuda_stream = 91


class _StubCUresult:
    CUDA_ERROR_NOT_FOUND = 500


class _RecordingDriver:
    """Minimal Driver API recorder for tests that must inspect the launch ABI."""

    CUstream = staticmethod(lambda value: ("stream", value))
    CUresult = _StubCUresult

    def __init__(
        self,
        *,
        entry_name: str = "test_kernel",
        launch_error: int = 0,
        current_context: int = 7,
    ) -> None:
        self.entry_name = entry_name.encode("utf-8")
        self.launch_error = launch_error
        self.current_context = current_context
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.expected_argument_count = 0
        self.launched_argument_values: list[int] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def call_names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def cuInit(self, flags: int) -> tuple[int]:
        self._record("cuInit", flags)
        return (0,)

    def cuCtxGetCurrent(self) -> tuple[int, int]:
        self._record("cuCtxGetCurrent")
        return (0, self.current_context)

    def cuDeviceGet(self, device: int) -> tuple[int, int]:
        self._record("cuDeviceGet", device)
        return (0, device)

    def cuDevicePrimaryCtxRetain(self, device: int) -> tuple[int, int]:
        self._record("cuDevicePrimaryCtxRetain", device)
        return (0, 70 + device)

    def cuCtxSetCurrent(self, context: int) -> tuple[int]:
        self._record("cuCtxSetCurrent", context)
        return (0,)

    def cuLibraryLoadFromFile(self, **kwargs: object) -> tuple[int, int]:
        self._record("cuLibraryLoadFromFile", **kwargs)
        return (0, 11)

    def cuLibraryGetKernel(self, library: int, name: bytes) -> tuple[int, int]:
        self._record("cuLibraryGetKernel", library, name)
        return (0, 22) if name == self.entry_name else (500, 0)

    def cuKernelGetFunction(self, kernel: int) -> tuple[int, int]:
        self._record("cuKernelGetFunction", kernel)
        return (0, 33)

    def cuLaunchKernel(self, *args: object) -> tuple[int]:
        self._record("cuLaunchKernel", *args)
        kernel_params = int(args[-2])
        if self.expected_argument_count:
            array_type = ctypes.c_void_p * self.expected_argument_count
            parameter_addresses = ctypes.cast(
                kernel_params, ctypes.POINTER(array_type)
            ).contents
            self.launched_argument_values = [
                ctypes.cast(address, ctypes.POINTER(ctypes.c_uint64)).contents.value
                for address in parameter_addresses
            ]
        else:
            self.launched_argument_values = []
        return (self.launch_error,)

    def cuStreamSynchronize(self, stream: object) -> tuple[int]:
        self._record("cuStreamSynchronize", stream)
        return (0,)

    def cuLibraryUnload(self, library: int) -> tuple[int]:
        self._record("cuLibraryUnload", library)
        return (0,)

    def cuGetErrorName(self, error: int) -> tuple[int, bytes]:
        return (0, b"CUDA_ERROR_TEST")

    def cuGetErrorString(self, error: int) -> tuple[int, bytes]:
        return (0, b"test failure")


@pytest.fixture
def bytecode_path(tmp_path: Path) -> Path:
    path = tmp_path / "kernel.tileirbc"
    path.write_bytes(_test_bytecode())
    return path


@pytest.fixture(autouse=True)
def clear_retained_contexts() -> None:
    cuda_tile_runner._retained_primary_contexts.clear()
    yield
    cuda_tile_runner._retained_primary_contexts.clear()


def _launch_with_driver(
    monkeypatch: pytest.MonkeyPatch,
    bytecode_path: Path,
    driver: _RecordingDriver,
    *buffers: object,
    **kwargs: object,
) -> None:
    monkeypatch.setattr(cuda_tile_runner, "_load_driver", lambda: driver)
    cuda_tile_runner.launch(bytecode_path, *buffers, **kwargs)


def test_launch_packs_buffer_addresses_in_order(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver()
    driver.expected_argument_count = 4

    _launch_with_driver(
        monkeypatch,
        bytecode_path,
        driver,
        _DataPtrBuffer(0x1000),
        _CudaArrayBuffer(0x2000),
        _PointerBuffer(0x3000),
        0x4000,
        grid=(5, 3, 1),
        stream=_TorchStreamLike(),
    )

    assert driver.launched_argument_values == [0x1000, 0x2000, 0x3000, 0x4000]
    launch_call = next(call for call in driver.calls if call[0] == "cuLaunchKernel")
    launch_args = launch_call[1]
    assert launch_args[1:7] == (5, 3, 1, 1, 1, 1)
    assert launch_args[7] == 0
    assert launch_args[8] == ("stream", 91)
    assert driver.call_names()[-2:] == ["cuStreamSynchronize", "cuLibraryUnload"]


def test_entry_name_is_discovered_from_bytecode(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver()

    _launch_with_driver(monkeypatch, bytecode_path, driver)

    assert "cuLibraryGetKernel" in driver.call_names()
    kernel_call = next(call for call in driver.calls if call[0] == "cuLibraryGetKernel")
    assert kernel_call[1] == (11, b"test_kernel")


def test_invalid_bytecode_is_rejected_before_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver()
    bytecode_path.write_bytes(b"not tile ir")

    with pytest.raises(cuda_tile_runner.CudaTileLaunchError, match="not CUDA Tile"):
        _launch_with_driver(monkeypatch, bytecode_path, driver)

    assert driver.calls == []


def test_launch_error_is_descriptive_and_library_is_unloaded(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver(launch_error=700)

    with pytest.raises(cuda_tile_runner.CudaError) as raised:
        _launch_with_driver(monkeypatch, bytecode_path, driver)

    assert raised.value.operation == "cuLaunchKernel"
    assert raised.value.code == 700
    assert "CUDA_ERROR_TEST: test failure" in str(raised.value)
    assert driver.call_names()[-1] == "cuLibraryUnload"
    assert "cuStreamSynchronize" not in driver.call_names()


def test_invalid_grid_and_null_pointer_fail_before_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver()
    with pytest.raises(cuda_tile_runner.CudaTileLaunchError, match="grid dimensions"):
        _launch_with_driver(monkeypatch, bytecode_path, driver, grid=(1, 0, 1))
    with pytest.raises(cuda_tile_runner.CudaTileLaunchError, match="null or invalid"):
        _launch_with_driver(
            monkeypatch, bytecode_path, driver, _DataPtrBuffer(0)
        )
    assert driver.calls == []


def test_missing_context_retains_primary_context(
    monkeypatch: pytest.MonkeyPatch, bytecode_path: Path
) -> None:
    driver = _RecordingDriver(current_context=0)

    _launch_with_driver(monkeypatch, bytecode_path, driver, device=2)

    assert "cuDeviceGet" in driver.call_names()
    assert "cuDevicePrimaryCtxRetain" in driver.call_names()
    assert "cuCtxSetCurrent" in driver.call_names()
