"""Launch CUDA Tile IR bytecode with the CUDA Driver API.

CUDA Tile kernels carry their physical CTA configuration in compiled kernel
metadata.  Consequently, the CUDA launch receives a placeholder block shape
of ``(1, 1, 1)`` while ``grid`` specifies the logical tile-program grid.  The
driver resolves the actual threads-per-CTA and, when applicable, cluster shape.

The only runtime dependency is ``cuda-python`` (``cuda.bindings.driver``).
It is imported lazily so applications can import this module on CPU-only hosts.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

__all__ = ["CudaError", "CudaTileLaunchError", "launch"]


class CudaTileLaunchError(RuntimeError):
    """Raised when a CUDA Tile bytecode launch cannot be prepared."""


class CudaError(CudaTileLaunchError):
    """Raised when a CUDA Driver API operation fails."""

    def __init__(self, operation: str, code: int, description: str) -> None:
        super().__init__(f"{operation} failed with CUDA error {code}: {description}")
        self.operation = operation
        self.code = code
        self.description = description


# Retaining a primary context is intentionally process-lifetime state.  This is
# used only when the application has not already made a CUDA context current.
# Frameworks such as PyTorch normally establish a current context first.
_retained_primary_contexts: dict[int, object] = {}

_TILE_IR_MAGIC = b"\x7fTileIR\x00"
_SECTION_END = 0
_SECTION_STRING = 1
_SECTION_FUNCTION = 2
_ALIGNMENT_PADDING_BYTE = 0xCB


def _load_driver() -> Any:
    try:
        from cuda.bindings import driver  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        raise CudaTileLaunchError(
            "cuda-python is required; install it with `python -m pip install "
            "cuda-python`"
        ) from exc
    return driver


def _integer_value(value: object) -> int:
    raw_value = getattr(value, "value", value)
    if raw_value is None:
        return 0
    return int(raw_value)


def _decode_cuda_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cuda_error_description(driver: Any, error: object) -> str:
    details: list[str] = []
    for query_name in ("cuGetErrorName", "cuGetErrorString"):
        query = getattr(driver, query_name, None)
        if query is None:
            continue
        try:
            result = query(error)
            if isinstance(result, tuple) and len(result) >= 2:
                if _integer_value(result[0]) == 0:
                    details.append(_decode_cuda_text(result[1]))
        except Exception:
            # Error reporting must not hide the original CUDA failure.
            continue
    return ": ".join(details) if details else "unknown error"


def _check_cuda(driver: Any, result: object, operation: str) -> Any:
    """Validate and unpack a cuda-python Driver API return tuple."""
    if not isinstance(result, tuple) or not result:
        raise CudaTileLaunchError(
            f"{operation} returned an unexpected cuda-python result: {result!r}"
        )

    error = result[0]
    error_code = _integer_value(error)
    if error_code != 0:
        raise CudaError(
            operation,
            error_code,
            _cuda_error_description(driver, error),
        )

    values = result[1:]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _validate_bytecode_path(bytecode_path: str | os.PathLike[str]) -> Path:
    try:
        path = Path(bytecode_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise CudaTileLaunchError(
            f"CUDA Tile bytecode file does not exist: {bytecode_path!r}"
        ) from exc
    if not path.is_file():
        raise CudaTileLaunchError(f"CUDA Tile bytecode path is not a file: {path}")
    return path


def _read_varint(
    data: bytes | memoryview, offset: int, limit: int, description: str
) -> tuple[int, int]:
    value = 0
    for byte_index in range(10):
        if offset >= limit:
            raise CudaTileLaunchError(
                f"truncated CUDA Tile bytecode while reading {description}"
            )
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << (7 * byte_index)
        if byte & 0x80 == 0:
            return value, offset
    raise CudaTileLaunchError(
        f"invalid CUDA Tile bytecode: {description} varint exceeds 64 bits"
    )


def _aligned_offset(offset: int, alignment: int, limit: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise CudaTileLaunchError(
            f"invalid CUDA Tile bytecode section alignment: {alignment}"
        )
    aligned = (offset + alignment - 1) & -alignment
    if aligned > limit:
        raise CudaTileLaunchError("CUDA Tile bytecode section padding is truncated")
    return aligned


def _bytecode_sections(data: bytes) -> dict[int, memoryview]:
    """Return the Tile IR sections needed for entry-name discovery."""
    if len(data) < 12 or not data.startswith(_TILE_IR_MAGIC):
        raise CudaTileLaunchError("file is not CUDA Tile IR bytecode")

    view = memoryview(data)
    offset = 12  # magic[8], major[1], minor[1], version tag[2]
    sections: dict[int, memoryview] = {}
    saw_end = False
    while offset < len(data):
        id_and_alignment = data[offset]
        offset += 1
        section_id = id_and_alignment & 0x7F
        if section_id == _SECTION_END:
            saw_end = True
            break

        length, offset = _read_varint(
            data, offset, len(data), f"section {section_id} length"
        )
        if id_and_alignment & 0x80:
            alignment, offset = _read_varint(
                data, offset, len(data), f"section {section_id} alignment"
            )
            aligned = _aligned_offset(offset, alignment, len(data))
            if any(byte != _ALIGNMENT_PADDING_BYTE for byte in data[offset:aligned]):
                raise CudaTileLaunchError(
                    f"invalid padding in CUDA Tile bytecode section {section_id}"
                )
            offset = aligned

        end = offset + length
        if end > len(data):
            raise CudaTileLaunchError(
                f"CUDA Tile bytecode section {section_id} exceeds the file size"
            )
        if section_id in sections:
            raise CudaTileLaunchError(
                f"CUDA Tile bytecode contains duplicate section {section_id}"
            )
        sections[section_id] = view[offset:end]
        offset = end

    if not saw_end:
        raise CudaTileLaunchError("CUDA Tile bytecode is missing its end marker")
    return sections


def _parse_string_table(payload: memoryview) -> list[str]:
    count, offset = _read_varint(payload, 0, len(payload), "string count")
    offset = _aligned_offset(offset, 4, len(payload))
    offset_table_size = count * 4
    data_start = offset + offset_table_size
    if data_start > len(payload):
        raise CudaTileLaunchError("CUDA Tile bytecode string-offset table is truncated")

    string_data = bytes(payload[data_start:])
    starts = [
        int.from_bytes(payload[offset + i * 4 : offset + (i + 1) * 4], "little")
        for i in range(count)
    ]
    if starts and starts[0] != 0:
        raise CudaTileLaunchError("first CUDA Tile bytecode string offset is not zero")
    if any(start > len(string_data) for start in starts) or any(
        left > right for left, right in zip(starts, starts[1:])
    ):
        raise CudaTileLaunchError("CUDA Tile bytecode contains invalid string offsets")

    strings: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(string_data)
        raw_string = string_data[start:end]
        if b"\x00" in raw_string:
            raise CudaTileLaunchError("CUDA Tile bytecode contains a NUL in a string")
        try:
            strings.append(raw_string.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise CudaTileLaunchError(
                "CUDA Tile bytecode string table is not valid UTF-8"
            ) from exc
    return strings


def _entry_name_candidates(bytecode_path: Path) -> list[str]:
    """Find likely entry symbols without requiring Tile IR compiler bindings.

    CUDA currently reports zero from ``cuLibraryGetKernelCount`` for lazily
    loaded Tile IR.  The bytecode's stable string table therefore supplies the
    symbols to probe with ``cuLibraryGetKernel``.  The first function name is
    tried first; all remaining strings are fallbacks for modules containing
    helper functions before their sole entry point.
    """
    try:
        data = bytecode_path.read_bytes()
    except OSError as exc:
        raise CudaTileLaunchError(
            f"failed to read CUDA Tile bytecode: {bytecode_path}"
        ) from exc
    sections = _bytecode_sections(data)
    string_payload = sections.get(_SECTION_STRING)
    function_payload = sections.get(_SECTION_FUNCTION)
    if string_payload is None or function_payload is None:
        raise CudaTileLaunchError(
            "CUDA Tile bytecode must contain string and function sections"
        )

    strings = _parse_string_table(string_payload)
    function_count, function_offset = _read_varint(
        function_payload, 0, len(function_payload), "function count"
    )
    if function_count == 0:
        raise CudaTileLaunchError("CUDA Tile bytecode contains no functions")
    first_name_index, _ = _read_varint(
        function_payload,
        function_offset,
        len(function_payload),
        "first function name index",
    )
    if first_name_index >= len(strings):
        raise CudaTileLaunchError(
            "CUDA Tile bytecode first function has an invalid name index"
        )

    ordered = [strings[first_name_index], *strings]
    return list(dict.fromkeys(name for name in ordered if name))


def _validate_grid(grid: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(grid, tuple) or len(grid) != 3:
        raise CudaTileLaunchError("grid must be a three-element tuple")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in grid):
        raise CudaTileLaunchError(
            f"grid dimensions must be positive integers, got {grid!r}"
        )
    return grid


def _pointer_value(value: object, *, source: str) -> int:
    try:
        address = _integer_value(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CudaTileLaunchError(
            f"{source} did not produce an integer device address"
        ) from exc
    if address <= 0:
        raise CudaTileLaunchError(
            f"{source} produced a null or invalid device address: {address}"
        )
    if address > ctypes.c_uint64(-1).value:
        raise CudaTileLaunchError(
            f"{source} produced an address wider than 64 bits: {address}"
        )
    return address


def _device_pointer(buffer: object, index: int) -> int:
    """Extract a CUDA device address without taking ownership of ``buffer``."""
    source = f"buffer {index} ({type(buffer).__name__})"

    if isinstance(buffer, bool):
        raise CudaTileLaunchError(f"{source} is a bool, not a device buffer")
    if isinstance(buffer, int):
        return _pointer_value(buffer, source=source)
    if isinstance(buffer, ctypes.c_void_p):
        return _pointer_value(buffer.value, source=source)

    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        return _pointer_value(data_ptr(), source=f"{source}.data_ptr()")

    cuda_array_interface = getattr(buffer, "__cuda_array_interface__", None)
    if cuda_array_interface is not None:
        try:
            data = cuda_array_interface["data"]
            address = data[0]
        except (KeyError, IndexError, TypeError) as exc:
            raise CudaTileLaunchError(
                f"{source} has an invalid __cuda_array_interface__['data'] entry"
            ) from exc
        return _pointer_value(
            address,
            source=f"{source}.__cuda_array_interface__['data'][0]",
        )

    # DkgDSL's TensorDescriptor exposes the already-resolved address through
    # its ``pointer`` property.  Supporting it here also helps small wrappers
    # that use the same protocol.
    if hasattr(buffer, "pointer"):
        return _pointer_value(getattr(buffer, "pointer"), source=f"{source}.pointer")

    raise CudaTileLaunchError(
        f"{source} does not expose a device address; expected data_ptr(), "
        "__cuda_array_interface__, a pointer attribute, or a raw integer"
    )


def _is_null_handle(handle: object) -> bool:
    if handle is None:
        return True
    try:
        return _integer_value(handle) == 0
    except (TypeError, ValueError, OverflowError):
        return False


def _ensure_context(driver: Any, device: int) -> object:
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise CudaTileLaunchError(f"device must be a non-negative integer, got {device!r}")

    _check_cuda(driver, driver.cuInit(0), "cuInit")
    current = _check_cuda(driver, driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if not _is_null_handle(current):
        return current

    context = _retained_primary_contexts.get(device)
    if context is None:
        cuda_device = _check_cuda(
            driver, driver.cuDeviceGet(device), "cuDeviceGet"
        )
        context = _check_cuda(
            driver,
            driver.cuDevicePrimaryCtxRetain(cuda_device),
            "cuDevicePrimaryCtxRetain",
        )
        _retained_primary_contexts[device] = context

    _check_cuda(driver, driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    return context


def _normalize_stream(driver: Any, stream: object | int | None) -> object:
    if stream is None:
        return driver.CUstream(0)
    if isinstance(stream, bool):
        raise CudaTileLaunchError("stream must be a CUDA stream handle, not a bool")
    if isinstance(stream, int):
        return driver.CUstream(stream)

    # torch.cuda.Stream exposes its Driver API-compatible handle here.
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return driver.CUstream(_integer_value(cuda_stream))
    return stream


def _load_library(driver: Any, bytecode_path: Path) -> object:
    return _check_cuda(
        driver,
        driver.cuLibraryLoadFromFile(
            fileName=os.fsencode(bytecode_path),
            jitOptions=None,
            jitOptionsValues=None,
            numJitOptions=0,
            libraryOptions=None,
            libraryOptionValues=None,
            numLibraryOptions=0,
        ),
        "cuLibraryLoadFromFile",
    )


def _select_kernel(
    driver: Any, library: object, entry_name_candidates: list[str]
) -> object:
    not_found = _integer_value(
        getattr(getattr(driver, "CUresult", object), "CUDA_ERROR_NOT_FOUND", 500)
    )
    for entry_name in entry_name_candidates:
        operation = f"cuLibraryGetKernel({entry_name!r})"
        result = driver.cuLibraryGetKernel(library, entry_name.encode("utf-8"))
        if not isinstance(result, tuple) or not result:
            raise CudaTileLaunchError(
                f"{operation} returned an unexpected cuda-python result: {result!r}"
            )
        error = result[0]
        error_code = _integer_value(error)
        if error_code == 0:
            return _check_cuda(driver, result, operation)
        if error_code != not_found:
            raise CudaError(
                operation,
                error_code,
                _cuda_error_description(driver, error),
            )

    raise CudaTileLaunchError(
        "none of the symbols in the CUDA Tile bytecode string table resolved "
        "to a kernel entry point"
    )


def _pack_kernel_arguments(addresses: list[int]) -> tuple[list[ctypes.c_uint64], object, int]:
    """Build CUDA's ``void **kernelParams`` representation.

    Both returned containers must stay alive until ``cuLaunchKernel`` returns.
    """
    argument_values = [ctypes.c_uint64(address) for address in addresses]
    if not argument_values:
        return argument_values, None, 0

    pointer_array_type = ctypes.c_void_p * len(argument_values)
    pointer_array = pointer_array_type(
        *(ctypes.addressof(argument) for argument in argument_values)
    )
    return argument_values, pointer_array, ctypes.addressof(pointer_array)


def launch(
    bytecode_path: str | os.PathLike[str],
    *buffers: object,
    grid: tuple[int, int, int] = (1, 1, 1),
    stream: object | int | None = None,
    device: int = 0,
) -> None:
    """Load and synchronously launch CUDA Tile IR bytecode.

    Parameters
    ----------
    bytecode_path:
        Path to CUDA Tile IR bytecode accepted by ``cuLibraryLoadFromFile``.
    *buffers:
        Device buffers in kernel-argument order.  Each argument is passed as
        its 64-bit device address.  Supported objects expose ``data_ptr()``
        (PyTorch), ``__cuda_array_interface__`` (CuPy/Numba), or ``pointer``
        (DkgDSL TensorDescriptor).  Raw integer addresses are also accepted.
    grid:
        Logical CUDA Tile program grid.  This is not the physical CTA size.
    stream:
        CUDA stream handle, integer handle, or ``torch.cuda.Stream``.  The
        legacy default stream is used when omitted.  The stream is synchronized
        before this function returns.
    device:
        Device whose primary context is retained if no context is current.
    """
    path = _validate_bytecode_path(bytecode_path)
    grid_dims = _validate_grid(grid)
    addresses = [_device_pointer(buffer, index) for index, buffer in enumerate(buffers)]
    entry_names = _entry_name_candidates(path)

    driver = _load_driver()
    _ensure_context(driver, device)
    launch_stream = _normalize_stream(driver, stream)
    library = _load_library(driver, path)

    primary_error: BaseException | None = None
    try:
        kernel = _select_kernel(driver, library, entry_names)
        function = _check_cuda(
            driver, driver.cuKernelGetFunction(kernel), "cuKernelGetFunction"
        )
        argument_values, pointer_array, kernel_params = _pack_kernel_arguments(addresses)

        # Keep argument_values and pointer_array live across the binding call.
        # The local references are intentionally used after packing.
        _check_cuda(
            driver,
            driver.cuLaunchKernel(
                function,
                grid_dims[0],
                grid_dims[1],
                grid_dims[2],
                1,
                1,
                1,
                0,
                launch_stream,
                kernel_params,
                0,
            ),
            "cuLaunchKernel",
        )
        _check_cuda(
            driver,
            driver.cuStreamSynchronize(launch_stream),
            "cuStreamSynchronize",
        )
        _ = argument_values, pointer_array
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _check_cuda(driver, driver.cuLibraryUnload(library), "cuLibraryUnload")
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Additionally, cleanup failed: {cleanup_error}")
