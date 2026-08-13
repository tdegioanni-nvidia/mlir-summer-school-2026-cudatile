# CUDA Tile matmul operation cheat sheet

`%...` stands for an already-defined SSA value.

## `cuda_tile.module`

```mlir
cuda_tile.module @matmul { ... }
```

Creates the top-level CUDA Tile compilation unit.

## `entry`

```mlir
entry @matmul(%a: tile<ptr<f16>>, %b: tile<ptr<f16>>, %c: tile<ptr<f32>>) { ... }
```

Defines a host-launchable tile kernel with pointer arguments.

## `make_tensor_view`

```mlir
%a_tensor = make_tensor_view %a, shape = [256, 256], strides = [256, 1]
    : tensor_view<256x256xf16, strides=[256,1]>
```

Describes a shaped, strided tensor in global memory starting at a pointer.

## `make_partition_view`

```mlir
%a_view = make_partition_view %a_tensor
    : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>
```

Partitions a tensor view into an indexable grid of fixed-size tiles.

## `get_tile_block_id`

```mlir
%pid_m, %pid_n, %pid_z = get_tile_block_id : tile<i32>
```

Returns the current tile block's three grid coordinates.

## `get_index_space_shape`

```mlir
%tiles_m, %tiles_k = get_index_space_shape %a_view
    : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>> -> tile<i32>
```

Returns the number of addressable tiles along each view dimension.

## `constant`

```mlir
%zero = constant <f32: 0.0> : tile<16x16xf32>
```

Creates a tile filled with a scalar value (or with explicitly listed values).

## `for`

```mlir
%result = for %k in (%c0 to %tiles_k, step %c1) : tile<i32>
    iter_values(%acc = %zero) -> (tile<16x16xf32>) { ... }
```

Iterates over a half-open integer range while carrying the accumulator between iterations.

## `load_view_tko`

```mlir
%a_tile, %token = load_view_tko weak %a_view[%pid_m, %k]
    token = %prev_token // optional
    : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>,
      tile<i32> -> tile<16x16xf16>, token
```

Loads one indexed tile from a view and returns a memory-ordering token.

## `store_view_tko`

```mlir
%token = store_view_tko weak %result, %c_view[%pid_m, %pid_n]
    token = %prev_token // optional
    : tile<16x16xf32>,
      partition_view<tile=(16x16), tensor_view<256x256xf32, strides=[256,1]>>,
      tile<i32> -> token
```

Stores one tile into an indexed view and returns a memory-ordering token.

## `mmaf`

```mlir
%next = mmaf %a_tile, %b_tile, %acc
    : tile<16x16xf16>, tile<16x16xf16>, tile<16x16xf32>
```

Computes floating-point matrix multiply-accumulate, `%a_tile * %b_tile + %acc`.

## `continue`

```mlir
continue %next : tile<16x16xf32>
```

Ends the current loop iteration and supplies its loop-carried value to the next one.
