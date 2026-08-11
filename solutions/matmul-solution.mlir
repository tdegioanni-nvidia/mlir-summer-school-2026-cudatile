cuda_tile.module @matmul {
  // Computes C = A * B for row-major 256x256 matrices using a 16x16 grid.
  // Each program computes one 16x16 output tile.
  entry @matmul_256x256x256(
      %a: tile<ptr<f16>>,
      %b: tile<ptr<f16>>,
      %c: tile<ptr<f32>>) {
    %a_tensor = make_tensor_view %a,
        shape = [256, 256], strides = [256, 1]
        : tensor_view<256x256xf16, strides=[256,1]>
    %b_tensor = make_tensor_view %b,
        shape = [256, 256], strides = [256, 1]
        : tensor_view<256x256xf16, strides=[256,1]>
    %c_tensor = make_tensor_view %c,
        shape = [256, 256], strides = [256, 1]
        : tensor_view<256x256xf32, strides=[256,1]>

    %a_view = make_partition_view %a_tensor
        : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>
    %b_view = make_partition_view %b_tensor
        : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>
    %c_view = make_partition_view %c_tensor
        : partition_view<tile=(16x16), tensor_view<256x256xf32, strides=[256,1]>>

    %pid_m, %pid_n, %pid_z = get_tile_block_id : tile<i32>
    %a_tiles_m, %a_tiles_k = get_index_space_shape %a_view
        : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>
          -> tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %zero = constant <f32: 0.0> : tile<16x16xf32>

    %result = for %k in (%c0 to %a_tiles_k, step %c1) : tile<i32>
        iter_values(%acc = %zero) -> (tile<16x16xf32>) {
      %a_tile, %a_token = load_view_tko weak %a_view[%pid_m, %k]
          : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>,
            tile<i32> -> tile<16x16xf16>, token
      %b_tile, %b_token = load_view_tko weak %b_view[%k, %pid_n]
          : partition_view<tile=(16x16), tensor_view<256x256xf16, strides=[256,1]>>,
            tile<i32> -> tile<16x16xf16>, token
      %next = mmaf %a_tile, %b_tile, %acc
          : tile<16x16xf16>, tile<16x16xf16>, tile<16x16xf32>
      continue %next : tile<16x16xf32>
    }

    %store_token = store_view_tko weak %result,
        %c_view[%pid_m, %pid_n]
        : tile<16x16xf32>,
          partition_view<tile=(16x16), tensor_view<256x256xf32, strides=[256,1]>>,
          tile<i32> -> token
  }
}
