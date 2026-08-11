cuda_tile.module @token_example_module {
  entry @token_example(%scratch_ptr: tile<ptr<f32>>) {
    %scratch_tv = make_tensor_view %scratch_ptr, shape=[1], strides=[1]
      : tensor_view<1xf32, strides=[1]>
    %scratch_view = make_partition_view %scratch_tv
      : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>

    %c0 = constant <i32: 0> : tile<i32>
    %one = constant <f32: 1.0> : tile<1xf32>

    %store_done = store_view_tko weak %one, %scratch_view[%c0]
      : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token

    %loaded, %load_done = load_view_tko weak %scratch_view[%c0]
      : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> tile<1xf32>, token

    %print_done = print_tko "loaded = %f\n", %loaded token = %load_done
      : tile<1xf32> -> token
    return
  }
}
