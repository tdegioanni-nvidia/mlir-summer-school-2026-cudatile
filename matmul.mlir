cuda_tile.module @matmul {
  // Computes C = A * B for row-major 256x256 matrices using a 16x16 grid.
  // Each program computes one 16x16 output tile.
  entry @matmul_256x256x256(
      %a: tile<ptr<f16>>,
      %b: tile<ptr<f16>>,
      %c: tile<ptr<f32>>) {

  // ...

  }
}
