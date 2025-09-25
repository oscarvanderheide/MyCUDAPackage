#include <cuda_runtime.h>

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// This is the C-callable wrapper function that we will call from Julia.
extern "C" void add_vectors(const float *a, const float *b, float *c, int n) {
  int threads_per_block = 256;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

  add_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);

  // It's good practice to check for errors, though we'll keep it simple here.
  cudaDeviceSynchronize();
}
