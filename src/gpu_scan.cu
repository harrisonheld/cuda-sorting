#include "sort_kernels.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

// block scan using shared memory (Blelloch scan)
__device__ void block_scan_shared(int* s_data, int n) {
    int tid = threadIdx.x;
    int offset = 1;

    // upsweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            s_data[bi] += s_data[ai];
        }
        offset *= 2;
    }

    // A[0] = 0
    if (tid == 0) s_data[n - 1] = 0;

    // downsweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            int t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
    __syncthreads();
}

__global__ void scan_block(int* in, int* out, int* block_sums, size_t n) {
    extern __shared__ int s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    s_data[tid] = (idx < n) ? in[idx] : 0;
    __syncthreads();

    block_scan_shared(s_data, blockDim.x);

    if (idx < n) out[idx] = s_data[tid];

    if (tid == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = s_data[tid] + ((idx < n) ? in[idx] : 0);
    }
}

__global__ void add_offsets(int* out, int* block_offsets, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] += block_offsets[blockIdx.x];
}

void gpu_scan(int* d_in, int* d_out, size_t n) {
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;

    int* d_block_sums;
    cudaMalloc(&d_block_sums, gridSize * sizeof(int));

    scan_block<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    if (gridSize > 1) {
        int* d_block_sums_scanned;
        cudaMalloc(&d_block_sums_scanned, gridSize * sizeof(int));

        gpu_scan(d_block_sums, d_block_sums_scanned, gridSize);

        add_offsets<<<gridSize, blockSize>>>(d_out, d_block_sums_scanned, n);

        cudaFree(d_block_sums_scanned);
    }

    cudaFree(d_block_sums);
}
