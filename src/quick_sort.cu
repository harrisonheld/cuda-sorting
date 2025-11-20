#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdlib> // for rand
#include <iostream>

#define BLOCK_SIZE 256

// GPU kernel to count items < pivot and > pivot in each block
__global__ void count_kernel(int* d_arr, int* below_arr, int* above_arr, size_t n, int pivot) {
    __shared__ int s_below[BLOCK_SIZE];
    __shared__ int s_above[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int below_count = 0;
    int above_count = 0;

    if (gid < n) {
        int val = d_arr[gid];
        if (val < pivot) below_count = 1;
        else if (val > pivot) above_count = 1;
    }

    s_below[tid] = below_count;
    s_above[tid] = above_count;
    __syncthreads();

    // Block-level reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_below[tid] += s_below[tid + offset];
            s_above[tid] += s_above[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        below_arr[blockIdx.x] = s_below[0];
        above_arr[blockIdx.x] = s_above[0];
    }
}

// Kernel to reorder elements in-place based on offsets
__global__ void reorder_kernel(int* d_arr, int* offset_below, int* offset_above, size_t n, int pivot, int total_below) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int val = d_arr[gid];
    int block_id = blockIdx.x;

    // compute idx in-place using offsets
    if (val < pivot) {
        int idx = atomicAdd(&offset_below[block_id], 1);
        d_arr[idx] = val;
    } else if (val > pivot) {
        int idx = atomicAdd(&offset_above[block_id], 1) + total_below;
        d_arr[idx] = val;
    }
}

void quick_sort(int* arr, size_t n) {
    if (n <= 1) return;

    int pivot = arr[rand() % n];

    int* d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_below;
    int* d_above;
    cudaMalloc(&d_below, num_blocks * sizeof(int));
    cudaMalloc(&d_above, num_blocks * sizeof(int));

    // 1. Count items < pivot and > pivot per block
    count_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_below, d_above, n, pivot);
    cudaDeviceSynchronize();

    // 2. Prefix sums
    int* d_offset_below;
    int* d_offset_above;
    cudaMalloc(&d_offset_below, num_blocks * sizeof(int));
    cudaMalloc(&d_offset_above, num_blocks * sizeof(int));
    gpu_scan(d_below, d_offset_below, num_blocks);
    gpu_scan(d_above, d_offset_above, num_blocks);

    // 3. Compute total items < pivot
    int total_below;
    cudaMemcpy(&total_below, &d_offset_below[num_blocks - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int last_block_count;
    cudaMemcpy(&last_block_count, &d_below[num_blocks - 1], sizeof(int), cudaMemcpyDeviceToHost);
    total_below += last_block_count; // last prefix sum element + last block count

    // 4. Reorder in-place
    reorder_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_offset_below, d_offset_above, n, pivot, total_below);
    cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_below);
    cudaFree(d_above);
    cudaFree(d_offset_below);
    cudaFree(d_offset_above);
}
