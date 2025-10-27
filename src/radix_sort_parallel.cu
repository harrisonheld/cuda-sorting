#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ inline void swap_device(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

__global__ void count_bits(int* in, int* zeros, int* ones, int bit, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int b = (in[idx] >> bit) & 1;
    zeros[idx] = 1 - b;
    ones[idx]  = b;
}

// block scan using shared memory (blelloch scan)
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

    // A[0] := 0
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

// scatter elements according to prefix sums
__global__ void scatter(int* in, int* out, int* zeros_scan, int* ones_scan, int total_zeros, int bit, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = in[idx];
    int b = (val >> bit) & 1;
    int pos = b ? total_zeros + ones_scan[idx] : zeros_scan[idx];
    out[pos] = val;
}

// Chapter 11 Question 8 is "Give a parallel algorithm for Radix Sort using parallel prefix."
void gpu_scan(int* d_in, int* d_out, int n) {
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

void radix_sort_parallel(int* arr, size_t n) {
    int* d_in;
    int* d_out;
    int* d_zeros;
    int* d_ones;
    int* d_zeros_scan;
    int* d_ones_scan;

    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_zeros, n * sizeof(int));
    cudaMalloc(&d_ones, n * sizeof(int));
    cudaMalloc(&d_zeros_scan, n * sizeof(int));
    cudaMalloc(&d_ones_scan, n * sizeof(int));

    cudaMemcpy(d_in, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int max_bits = sizeof(int) * 8;
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (int bit = 0; bit < max_bits; bit++) {
        count_bits<<<gridSize, blockSize>>>(d_in, d_zeros, d_ones, bit, n);
        cudaDeviceSynchronize();

        gpu_scan(d_zeros, d_zeros_scan, n);
        gpu_scan(d_ones, d_ones_scan, n);
        cudaDeviceSynchronize();

        int total_zeros, last_zero;
        cudaMemcpy(&last_zero, &d_zeros[n-1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&total_zeros, &d_zeros_scan[n-1], sizeof(int), cudaMemcpyDeviceToHost);
        total_zeros += last_zero;

        scatter<<<gridSize, blockSize>>>(d_in, d_out, d_zeros_scan, d_ones_scan, total_zeros, bit, n);
        cudaDeviceSynchronize();

        std::swap(d_in, d_out);
    }

    cudaMemcpy(arr, d_in, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_zeros);
    cudaFree(d_ones);
    cudaFree(d_zeros_scan);
    cudaFree(d_ones_scan);
}
