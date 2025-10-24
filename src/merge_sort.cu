#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

__global__ void merge_kernel(int* A, int* B, size_t width, size_t n) {
    size_t low = (blockIdx.x * blockDim.x + threadIdx.x) * (2 * width);
    if (low >= n) return;

    size_t mid = min(low + width, n);
    size_t high = min(low + 2 * width, n);

    size_t i = low, j = mid, k = low;

    while (i < mid && j < high) {
        if (A[i] <= A[j]) B[k++] = A[i++];
        else B[k++] = A[j++];
    }

    while (i < mid) B[k++] = A[i++];
    while (j < high) B[k++] = A[j++];
}

// O(1) + O(log n) time
void merge_sort(int* A, size_t n) {
    int* d_A;
    int* d_B;

    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;

    // iterative parallel merge sort
    // notes for paper: 
    // We implement MergeSort iteratively rather than recursively
    // because CUDA kernels do not support recursion efficiently and dynamic
    // task spawning is costly. Instead of recursive calls like in the book
    // (splitting A into B and C and merging), we double the "width" of subarrays
    // in each iteration and merge them in parallel using a kernel.
    for (size_t width = 1; width < n; width *= 2) {
        size_t numThreads = (n + 2 * width - 1) / (2 * width);
        size_t numBlocks = (numThreads + blockSize - 1) / blockSize;

        merge_kernel<<<numBlocks, blockSize>>>(d_A, d_B, width, n);
        cudaDeviceSynchronize();

        std::swap(d_A, d_B);
    }

    cudaMemcpy(A, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}