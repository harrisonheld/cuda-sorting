#include "sort_kernels.cuh"
#include <cuda_runtime.h>

#define THREADS 256

__global__ void odd_even_kernel(int* arr, size_t n, int phase, bool* d_swapped) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Odd phase: start = 1, Even phase: start = 0
    size_t idx = 2 * i + phase;
    if (idx + 1 >= n) return;

    int a = arr[idx];
    int b = arr[idx + 1];
    if (a > b) {
        arr[idx] = b;
        arr[idx + 1] = a;
        *d_swapped = true;
    }
}

// O(n) time, O(n^2) work
void brick_sort(int* arr, size_t n) {
    if (n < 2) return;

    int* d_arr;
    bool* d_swapped;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_swapped, sizeof(bool));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    size_t blocks = (n + THREADS - 1) / (2 * THREADS);

    bool h_swapped = true;
    while (h_swapped) {
        h_swapped = false;
        cudaMemcpy(d_swapped, &h_swapped, sizeof(bool), cudaMemcpyHostToDevice);

        // odd indices (1, 3, 5, ...)
        odd_even_kernel<<<blocks, THREADS>>>(d_arr, n, 1, d_swapped);
        cudaDeviceSynchronize();

        // even indices (0, 2, 4, ...)
        odd_even_kernel<<<blocks, THREADS>>>(d_arr, n, 0, d_swapped);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_swapped, d_swapped, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_swapped);
}
