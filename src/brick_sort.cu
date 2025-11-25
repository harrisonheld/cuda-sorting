#include "sort_kernels.cuh"
#include <cuda_runtime.h>

#define THREADS 256

__global__ void odd_phase(int* A, size_t n, bool* d_found) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // j = 1, 3, 5, ...  (0-indexed → j = 0+1 = 1)
    size_t j = 2 * i + 1;
    if (j + 1 >= n) return;

    if (A[j] > A[j + 1]) {
        int tmp = A[j];
        A[j] = A[j + 1];
        A[j + 1] = tmp;
        *d_found = true;
    }
}

__global__ void even_phase(int* A, size_t n, bool* d_found) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // j = 2, 4, 6, ...  (0-indexed → j = 0)
    size_t j = 2 * i;
    if (j + 1 >= n) return;

    if (A[j] > A[j + 1]) {
        int tmp = A[j];
        A[j] = A[j + 1];
        A[j + 1] = tmp;
        *d_found = true;
    }
}

// O(n) time, O(n^2) work
void brick_sort(int* arr, size_t n) {
    if (n < 2) return;

    int* d_arr;
    bool* d_found;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // number of pairs:
    // odd phase handles ceil((n-1)/2)
    // even phase handles floor((n)/2)
    size_t pairs = (n + 1) / 2;
    size_t blocks = (pairs + THREADS - 1) / THREADS;

    bool h_found = true;

    while (h_found) {
        h_found = false;
        cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

        // forall odd j in parallel
        odd_phase<<<blocks, THREADS>>>(d_arr, n, d_found);
        cudaDeviceSynchronize();

        // forall even j in parallel
        even_phase<<<blocks, THREADS>>>(d_arr, n, d_found);
        cudaDeviceSynchronize();

        // check if any swap occurred
        cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_found);
}
