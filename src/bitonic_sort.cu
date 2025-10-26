#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <climits>

__device__ inline void swap_device(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

// stage j and subsequence size k
__global__ void bitonic_kernel(int* A, int n, int j, int k) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    unsigned int ixj = i ^ j;
    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        if ((ascending && A[i] > A[ixj]) || (!ascending && A[i] < A[ixj])) {
            int tmp = A[i];
            A[i] = A[ixj];
            A[ixj] = tmp;
        }
    }
}

// O(log^2 n) time
// O(n log^2 n) work
void bitonic_sort(int* arr, size_t n) {
    // n must be a power of 2!!
    // all our test cases will be powers of 2

    int* d_A;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMemcpy(d_A, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_kernel<<<blocks, threads>>>(d_A, n, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}
