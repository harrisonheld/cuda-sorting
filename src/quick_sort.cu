#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <stack>
#include <utility>
#include <iostream>

#define BLOCK_SIZE 256

__device__ inline void swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

// aka dutch national flag
__global__ void three_way_partition_kernel(int* arr, int low, int high, int pivot, int* d_p, int* d_q) {
    int p = low;
    int q = low;
    int k = high;

    while (q < k) {
        int val = arr[q];
        if (val < pivot) {
            swap(arr[p], arr[q]);
            p++; 
            q++;
        } else if (val > pivot) {
            k--;
            swap(arr[q], arr[k]);
        } else {
            q++;
        }
    }

    *d_p = p;
    *d_q = q;
}

void quick_sort(int* arr, size_t n) {
    if (n < 2)
        return;

    int* d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    struct Range { int low, high; };
    std::stack<Range> stack;
    stack.push({0, (int)n});

    int* d_p; int* d_q;
    cudaMalloc(&d_p, sizeof(int));
    cudaMalloc(&d_q, sizeof(int));

    while (!stack.empty()) {
        Range r = stack.top(); stack.pop();
        if (r.high - r.low < 2) continue;

        // pivot is arbitrary
        // note for paper: considered just choosing the midpoint,
        // but internet research shows that random pivot improves the average case time
        int pivot_idx = r.low + rand() % (r.high - r.low);
        int pivot;
        cudaMemcpy(&pivot, d_arr + pivot_idx, sizeof(int), cudaMemcpyDeviceToHost);

        three_way_partition_kernel<<<1, 1>>>(d_arr, r.low, r.high, pivot, d_p, d_q);
        cudaDeviceSynchronize();

        int p, q;
        cudaMemcpy(&p, d_p, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&q, d_q, sizeof(int), cudaMemcpyDeviceToHost);

        if (p > r.low)
        stack.push({r.low, p});
        if (r.high > q)
            stack.push({q, r.high});
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_p);
    cudaFree(d_q);
}
