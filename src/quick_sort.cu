#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define RECURSION_LIMIT 24
#define ARR_LENGTH_MIN  128

// __device__ means this code runs on the GPU and is callable from GPU code
__device__ void sequential_device_sort(int* data, size_t left, size_t right)
{
    for (size_t i = left + 1; i <= right; i++) {
        int key = data[i];
        size_t j = i;
        while (j > left && data[j-1] > key) {
            data[j] = data[j-1];
            j--;
        }
        data[j] = key;
    }
}

// quicksort_kernel, to be used recursively
__global__ void quicksort_kernel(int* data, size_t left, size_t right, int depth)
{
    if (depth >= RECURSION_LIMIT || (right - left) <= ARR_LENGTH_MIN) {
        sequential_device_sort(data, left, right);
        return;
    }

    size_t pivot_idx = (left + right) / 2;
    int pivot_val = data[pivot_idx];

    int* start_ptr = data + left;
    int* end_ptr   = data + right;

    // partition around pivot
    while (start_ptr <= end_ptr)
    {
        while (*start_ptr < pivot_val) start_ptr++;
        while (*end_ptr > pivot_val) end_ptr--;

        if (start_ptr <= end_ptr)
        {
            int tmp = *start_ptr;
            *start_ptr = *end_ptr;
            *end_ptr = tmp;

            start_ptr++;
            end_ptr--;
        }
    }

    size_t left_limit  = end_ptr - data;
    size_t right_start = start_ptr - data;

    // recursive call for left partition
    if (left < left_limit)
    {
        cudaStream_t left_stream;
        cudaStreamCreateWithFlags(&left_stream, cudaStreamNonBlocking);
        quicksort_kernel<<<1,1,0,left_stream>>>(data, left, left_limit, depth+1);
        cudaStreamDestroy(left_stream);
    }
    // recursive call for right partition
    if (right_start < right)
    {
        cudaStream_t right_stream;
        cudaStreamCreateWithFlags(&right_stream, cudaStreamNonBlocking);
        quicksort_kernel<<<1,1,0,right_stream>>>(data, right_start, right, depth+1);
        cudaStreamDestroy(right_stream);
    }
}


void quick_sort(int* arr, size_t n)
{
    if (n <= 1) return;
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, RECURSION_LIMIT);

    // copy to GPU
    int* d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // launch recursive partition
    quicksort_kernel<<<1,1>>>(d_arr, 0, (int)n - 1, 0);
    cudaDeviceSynchronize();

    // copy sorted from GPU onto CPU
    // free memory off GPU
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}
