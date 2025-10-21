#include "sort_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

void cpu_sort(int* arr, int n) {
    std::sort(arr, arr + n);
}

void merge_sort(int* arr, int n) {
    return; // TODO
}

void quick_sort(int* arr, int n) {
    return; // TODO
}

void radix_sort(int* arr, int n) {
    return; // TODO
}

void bitonic_sort(int* arr, int n) {
    return; // TODO
}

void brick_sort(int* arr, int n) {
    return; // TODO
}