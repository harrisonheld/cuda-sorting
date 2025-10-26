#pragma once
#include <cuda_runtime.h>

void cpu_sort(int* arr, size_t n);
void merge_sort(int* arr, size_t n);
void quick_sort(int* arr, size_t n);
void radix_sort_sequential(int* arr, size_t n);
void radix_sort_parallel(int* arr, size_t n);
void bitonic_sort(int* arr, size_t n);
void brick_sort(int* arr, size_t n);
