#pragma once
#include <cuda_runtime.h>

void cpu_sort(int* arr, int n);
void merge_sort(int* arr, int n);
void quick_sort(int* arr, int n);
void radix_sort(int* arr, int n);
void bitonic_sort(int* arr, int n);
void brick_sort(int* arr, int n);
