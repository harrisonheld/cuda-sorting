#include "sort_kernels.cuh"
#include <algorithm>

void cpu_sort(int* arr, size_t n) {
    std::sort(arr, arr + n);
}
