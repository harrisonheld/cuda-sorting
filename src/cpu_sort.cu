#include "sort_kernels.cuh"
#include <algorithm>

void cpu_sort(int* arr, int n) {
    std::sort(arr, arr + n);
}
