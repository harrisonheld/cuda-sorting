#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "sort_kernels.cuh"

void print_array(int* arr, int n) {
    for (int i = 0; i < n; i++) std::cout << arr[i] << " ";
    std::cout << "\n";
}

int main() {
    const int N = 1 << 24;
    std::vector<int> arr(N);
    for (int i = 0; i < N; ++i) arr[i] = rand() % 100000;

    int* copy = new int[N];

    auto test_sort = [&](const char* name, void(*sort_func)(int*, int)) {
        std::copy(arr.begin(), arr.end(), copy);
        auto start = std::chrono::high_resolution_clock::now();
        sort_func(copy, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << name << " took " << diff.count() << "s to sort " << N << " elements.\n";
    };

    test_sort("Merge Sort", merge_sort);
    test_sort("Quick Sort", quick_sort);
    test_sort("Radix Sort", radix_sort);
    test_sort("Bitonic Sort", bitonic_sort);
    test_sort("Brick Sort", brick_sort);

    delete[] copy;
    return 0;
}
