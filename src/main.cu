#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "sort_kernels.cuh"

void print_array(int* arr, int n) {
    for (int i = 0; i < n; i++) std::cout << arr[i] << " ";
    std::cout << "\n";
}

void test_sort_algorithm(const char* name, void(*sort_func)(int*, int), const std::vector<int>& arr) {
    int* copy = new int[arr.size()];
    std::copy(arr.begin(), arr.end(), copy);

    auto start = std::chrono::high_resolution_clock::now();
    sort_func(copy, arr.size());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << name << " took " << diff.count() 
              << "s to sort " << arr.size() << " elements.\n";

    delete[] copy;
}

int main() {
    // array sizes to test
    std::vector<int> sizes = {1 << 16, 1 << 20, 1 << 24, 1 << 28};

    for (auto n : sizes) {
        std::cout << "\nCreating array of size " << n << "\n";
        std::vector<int> arr(n);
        for (int i = 0; i < n; ++i) 
            arr[i] = rand() % 100000;

        test_sort_algorithm("CPU std::sort", cpu_sort, arr);
        test_sort_algorithm("Merge Sort", merge_sort, arr);
        test_sort_algorithm("Quick Sort", quick_sort, arr);
        test_sort_algorithm("Radix Sort", radix_sort, arr);
        test_sort_algorithm("Bitonic Sort", bitonic_sort, arr);
        test_sort_algorithm("Brick Sort", brick_sort, arr);
    }

    return 0;
}
