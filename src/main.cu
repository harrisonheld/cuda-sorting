#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "sort_kernels.cuh"

void print_array(int* arr, int n) {
    for (int i = 0; i < n; i++) std::cout << arr[i] << " ";
    std::cout << "\n";
}

bool is_sorted(const int* arr, int n, int samples = 1000) {
    std::srand(std::time(nullptr));
    for (int s = 0; s < samples; ++s) {
        int i = std::rand() % (n - 1); // random index
        if (arr[i] > arr[i + 1])
            return false; // definitely not sorted
    }
    return true; // probably sorted
}

void test_sort_algorithm(const char* name, void(*sort_func)(int*, size_t), const std::vector<int>& arr) {
    int* copy = new int[arr.size()];
    std::copy(arr.begin(), arr.end(), copy);

    auto start = std::chrono::high_resolution_clock::now();
    sort_func(copy, arr.size());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    bool correct = is_sorted(copy, arr.size());

    // CSV output: size,algorithm,time,correct
    std::cout << arr.size() << ",\"" << name << "\"," << diff.count() 
              << "," << (correct ? "PASS" : "FAIL") << "\n";

    delete[] copy;
}

int main() {
    // print csv header
    std::cout << "Size,Algorithm,Time(s),Result\n";

    std::vector<int> sizes = {1 << 12, 1<< 14, 1 << 16, 1 << 18,1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28};

    for (auto n : sizes) {
        std::vector<int> arr(n);
        for (int i = 0; i < n; ++i) 
            arr[i] = rand() % 100000;

        test_sort_algorithm("CPU std::sort", cpu_sort, arr);
        test_sort_algorithm("Merge Sort", merge_sort, arr);
        test_sort_algorithm("Quick Sort", quick_sort, arr);
        test_sort_algorithm("Radix Sort Sequential", radix_sort_sequential, arr);
        test_sort_algorithm("Radix Sort Parallel", radix_sort_parallel, arr);
        test_sort_algorithm("Bitonic Sort", bitonic_sort, arr);
        test_sort_algorithm("Brick Sort", brick_sort, arr);
    }

    return 0;
}
