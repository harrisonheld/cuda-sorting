#include "sort_kernels.cuh"
#include <vector>
#include <algorithm>

// expect O(n) time
void radix_sort_sequential(int* arr, size_t n) {
    int max_val = *std::max_element(arr, arr + n);

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        std::vector<int> output(n);
        int count[10] = {0}; // init all elements to 0

        // count digit occurrences
        for (size_t i = 0; i < n; i++)
            count[(arr[i] / exp) % 10]++;

        // prefix sum (O(1) as the array size is constant 10)
        for (int i = 1; i < 10; i++)
            count[i] += count[i - 1];

        // this is the StableSort line from the pseudocode
        for (int i = n - 1; i >= 0; i--) {
            int digit = (arr[i] / exp) % 10;
            output[count[digit] - 1] = arr[i];
            count[digit]--;
        }

        std::copy(output.begin(), output.end(), arr);
    }
}