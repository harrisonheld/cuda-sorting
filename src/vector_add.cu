#include <iostream>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20; // 1M elements
    size_t size = n * sizeof(float);
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize data
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    vector_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Check result
    std::cout << "c[0] = " << c[0] << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
}
