#include <stdio.h>
#include <cuda_runtime.h>
#include "../profiler/timing.cuh"
__global__ void elementwise_add(const float*a, const float*b, float*c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    const int N = 1 << 20;
    float *a, *b, *c;
    // allocate memory on the host
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));

    // initialize input arrays
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }
    float *d_a, *d_b, *d_c;

    // allocate memory on the device
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    // copy input arrays from host to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    GpuTimer timer;
    timer.start();
    elementwise_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    timer.stop();

    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nTime taken: %f ms\n", timer.elapsed_ms());
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", c[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}