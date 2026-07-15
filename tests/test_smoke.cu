// Smoke test: proves the toolchain (nvcc + driver + GPU) works end to end.
// Launches a trivial kernel, copies the result back, verifies it.
// Exit 0 = pass. Not a feature kernel — those live in kernels/.

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void fill(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * 2;
}

int main() {
    const int n = 1024;
    int *d = nullptr;
    int h[n];

    CHECK(cudaMalloc(&d, n * sizeof(int)));
    fill<<<(n + 255) / 256, 256>>>(d, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));

    for (int i = 0; i < n; ++i) {
        if (h[i] != i * 2) {
            fprintf(stderr, "mismatch at %d: got %d want %d\n", i, h[i], i * 2);
            return 1;
        }
    }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("smoke PASS on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    return 0;
}
