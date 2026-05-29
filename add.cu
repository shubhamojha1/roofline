#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int n){
    // kernel function definition.
    // __global__ marks it as a cuda kernel (runs on GPU)
    // i = index
    // each thread calculates a unique index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

int main(){
    int n = 1024;
    int size = n * sizeof(int);

    int *h_a = (int*)malloc(size); // CPU Arrays
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    for(int i=0;i<n;i++){
        h_a[i] = i; // [0,1,2,3 ... 1023]
        h_b[i] = i*2; // [0,2,4 ... 2046]
    }

    int *d_a, *d_b, *d_c; 
    cudaMalloc(&d_a, size); // allocate memory on GPU
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // reserve space on the GPU (VRAM)

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); // copy data CPU -> GPU
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add<<<5, 256>>>(d_a, d_b, d_c, n); // launch kernel
    // 5 blocks, 256 threads per block = 1280 threads total

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // copy result back

    printf("c[0] = %d, c[1023] = %d\n", h_c[0], h_c[1023]);
}
/*
Grid
 └── Blocks
      └── Threads
Grid = group of blocks
Block = Group of threads
Thread = executes kernel

add<<<1, 256>>>(A,B,C) ==> 1 block, 256 threads
*/