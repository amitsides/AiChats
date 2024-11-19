#include <cuda_runtime.h>
#include <iostream>

// Matrix size (N x N)
#define N 1024
#define BLOCK_SIZE 16

// Kernel to perform matrix multiplication
__global__ void matrixMulShared(float *A, float *B, float *C, int n) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0;

    // Loop over submatrices of A and B to compute C
    for (int k = 0; k < n / BLOCK_SIZE; ++k) {
        // Load submatrices into shared memory
        Asub[threadIdx.y][threadIdx.x] = A[row * n + (k * BLOCK_SIZE + threadIdx.x)];
        Bsub[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // Synchronize threads

        // Perform computation for this submatrix
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            Cvalue += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }

        __syncthreads(); // Synchronize threads
    }

    // Write the computed value to the output matrix
    if (row < n && col < n) {
        C[row * n + col] = Cvalue;
    }
}

// Host function to initialize and call the kernel
void matrixMultiplication() {
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    matrixMultiplication();
    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    return 0;
}
