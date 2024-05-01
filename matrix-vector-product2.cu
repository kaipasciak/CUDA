// CUDA Matrix Vector Product Program
// CS 3220
// Authors: Kai Pasciak, Walter Clay

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5000  // Dimension of the matrix and vector
#define THREADS_PER_BLOCK 256  // Number of threads per block

using namespace std;

// CUDA kernel for performing matrix-vector multiplication
__global__ void MxV(float *M, float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    if (row < n) {
        float sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += M[row * n + j] * x[j];  // Perform the dot product
        }
        y[row] = sum;  // Store the result in y
    }
}

// CPU function for matrix-vector multiplication
void matrixVectorProductCPU(float *M, float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += M[i * n + j] * x[j];
        }
    }
}

// Function to calculate the Euclidean norm of a vector
float vectorNorm(float *v, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

int main() {
    float *M, *x, *y, *y_cpu;
    float *dev_M, *dev_x, *dev_y;
    int size = N * N * sizeof(float);
    int vectorSize = N * sizeof(float);

    // Allocate memory on the host
    M = (float *)malloc(size);
    x = (float *)malloc(vectorSize);
    y = (float *)malloc(vectorSize);
    y_cpu = (float *)malloc(vectorSize);

    // Allocate memory on the device
    cudaMalloc(&dev_M, size);
    cudaMalloc(&dev_x, vectorSize);
    cudaMalloc(&dev_y, vectorSize);

    // Initialize matrix M and vector x with random values
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N * N; i++) {
        M[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    // Copy data from host to device
    cudaMemcpy(dev_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, vectorSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    // Setup timing variables for GPU computation excluding memory copy
    cudaEvent_t startCompute, stopCompute;
    cudaEventCreate(&startCompute);
    cudaEventCreate(&stopCompute);
    cudaEventRecord(startCompute);

    // Launch the kernel
    MxV<<<blocks, threads>>>(dev_M, dev_x, dev_y, N);
    cudaDeviceSynchronize();  // Ensure all threads have finished

    // Stop timing after synchronization
    cudaEventRecord(stopCompute);
    cudaEventSynchronize(stopCompute);

    // Measure elapsed time for the computation only
    float computeMilliseconds = 0;
    cudaEventElapsedTime(&computeMilliseconds, startCompute, stopCompute);
    cout << "GPU Compute Time (excluding memory copies): " << computeMilliseconds << " ms" << endl;

    // Copy result back to host
    cudaMemcpy(y, dev_y, vectorSize, cudaMemcpyDeviceToHost);

    // Perform computation on the CPU
    matrixVectorProductCPU(M, x, y_cpu, N);

    // Compute relative error
    float relativeError = vectorNorm(y, N) / vectorNorm(y_cpu, N);
    cout << "Relative Error: " << relativeError << endl;

    // Clean up
    cudaFree(dev_M);
    cudaFree(dev_x);
    cudaFree(dev_y);
    free(M);
    free(x);
    free(y);
    free(y_cpu);

    cudaEventDestroy(startCompute);
    cudaEventDestroy(stopCompute);

    return 0;
}
