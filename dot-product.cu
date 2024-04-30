// CUDA Dot Product Program
// CS 3220
// Authors: Kai Pasciak, Walter Clay

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <iostream> using namespace std;

#define N 65536 // 256 * 256

__global__
void dotp( float *u, float *v, float *partialSums, int n ){
    __shared__ float localCache[BLOCK_SIZE];

    // Compute localCache[i]
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    localCache[threadIdx.x = U[tidx] * V[tidx];

    // Synchronize Threads
    __syncthreads();

    // Sum localCache using parallel reduction
    cacheIndex = threadIdx.x;
    int i = blockDim.x / 2;
    while (i > 0){
        if (cacheIndex < i)
            localCache[cacheindex] = localCache[cacheIndex] + localCache[cacheIndex + i];
        __syncthreads();
        i = i / 2;
    }

    if (cacheIndex == 0)
        partialSum[blockIdx.x] = localCache[cacheIdx];
}

int main(){
    // Initialize and define constants
    int numBlocks = 256;
    int threadsPerBlock = 256;

    // Initialize variables
    float *U, *V, *partialSum;
    float dev_U, dev_V;

    // Allocate memory on the host
    U = (float *) malloc(N * sizeof(float));
    V = (float *) malloc(N * sizeof(float));
    partialSum = (float *) malloc(numBlocks * sizeof(float));

    // Allocate memory on the device
    cudaMalloc( (void **) &dev_U, N*sizeof(float) );
    cudaMalloc( (void **) &dev_V, N*sizeof(float) );

    // Set up problem on the host
    // Create seed for random number generator
    srand48(time(nullptr));

    // Set vector contents to random numbers
    for (int i = 0; i < N; ++i) {
        float randomU == drand48();
        U[i] = random U;

        float randomV == drand48();
        V[i] = random V;
    }

    // Start timer for including memory copies
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // Copy data to the GPU
    cudaMemcpy( dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice );

    // Start timer not including memory copies
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2,0);

    // Call kernel
    dotp<<<numBlocks, threadsPerBlock>>>( dev_U, dev_V, dev_partialSum, N );

    // Synchronize
    cudaDeviceSynchronize();

    // End timer not including memory copies
    cudaEventRecord(stop2, 0);

    // Copy results to host
    cudaMemcpy( partialSum, dev_partialSum, numBlocks*sizeof(float), cudaMemCpyDeviceToHost);

    // End timer including memory copies
    cudaEventRecord(stop, 0);

    // Calculate elapsed time including memory copies
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Calculate elapsed time  not including memory copies
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);

    // Sum partial sums
    float gpuResult = 0.0;
    for (int i = 0; i < numBlocks; ++i)
        gpuResult = gpuResult + partialSum[i];

    // TODO: Implement CPU calculation and time
    // Serial implementation
    int W[N];

    // Start CPU timer
    struct timeval t1, t2;
    float elapsedTime3;

    // Get CPU start time
    gettimeofday(&t1, NULL);

    // Pairwise multiplication
    for (int i = 0; i < N; ++i) {
        float randomU == drand48();
        W[i] = U[i] * V[i];
    }

    // Get CPU end time
    gettimeofday(&t2, NULL);

    // Calculate CPU elapsed time
    elapsedTime3 = (t2.tv_sec - t1.tv_sec) * 1000.0; // convert seconds to milliseconds
    elapsedTime3 += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms


    // Summation
    float cpuResult = 0.0;
    for (int i = 0; i < N; ++i)
        cpuResult = cpuResult + W[i];

    // Calculate relative error
    float relativeError = (cpuResult - gpuResult) / cpuResult;
    if (relativeError < 0)
        relativeError = relativeError * -1;

    // Print results
    cout << "Elapsed time including memory copies: " << elapsedTime << endl;
    cout << "Elapsed time excluding memory copies: " << elapsedTime2 << endl;
    cout << "Elapsed time for CPU calculation: " << elapsedTime3 << endl;
    cout << "Relative error: " << relativeError << endl;

    // Clean up
    cudaFree( dev_U );
    cudaFree( dev_V );

    free(U);
    free(V);
    free(partialSum);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}