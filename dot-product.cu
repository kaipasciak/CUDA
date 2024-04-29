// CUDA Dot Product Program
// CS 3220
// Authors: Kai Pasciak, Walter Clay

#include <stdio.h>
#include <stdlib.h>

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
    U = (float *) malloc(N * sizeof(float));
    V = (float *) malloc(N * sizeof(float));
    partialSum = (float *) malloc(numBlocks * sizeof(float));
    for (int i = 0; i < N; ++i) {
        U[i] = (float) (i + 1);
        V[i] = 1.0 / U[i];
    }

    cudaMemcpy( dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice );

    // Start timer
    // TODO: Implement events for elapsed time
    dotp<<<numBlocks, threadsPerBlock>>>( dev_U, dev_V, dev_partialSum, N );
    cudaDeviceSynchronize();
    cudaMemcpy( partialSum, dev_partialSum, numBlocks*sizeof(float), cudamemcpyDeviceToHost);

    // Sum partial sums
    float gpuResult = 0.0;
    for (int i = 0; i < numBlocks; ++i)
        gpuResult = gpuResult + partialSum[i];

    return 0;
}