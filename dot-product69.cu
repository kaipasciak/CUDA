#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>

using namespace std;

#define N 65536 // 256 * 256

__global__ void dotp( float *u, float *v, float *partialSums, int n ){
    __shared__ float localCache[256];

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < n) {
        localCache[threadIdx.x] = u[tidx] * v[tidx];

        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i /= 2){
            if (threadIdx.x < i)
                localCache[threadIdx.x] += localCache[threadIdx.x + i];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            partialSums[blockIdx.x] = localCache[0];
    }
}

int main(){
    int numBlocks = 256;
    int threadsPerBlock = 256;

    float *U, *V, *partialSum, *dev_U, *dev_V, *dev_partialSum;

    U = (float *) malloc(N * sizeof(float));
    V = (float *) malloc(N * sizeof(float));
    partialSum = (float *) malloc(numBlocks * sizeof(float));

    cudaMalloc(&dev_U, N*sizeof(float));
    cudaMalloc(&dev_V, N*sizeof(float));
    cudaMalloc(&dev_partialSum, numBlocks*sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        U[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaEvent_t start, stop, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    cudaMemcpy(dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2,0);

    dotp<<<numBlocks, threadsPerBlock>>>(dev_U, dev_V, dev_partialSum, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop2, 0);
    cudaMemcpy(partialSum, dev_partialSum, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    float elapsedTime, elapsedTime2;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Elapased Time : " << elapsedTime << "ms" << endl;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    cout << "Elapsed Time : " << elapsedTime2 << "ms" << endl;

    float gpuResult = 0.0;
    for (int i = 0; i < numBlocks; ++i)
        gpuResult += partialSum[i];

    float *W = new float[N];
    struct timeval t1, t2;
    float elapsedTime3;
    gettimeofday(&t1, NULL);

    for (int i = 0; i < N; ++i)
        W[i] = U[i] * V[i];

    gettimeofday(&t2, NULL);
    elapsedTime3 = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime3 += (t2.tv_usec - t1.tv_usec) / 1000.0;

    float cpuResult = 0.0;
    for (int i = 0; i < N; ++i)
        cpuResult += W[i];

    float relativeError = fabs((cpuResult - gpuResult) / cpuResult);

    cout << "Elapsed time including memory copies: " << elapsedTime << " ms" << endl;
    cout << "Elapsed time excluding memory copies: " << elapsedTime2 << " ms" << endl;
    cout << "Elapsed time for CPU calculation: " << elapsedTime3 << " ms" << endl;
    cout << "Relative error: " << relativeError << endl;

    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_partialSum);
    free(U);
    free(V);
    free(partialSum);
    delete[] W;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return 0;
}
