using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
__global__ void normalize(float* r, const float* d, int nx, int ny) {
    int y = threadIdx.x + blockIdx.x*blockDim.x;

    if (y >= ny) return;

    float avg = 0.f;
    for (int x = 0; x < nx; x++) avg += d[x + y*nx];
    avg /= nx;

    float sq = 0.f;
    for (int x = 0; x < nx; x++) {
        float norm = d[x + y*nx] - avg;
        r[x + y*nx] = norm;
        sq += norm*norm;
    }

    float inv_sq = 1.f / sqrtf(sq);
    for (int x = 0; x < nx; x++) r[x + y*nx] *= inv_sq;
}

__global__ void corr_kernel(float* r, const float* d, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ny || j >= ny) return;
    if (j <= i) return;

    float sum = 0.f;
    for (int x = 0; x < nx; x++) {
        sum += d[i*nx + x]*d[j*nx + x];
    }

    r[j + i*ny] = sum;
}

void correlate(int ny, int nx, const float *data, float *result) {
    float* d_data = NULL;
    CHECK(cudaMalloc((void**)&d_data, ny * nx * sizeof(float)));
    float* d_X = NULL;
    CHECK(cudaMalloc((void**)&d_X, ny * nx * sizeof(float)));
    float* d_result = NULL;
    CHECK(cudaMalloc((void**)&d_result, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_result, 0, ny * ny * sizeof(float)));

    {
        int threads = 64;
        int blocks = divup(ny, threads);
        normalize<<<blocks, threads>>>(d_X, d_data, nx, ny);
        CHECK(cudaGetLastError());
    }
    
    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
        corr_kernel<<<dimGrid, dimBlock>>>(d_result, d_X, nx, ny);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < ny; i++) result[i + i*ny] = 1.f;

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_X));
    CHECK(cudaFree(d_result));
}
