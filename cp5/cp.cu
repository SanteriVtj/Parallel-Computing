using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int unroll = 8;
constexpr int tile = 16;
constexpr int threads = 64;

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
__global__ void normalize(float* __restrict__ r, const float* __restrict__ d, int nx, int ny, int nx_padded, int ny_padded) {
    int y = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (y >= ny) return;

    float avg = 0.f;
    for (int x = 0; x < nx; x++) avg += d[x + y*nx];
    avg /= nx;

    float sq = 0.f;
    for (int x = 0; x < nx; x++) {
        float norm = d[x + y*nx] - avg;
        r[x + y*nx_padded] = norm;
        sq += norm*norm;
    }

    float inv_sq = __frsqrt_rn(sq);
    for (int x = 0; x < nx; x++) r[x + y*nx_padded] *= inv_sq;
}

__global__ void corr_kernel(float* r, const float* d, int nx_padded, int ny, int ny_padded) {
    int i = unroll*(threadIdx.x + blockIdx.x * blockDim.x);
    int j = unroll*(threadIdx.y + blockIdx.y * blockDim.y);
    
    if (i >= ny_padded || j >= ny_padded) return;

    __shared__ float xx[tile][tile*unroll];
    __shared__ float yy[tile][tile*unroll];

    float v[unroll][unroll] = {};

    for (int x = 0; x < nx_padded; x += tile) {
        for (int ib = 0; ib < unroll; ib++) {
            xx[threadIdx.y][threadIdx.x*unroll + ib] = __ldg(&d[(i + ib)*nx_padded + x + threadIdx.y]);
            yy[threadIdx.x][threadIdx.y*unroll + ib] = __ldg(&d[(j + ib)*nx_padded + x + threadIdx.x]);
        }

        __syncthreads();

        // int last = min(tile, nx-x);
        for (int xa = 0; xa < tile; xa++) {
            float a[unroll];
            float b[unroll];
            for (int ib = 0; ib < unroll; ib++) {
                a[ib] = xx[xa][threadIdx.x*unroll + ib];
                b[ib] = yy[xa][threadIdx.y*unroll + ib];
            }
            for (int ib = 0; ib < unroll; ib++) {
                for (int jb = 0; jb < unroll; jb++) {
                    v[ib][jb] += __fmul_rn(a[ib], b[jb]);
                }
            }
        }

        __syncthreads();
    }

    for (int ib = 0; ib < unroll; ib++) {
        for (int jb = 0; jb < unroll; jb++) {
            if (i + ib < ny && j + jb < ny && j + jb > i + ib) {
                r[j + jb + (i + ib)*ny] = v[ib][jb];
            }
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    int nx_padded = roundup(nx, tile);
    int ny_padded = roundup(ny, tile*unroll);

    float* d_data = NULL;
    CHECK(cudaMalloc((void**)&d_data, ny * nx * sizeof(float)));
    float* d_X = NULL;
    CHECK(cudaMalloc((void**)&d_X, ny_padded * nx_padded * sizeof(float)));
    float* d_result = NULL;
    CHECK(cudaMalloc((void**)&d_result, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_X, 0, ny_padded * nx_padded * sizeof(float)));
    CHECK(cudaMemset(d_result, 0, ny * ny * sizeof(float)));

    {
        int blocks = divup(ny, threads);
        normalize<<<blocks, threads>>>(d_X, d_data, nx, ny, nx_padded, ny_padded);
        CHECK(cudaGetLastError());
    }
    
    {
        dim3 dimBlock(tile, tile);
        dim3 dimGrid(divup(ny_padded, dimBlock.x*unroll), divup(ny_padded, dimBlock.y*unroll));
        corr_kernel<<<dimGrid, dimBlock>>>(d_result, d_X, nx_padded, ny, ny_padded);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < ny; i++) result[i + i*ny] = 1.f;

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_X));
    CHECK(cudaFree(d_result));
}
