using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>
// #include "is.hpp"

constexpr int unroll = 4;
constexpr int batch = 8;

typedef double double4_t __attribute__((vector_size(unroll * sizeof(double))));

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
struct Sums {
    int nx, ny;
    vector<float> sum;
};

static Sums prefix_sum(const float *data, int ny, int nx) {
    Sums pc;
    pc.nx = nx;
    pc.ny = ny;
    int stride = nx + 1;
    
    pc.sum.assign(stride * (ny+1), 0.f);

    for (int h = 1; h <= ny; h++) {
        for (int w = 1; w <= nx; w++) {
            float d = data[(w-1) + nx*(h-1)];
            pc.sum[w + stride*h] = d + pc.sum[w + stride*(h-1)];
        }
    }

    #pragma omp parallel for
    for (int h = 1; h <= ny; h++) {
        for (int w = 1; w <= nx; w++) {
            pc.sum[w + stride*h]    += pc.sum[(w-1) + stride*h];
        }
    }
    return pc;
}

static float get_sum(const Sums& pc, int x0, int x1, int y0, int y1) {
    int stride = pc.nx + 1;
    return pc.sum[x1 + stride*y1]
         - pc.sum[x0 + stride*y1]
         - pc.sum[x1 + stride*y0]
         + pc.sum[x0 + stride*y0];
}

static void get_inverse(vector<float>& inv_inner, vector<float>& inv_outer, int nx, int ny) {
    int total = nx*ny;
    int stride = nx + 1;
    for (int i = 1; i <= ny; i++) {
        for (int j = 1; j <= nx; j++) {
            int n_inner = i*j;
            int n_outer = total - n_inner;
            inv_inner[j + i*stride] = 1.f/n_inner;
            inv_outer[j + i*stride] = (n_outer > 0) ? 1.f/n_outer : 0.f;
        }
    }
}

static Result opt(const Sums& pc, const float tsum) {
    int nx = pc.nx;
    int ny = pc.ny;
    int stride = nx + 1;

    Result result{0, 0, ny, nx, {0,0,0}, {0,0,0}};
    float best_loss = -numeric_limits<double>::infinity();

    vector<float> inv_inner(stride*(ny+1)), inv_outer(stride*(ny+1));
    get_inverse(inv_inner, inv_outer, nx, ny);
    
    #pragma omp parallel
    {
        Result per_thread_res = result;
        float per_thread_best = -numeric_limits<float>::infinity();

        vector<float> row_d(stride);

        #pragma omp for schedule(dynamic, 4)
        for (int y0 = ny-1;  y0 >= 0; y0--) {
            for (int y1 = y0+1; y1 <= ny; y1++) {

                const float* inv_inner_row = &inv_inner[(y1-y0)*stride];
                const float* inv_outer_row = &inv_outer[(y1-y0)*stride];

                const float* row_sum_y0 = &pc.sum[stride*y0];
                const float* row_sum_y1 = &pc.sum[stride*y1];

                for (int x = 0; x <= nx; x++) row_d[x] = row_sum_y1[x] - row_sum_y0[x];

                for (int x0 = 0; x0 < nx; x0++) {
                    float sum_x0 = row_d[x0];
                    
                    for(int x1 = x0 + 1; x1 <= nx; x1++) {
                        float s = row_d[x1]-sum_x0;
                        float r = tsum -s;
                        float L = s*s*inv_inner_row[x1-x0] + r*r*inv_outer_row[x1-x0];

                        if(L > per_thread_best) {
                            per_thread_best = L;
                            per_thread_res.y0 = y0;
                            per_thread_res.y1 = y1;
                            per_thread_res.x0 = x0;
                            per_thread_res.x1 = x1;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (per_thread_best > best_loss) {
                best_loss = per_thread_best;
                result = per_thread_res;
            }
        }
    }
    
    return result;
}

static void compute_col(Result& result, const Sums& pc, const float tsum) {
    int nx = pc.nx;
    int ny = pc.ny;

    int n_inner = (result.x1-result.x0)*(result.y1-result.y0);
    int n_outer = nx*ny-n_inner;

    float s = get_sum(pc, result.x0, result.x1, result.y0, result.y1);

    for (int c = 0; c < 3; c++) {
        result.inner[c] = float(s/n_inner);
        result.outer[c] = (n_outer > 0) ? float((tsum-s)/n_outer) : result.inner[c];
    }
}

Result segment(int ny, int nx, const float *data) {
    // Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    vector<float> monochromatic(nx*ny);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            monochromatic[j + nx*i] = data[3*j + 3*nx*i];
        }
    }

    Sums pc = prefix_sum(monochromatic.data(), ny, nx);
    
    float tsum = get_sum(pc, 0, nx, 0, ny);

    Result result = opt(pc, tsum);
    compute_col(result, pc, tsum);

    return result;
}
