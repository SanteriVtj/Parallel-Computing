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
    vector<double4_t> sum;
    vector<double4_t> sum_sq;
};

static Sums prefix_sum(const float *data, int ny, int nx) {
    Sums pc;
    pc.nx = nx;
    pc.ny = ny;
    int stride = nx + 1;
    
    pc.sum.assign(stride * (ny+1), (double4_t){0, 0, 0, 0});
    pc.sum_sq.assign(stride * (ny+1), (double4_t){0, 0, 0, 0});

    for (int h = 1; h <= ny; h++) {
        for (int w = 1; w <= nx; w++) {
            double4_t d = {
                data[    3*(w-1) + 3*nx*(h-1)],
                data[1 + 3*(w-1) + 3*nx*(h-1)],
                data[2 + 3*(w-1) + 3*nx*(h-1)],
                0.0
            };
            pc.sum[w + stride*h]    = d    + pc.sum[w + stride*(h-1)];
            pc.sum_sq[w + stride*h] = d*d  + pc.sum_sq[w + stride*(h-1)];
        }
    }

    #pragma omp parallel for
    for (int h = 1; h <= ny; h++) {
        for (int w = 1; w <= nx; w++) {
            pc.sum[w + stride*h]    += pc.sum[(w-1) + stride*h];
            pc.sum_sq[w + stride*h] += pc.sum_sq[(w-1) + stride*h];
        }
    }

    // for (int h = 1; h <= ny; h++) {
    //     for (int w = 1; w <= nx; w++) {
    //         // for (int c = 0; c < 3; c++) {
    //         double4_t d = {
    //             data[0 + 3*(w-1) + 3*nx*(h-1)],
    //             data[1 + 3*(w-1) + 3*nx*(h-1)],
    //             data[2 + 3*(w-1) + 3*nx*(h-1)],
    //             0.0
    //         };
            
    //         pc.sum[w + stride*h] = d
    //             + pc.sum[w       + stride*(h-1)]
    //             + pc.sum[(w-1)   + stride*h]
    //             - pc.sum[(w-1)   + stride*(h-1)];

    //         pc.sum_sq[w + stride*h] = d*d
    //             + pc.sum_sq[w     + stride*(h-1)]
    //             + pc.sum_sq[(w-1) + stride*h]
    //             - pc.sum_sq[(w-1) + stride*(h-1)];
    //         // }
    //     }
    // }
    return pc;
}

static double4_t get_sum(const Sums& pc, int x0, int x1, int y0, int y1) {
    int stride = pc.nx + 1;
    return pc.sum[x1 + stride*y1]
         - pc.sum[x0 + stride*y1]
         - pc.sum[x1 + stride*y0]
         + pc.sum[x0 + stride*y0];
}

static double4_t get_sum_sq(const Sums& pc, int x0, int x1, int y0, int y1) {
    int stride = pc.nx + 1;
    return pc.sum_sq[x1 + stride*y1]
         - pc.sum_sq[x0 + stride*y1]
         - pc.sum_sq[x1 + stride*y0]
         + pc.sum_sq[x0 + stride*y0];
}

static Result opt(const Sums& pc, const double4_t tsum, const double4_t tsum_sq) {
    int nx = pc.nx;
    int ny = pc.ny;
    int stride = nx + 1;

    Result result{0, 0, ny, nx, {0,0,0}, {0,0,0}};
    double best_loss = numeric_limits<double>::infinity();
    
    #pragma omp parallel
    {
        Result per_thread_res = result;
        double per_thread_best = numeric_limits<double>::infinity();

        vector<double4_t> row_d(stride);
        vector<double4_t> row_d_sq(stride);

        #pragma omp for schedule(dynamic, 4)
        for (int y0 = 0; y0 < ny; y0++) {
            for (int y1 = y0+1; y1 <= ny; y1++) {

                const double4_t* row_sum_y0 = &pc.sum[stride*y0];
                const double4_t* row_sum_y1 = &pc.sum[stride*y1];
                const double4_t* row_sum_sq_y0 = &pc.sum_sq[stride*y0];
                const double4_t* row_sum_sq_y1 = &pc.sum_sq[stride*y1];

                for (int x = 0; x <= nx; x++) {
                    row_d[x] = row_sum_y1[x] - row_sum_y0[x];
                    row_d_sq[x] = row_sum_sq_y1[x] - row_sum_sq_y0[x];
                }

                for (int x0b = 0; x0b < nx; x0b += batch) {
                    double4_t sum_x0[8];
                    double4_t sum_sq_x0[8];
                    int x0_end = min(x0b + batch, nx);

                    for (int i = 0; i < x0_end - x0b; i++) {
                        int x0 = x0b + i;
                        sum_x0[i] = row_sum_y1[x0] - row_sum_y0[x0];
                        sum_sq_x0[i] = row_sum_sq_y1[x0] - row_sum_sq_y0[x0];
                    }

                    for (int x1 = x0b + 1; x1 <= nx; x1++) {
                        double4_t rx1 = row_d[x1];
                        double4_t rx1_sq = row_d_sq[x1];

                        int i_end = min(x0_end - x0b, x1 - x0b);
                        for (int i = 0; i < i_end; i++) {
                            int x0 = x0b + i;

                            double4_t s = rx1 - sum_x0[i];
                            double4_t s_sq = rx1_sq - sum_sq_x0[i];

                            int n_inner = (x1-x0)*(y1-y0);
                            int n_outer = nx*ny - n_inner;
                            double inv_n_inner = 1.0/n_inner;
                            double inv_n_outer = (n_outer > 0) ? 1.0/n_outer : 0.0;

                            double4_t L = s_sq - s*s*inv_n_inner;

                            double4_t r = tsum - s;
                            double4_t r_sq = tsum_sq - s_sq;
                            L += r_sq - r*r*inv_n_outer;

                            double LL = L[0] + L[1] + L[2];

                            if (LL < per_thread_best) {
                                per_thread_best = LL;
                                per_thread_res.y0 = y0;
                                per_thread_res.y1 = y1;
                                per_thread_res.x0 = x0;
                                per_thread_res.x1 = x1;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (per_thread_best < best_loss) {
                best_loss = per_thread_best;
                result = per_thread_res;
            }
        }
    }
    
    return result;
}

static void compute_col(Result& result, const Sums& pc, const double4_t tsum) {
    int nx = pc.nx;
    int ny = pc.ny;

    int n_inner = (result.x1-result.x0)*(result.y1-result.y0);
    int n_outer = nx*ny-n_inner;

    double4_t s = get_sum(pc, result.x0, result.x1, result.y0, result.y1);

    for (int c = 0; c < 3; c++) {

        result.inner[c] = float(s[c]/n_inner);
        result.outer[c] = (n_outer > 0) ? float((tsum[c]-s[c])/n_outer) : result.inner[c];
    }
}

Result segment(int ny, int nx, const float *data) {
    // Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    Sums pc = prefix_sum(data, ny, nx);
    
    double4_t tsum = {0.0,0.0,0.0,0.0};
    double4_t tsum_sq = {0.0,0.0,0.0,0.0};
    // for (int c = 0; c < 3; c++) {
    tsum = get_sum(pc, 0, nx, 0, ny);
    tsum_sq = get_sum_sq(pc, 0, nx, 0, ny);
    // }

    Result result = opt(pc, tsum, tsum_sq);
    compute_col(result, pc, tsum);

    // int stride = nx + 1;
    // for (int i = 1; i < ny+1; i++) {
    //     for (int j = 1; j < nx+1; j++) {
    //         cout << pc.sum[3*j+3*stride*i] << " ";
    //     }
    //     cout << "\n";
    // }

    // double tsum[3], tsum_sq[3];
    // for (int c = 0; c < 3; c++) {
    //     tsum[c]    = get_sum(pc, c, 0, nx, 0, ny);
    //     tsum_sq[c] = get_sum_sq(pc, c, 0, nx, 0, ny);
    // }

    // double L = loss(pc, tsum, tsum_sq, 0, 1, 0, 1);

    // cout << L << "\n";

    return result;
}
