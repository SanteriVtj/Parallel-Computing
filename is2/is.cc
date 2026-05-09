using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
// #include "is.hpp"

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
    vector<double> sum;
    vector<double> sum_sq;
};

static Sums prefix_sum(const float *data, int ny, int nx) {
    Sums pc;
    pc.nx = nx;
    pc.ny = ny;
    int stride = nx + 1;
    pc.sum.assign(3 * stride * (ny+1), 0.0);
    pc.sum_sq.assign(3 * stride * (ny+1), 0.0);

    for (int h = 1; h <= ny; h++) {
        for (int w = 1; w <= nx; w++) {
            for (int c = 0; c < 3; c++) {
                double d = data[c + 3*(w-1) + 3*nx*(h-1)];
                
                pc.sum[c + 3*w + 3*stride*h] = d
                    + pc.sum[c + 3*w     + 3*stride*(h-1)]
                    + pc.sum[c + 3*(w-1) + 3*stride*h]
                    - pc.sum[c + 3*(w-1) + 3*stride*(h-1)];

                pc.sum_sq[c + 3*w + 3*stride*h] = d*d
                    + pc.sum_sq[c + 3*w     + 3*stride*(h-1)]
                    + pc.sum_sq[c + 3*(w-1) + 3*stride*h]
                    - pc.sum_sq[c + 3*(w-1) + 3*stride*(h-1)];
            }
        }
    }
    return pc;
}

static double get_sum(const Sums& pc, int c, int x0, int x1, int y0, int y1) {
    int stride = pc.nx + 1;
    return pc.sum[c + 3*x1 + 3*stride*y1]
         - pc.sum[c + 3*x0 + 3*stride*y1]
         - pc.sum[c + 3*x1 + 3*stride*y0]
         + pc.sum[c + 3*x0 + 3*stride*y0];
}

static double get_sum_sq(const Sums& pc, int c, int x0, int x1, int y0, int y1) {
    int stride = pc.nx + 1;
    return pc.sum_sq[c + 3*x1 + 3*stride*y1]
         - pc.sum_sq[c + 3*x0 + 3*stride*y1]
         - pc.sum_sq[c + 3*x1 + 3*stride*y0]
         + pc.sum_sq[c + 3*x0 + 3*stride*y0];
}

static double loss(const Sums& pc, const double tsum[3], const double tsum_sq[3], int x0, int x1, int y0, int y1) {
    int nx = pc.nx;
    int ny = pc.ny;
    
    int n_inner = (x1-x0)*(y1-y0);
    int n_outer = nx*ny-n_inner;

    double L = 0.0;

    for (int c = 0; c < 3; c++) {
        double sum = get_sum(pc, c, x0, x1, y0, y1);
        double sum_sq = get_sum_sq(pc, c, x0, x1, y0, y1);
        L += sum_sq - sum*sum/n_inner;

        if (n_outer > 0) {
            double res_sum = tsum[c] - sum;
            double res_sum_sq = tsum_sq[c] - sum_sq;
            L += res_sum_sq - (res_sum * res_sum) / n_outer;
        }
    }
    return L;
}

static Result opt(const Sums& pc, const double tsum[3], const double tsum_sq[3]) {
    int nx = pc.nx;
    int ny = pc.ny;

    Result result{0, 0, ny, nx, {0,0,0}, {0,0,0}};
    double best_loss = numeric_limits<double>::infinity();

    for (int y0 = 0; y0 < ny; y0++) {
        for (int y1 = y0+1; y1 <= ny; y1++) {
            for (int x0 = 0; x0 < nx; x0++) {
                for (int x1 = x0+1; x1 <= nx; x1++) {
                    double L = loss(pc, tsum, tsum_sq, x0, x1, y0, y1);

                    if (L < best_loss) {
                        best_loss = L;
                        result.y0 = y0;
                        result.y1 = y1;
                        result.x0 = x0;
                        result.x1 = x1;
                    }
                }
            }
        }
    }
    
    return result;
}

static void compute_col(Result& result, const Sums& pc, const double tsum[3]) {
    int nx = pc.nx;
    int ny = pc.ny;

    int n_inner = (result.x1-result.x0)*(result.y1-result.y0);
    int n_outer = nx*ny-n_inner;

    for (int c = 0; c < 3; c++) {
        double s = get_sum(pc, c, result.x0, result.x1, result.y0, result.y1);

        result.inner[c] = float(s/n_inner);
        result.outer[c] = (n_outer > 0) ? float((tsum[c]-s)/n_outer) : result.inner[c];
    }
}

Result segment(int ny, int nx, const float *data) {
    // Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    Sums pc = prefix_sum(data, ny, nx);
    
    double tsum[3], tsum_sq[3];
    for (int c = 0; c < 3; c++) {
        tsum[c] = get_sum(pc, c, 0, nx, 0, ny);
        tsum_sq[c] = get_sum_sq(pc, c, 0, nx, 0, ny);
    }

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
