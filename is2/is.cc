using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "is.hpp"

// struct Result {
//     int y0;
//     int x0;
//     int y1;
//     int x1;
//     float outer[3];
//     float inner[3];
// };

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
    pc.sum.assign(3*nx*ny, 0.0);
    pc.sum_sq.assign(3*nx*ny, 0.0);

    for (int h = 0; h < ny; h++) {
        for (int w = 0; w < nx; w++) {
            for (int c = 0; c < 3; c++) {
                double d = data[c + 3 * w + 3 * nx * h];

                double s = d;
                double s_sq = d*d;

                if (h > 0) {
                    s += pc.sum[c + 3 * w + 3 * nx * (h-1)];
                    s_sq += pc.sum_sq[c + 3 * w + 3 * nx * (h-1)];
                }
                
                if (w > 0) {
                    s += pc.sum[c + 3 * (w-1) + 3 * nx * h];
                    s_sq += pc.sum_sq[c + 3 * (w-1) + 3 * nx * h];
                }
                if (h > 0 && w > 0) {
                    s -= pc.sum[c + 3 * (w-1) + 3 * nx * (h-1)];
                    s_sq -= pc.sum_sq[c + 3 * (w-1) + 3 * nx * (h-1)];
                }

                pc.sum[c + 3 * w + 3 * nx * h] = s;
                pc.sum_sq[c + 3 * w + 3 * nx * h] = s_sq;
            }
        }
    }

    return pc;
}

static double get_sum(const Sums& pc, int c, int x0, int x1, int y0, int y1) {
    int nx = pc.nx;
    int ny = pc.ny;
    return pc.sum[c + 3 * x1 + 3 * nx * y1] - pc.sum[c + 3 * x1 + 3 * nx * y0] - pc.sum[c + 3 * x0 + 3 * nx * y1] + pc.sum[c + 3 * x0 + 3 * nx * y0];
}

static double get_sum_sq(const Sums& pc, int c, int x0, int x1, int y0, int y1) {
    int nx = pc.nx;
    int ny = pc.ny;
    return pc.sum_sq[c + 3 * x1 + 3 * nx * y1] - pc.sum_sq[c + 3 * x1 + 3 * nx * y0] - pc.sum_sq[c + 3 * x0 + 3 * nx * y1] + pc.sum_sq[c + 3 * x0 + 3 * nx * y0];
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
            double res_sum_sq = tsum_sq[c]-sum_sq;
            L += (tsum_sq[c]-sum_sq)-(res_sum_sq*res_sum_sq)/n_outer;
        }
    }
    return L;
}

Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    Sums pc = prefix_sum(data, ny, nx);

    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            cout << pc.sum[3*j+3*nx*i] << " ";
        }
        cout << "\n";
    }

    double tsum[3], tsum_sq[3];
    for (int c = 0; c < 3; c++) {
        tsum[c] = get_sum(pc, c, 0, 0, ny, nx);
        tsum_sq[c] = get_sum_sq(pc, c, 0, 0, ny, nx);
    }

    double L = loss(pc, tsum, tsum_sq, 0, 1, 0, 1);

    cout << L << "\n";

    return result;
}
