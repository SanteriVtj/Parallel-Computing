#include <iostream>
using namespace std;
#include <vector>
#include <cmath>
#include <numeric>
constexpr int unroll = 4;

constexpr float infty = std::numeric_limits<float>::infinity();
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    int nx_padded = (nx + unroll - 1)/unroll*unroll;
    
    vector<double> X(nx_padded*ny, 0.0);
    
    #pragma omp parallel for
    for (int y=0; y<ny; y++) {
        int row;
        double avg, sq, inv_sq, norm;
        avg = 0;
        sq = 0;
        row = y*nx_padded;
        
        for (int x=0; x<nx; x++) avg += data[x + y*nx];
        avg /= nx;
        
        for (int x=0; x<nx; x++) {
            double norm = data[x + y*nx] - avg;
            X[x + row] = norm;
            sq += norm*norm;
        }
        
        inv_sq = 1.0/sqrt(sq);
        
        for (int x=0; x<nx; x++) X[x + row] *= inv_sq;
    }

    #pragma omp parallel for
    for (int i = 0; i<ny; i++) {
        for (int j = i+1; j<ny; j++) {
            int row1, row2;
            double sum;

            row1 = i*nx_padded;
            row2 = j*nx_padded;

            double sums[unroll];
            for (int x = 0; x < unroll; x++) sums[x] = 0;
            
            for (int x = 0; x<nx_padded; x += unroll) {
                for (int ur = 0; ur < unroll; ur++) {
                    sums[ur] += X[x + ur + row1] * X[x + ur + row2];
                }
            }
            sum = 0;
            for (int x = 0; x < unroll; x++) sum += sums[x];
            
            result[j + i*ny] = float(sum);
        }
    }

    for (int i = 0; i<ny; i++) result[i + i*ny] = 1.f;
}
