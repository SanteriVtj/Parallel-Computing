using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
constexpr int unroll = 4;

void correlate(int ny, int nx, const float *data, float *result) {
    vector<double> X(nx*ny,0);
    
    int row;
    double avg, sq, v, norm;
    for (int y=0; y<ny; y++) {
        avg = 0;
        sq = 0;
        row = y*nx;
        
        for (int x=0; x<nx; x++) {
            v = double(data[x + row]);
            avg += v;
            X[x + row] = v;
        }
        
        avg /= nx;
        
        for (int x=0; x<nx; x++) {
            norm = X[x + row]-avg;
            sq += norm*norm;
            X[x + row] = norm;
        }
        
        sq = sqrt(sq);
        
        for (int x=0; x<nx; x++) {
            X[x + row] /= sq;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i<ny; i++) {
        for (int j = i+1; j<ny; j++) {
            int row1 = i*nx;
            int row2 = j*nx;
            double sum = 0;
            vector<double> sums(unroll, 0);
            for (int s = 0; s<nx; s += unroll) {
                for (int ur = 0; (ur<unroll) && ((s + ur) < nx); ur++) {
                    sums[ur] = X[s + ur + row1] * X[s + ur + row2];
                }
                sum += reduce(sums.begin(), sums.end());
                fill(sums.begin(),sums.end(),0);
            }
            result[j + i*ny] = float(sum);
        }
    }

    for (int i = 0; i<ny; i++)  {
        result[i + i*ny] = 1.f;
    }
}
