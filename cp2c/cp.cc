using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

constexpr int unroll = 4;
typedef double double4_t __attribute__ ((vector_size (unroll * sizeof(double))));

static inline double sumd4t(double4_t x) {
    double sum = 0;
    for (int i = 0; i<unroll; i++) {
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float *data, float *result) {
    int na = (nx + unroll - 1)/unroll;
    vector<double4_t> X(ny*na);
    
    double avg, sq, norm;

    for (int y = 0; y<ny; y++) {
        for (int x = 0; x<na; x++) {
            for (int ka = 0; (ka<unroll) && (x*unroll+ka<nx); ka++) {
                X[y*na + x][ka] = double(data[y*nx + x*unroll + ka]);
            }
        }
    }

    for (int y=0; y<ny; y++) {
        avg = 0;
        sq = 0;
        for (int x = 0; x<na; x++) {
            avg += sumd4t(X[y*na+x]);
        }
        
        avg /= nx;
        
        for (int x=0; x<na; x++) {
            for (int ka = 0; (ka<unroll)&&(x*unroll+ka<nx); ka++) {
                norm = X[x + y*na][ka]-avg;
                sq += norm*norm;
                X[x + y*na][ka] = norm;
            }
        }
        
        sq = sqrt(sq);
        
        for (int x=0; x<na; x++) {
            X[x + y*na] /= sq;
        }
    }

    // #pragma omp parallel for
    for (int i = 0; i<ny; i++) {
        for (int j = i+1; j<ny; j++) {
            double sum = 0;
            for (int s = 0; (s<na); s++) {
                double4_t sums = {0,0,0,0};
                sums = X[i*na + s] * X[j*na + s];
                sum += sumd4t(sums);
            }
            result[j + i*ny] = float(sum);
        }
    }

    for (int i = 0; i<ny; i++)  {
        result[i + i*ny] = 1.f;
    }
}
