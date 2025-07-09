using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

constexpr int unroll = 8;
constexpr int blocks = 7;
// typedef double float8_t __attribute__ ((vector_size (unroll * sizeof(double))));
typedef float float8_t __attribute__ ((vector_size (unroll * sizeof(float))));
constexpr float8_t float8_t_init {0,0,0,0};


static inline double sumd4t(float8_t x) {
    double sum = 0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i<unroll; i++) {
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float *data, float *result) {
    int na = (nx + unroll - 1)/unroll;
    int nc = (ny + blocks - 1) / blocks;
    int ncd = nc*blocks;
    vector<float8_t> X(ncd * na, float8_t_init);
    
    #pragma omp parallel for
    for (int y = 0; y<ny; y++) {
        for (int x = 0; x<na; x++) {
            for (int ka = 0; (ka<unroll) && (x*unroll+ka<nx); ka++) {
                X[y*na + x][ka] = data[y*nx + x*unroll + ka];
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (int y=0; y<ny; y++) {
        float avg, sq, norm;
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

    #pragma omp parallel for schedule(dynamic)
    for (int ib=0; ib<nc; ib++) {
        for (int jb=ib; jb<nc; jb++) {
            // Initialize the temp-array to zeros
            float8_t temp_array[blocks][blocks];
            for (int i = 0; i<blocks; i++) {
                for(int j = 0; j<blocks; j++) {
                    temp_array[i][j] = float8_t_init;
                }
            }

            
            for (int ka = 0; ka<na; ka++) {
                float8_t x0 = X[(ib*blocks + 0)*na + ka];
                float8_t x1 = X[(ib*blocks + 1)*na + ka];
                float8_t x2 = X[(ib*blocks + 2)*na + ka];
                float8_t x3 = X[(ib*blocks + 3)*na + ka];
                float8_t x4 = X[(ib*blocks + 4)*na + ka];
                float8_t x5 = X[(ib*blocks + 5)*na + ka];
                float8_t x6 = X[(ib*blocks + 6)*na + ka];

                float8_t y0 = X[(jb*blocks + 0)*na + ka];
                float8_t y1 = X[(jb*blocks + 1)*na + ka];
                float8_t y2 = X[(jb*blocks + 2)*na + ka];
                float8_t y3 = X[(jb*blocks + 3)*na + ka];
                float8_t y4 = X[(jb*blocks + 4)*na + ka];
                float8_t y5 = X[(jb*blocks + 5)*na + ka];
                float8_t y6 = X[(jb*blocks + 6)*na + ka];
                
                temp_array[0][0] += x0*y0;
                temp_array[0][1] += x0*y1;
                temp_array[0][2] += x0*y2;
                temp_array[0][3] += x0*y3;
                temp_array[0][4] += x0*y4;
                temp_array[0][5] += x0*y5;
                temp_array[0][6] += x0*y6;

                temp_array[1][0] += x1*y0;
                temp_array[1][1] += x1*y1;
                temp_array[1][2] += x1*y2;
                temp_array[1][3] += x1*y3;
                temp_array[1][4] += x1*y4;
                temp_array[1][5] += x1*y5;
                temp_array[1][6] += x1*y6;
                
                temp_array[2][0] += x2*y0;
                temp_array[2][1] += x2*y1;
                temp_array[2][2] += x2*y2;
                temp_array[2][3] += x2*y3;
                temp_array[2][4] += x2*y4;
                temp_array[2][5] += x2*y5;
                temp_array[2][6] += x2*y6;
                
                temp_array[3][0] += x3*y0;
                temp_array[3][1] += x3*y1;
                temp_array[3][2] += x3*y2;
                temp_array[3][3] += x3*y3;
                temp_array[3][4] += x3*y4;
                temp_array[3][5] += x3*y5;
                temp_array[3][6] += x3*y6;
                
                temp_array[4][0] += x4*y0;
                temp_array[4][1] += x4*y1;
                temp_array[4][2] += x4*y2;
                temp_array[4][3] += x4*y3;
                temp_array[4][4] += x4*y4;
                temp_array[4][5] += x4*y5;
                temp_array[4][6] += x4*y6;
                
                temp_array[5][0] += x5*y0;
                temp_array[5][1] += x5*y1;
                temp_array[5][2] += x5*y2;
                temp_array[5][3] += x5*y3;
                temp_array[5][4] += x5*y4;
                temp_array[5][5] += x5*y5;
                temp_array[5][6] += x5*y6;
                
                temp_array[6][0] += x6*y0;
                temp_array[6][1] += x6*y1;
                temp_array[6][2] += x6*y2;
                temp_array[6][3] += x6*y3;
                temp_array[6][4] += x6*y4;
                temp_array[6][5] += x6*y5;
                temp_array[6][6] += x6*y6;
            }

            //     for (int x = 0; x<blocks; x++) {
            //         int ix = ib*blocks + x;
            //         if (ix >= ny) continue;
            //         for (int y = 0; y<blocks; y++) {
            //             int iy = jb*blocks + y;
            //             if (iy >=  ny) continue;
            //             float8_t x0 = X[ix*na + ka];
            //             float8_t y0 = X[iy*na + ka];

            //             temp_array[x][y] += y0*x0;
            //         }
            //     }
            // }
            for (int x = 0; (x<blocks) & (x<ny); x++) {
                int ix = ib*blocks + x;
                if (ix >= ny) continue;
                for (int y = 0; (y<blocks) & (y<ny); y++) {
                    int iy = jb*blocks + y;
                    if (iy >=  ny) continue;
                    result[ix*ny + iy] = float(sumd4t(temp_array[x][y]));
                }
            }
        }
    }
}
