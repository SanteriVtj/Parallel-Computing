using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>

constexpr int unroll = 8;
constexpr int batch = 8;
constexpr int tile = 2*64;
typedef float double4_t __attribute__ ((vector_size (unroll * sizeof(float))));

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    int nx_padded = (nx + unroll - 1)/unroll;
    int ny_padded = (ny + batch - 1) / batch * batch;

    
    double4_t zero = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    vector<double4_t> X(nx_padded*ny_padded, zero);
    
    #pragma omp parallel 
    {
        #pragma omp for schedule(static)
        for (int y=0; y<ny; y++) {
            int row;
            float avg, sq, inv_sq, norm;
            avg = 0;
            sq = 0;
            row = y*nx_padded;
            
            for (int x=0; x<nx; x++) avg += data[x + y*nx];
            avg /= nx;
            
            for (int x=0; x<nx_padded; x++) {
                for (int p = 0; p < unroll; p++) {
                    int idx = p + x*unroll;
                    norm = (idx < nx) ? data[idx + y*nx] - avg : 0.0;
                    X[x + row][p] = norm;
                    sq += norm*norm;
                }
            }
            
            inv_sq = 1.0/sqrtf(sq);
            
            for (int x=0; x<nx_padded; x++) X[x + row] *= inv_sq;
        }

        int zmax = _pdep_u32(ny_padded/batch, 0x55555555u) | _pdep_u32(ny_padded/batch, 0xAAAAAAAAu);

        #pragma omp for schedule(dynamic, batch)
        for (int z = 0; z < zmax; z++) {
            int ia = _pext_u32(z, 0x55555555u) * batch;
            int ja = _pext_u32(z, 0xAAAAAAAAu) * batch;

            if (ja < ia || ia >= ny || ja >= ny) continue;

            double4_t sums[batch][batch] = {zero, zero, zero, zero};

            for (int xa = 0; xa < nx_padded; xa += tile) {
                int xb = min(xa + tile, nx_padded);

                for (int xc = xa; xc < xb; xc++) {
                    double4_t a[batch];
                    for (int ib = 0; ib < batch; ib++) a[ib] = X[xc + (ia + ib)*nx_padded];
                    
                    double4_t b[batch];
                    for (int jb = 0; jb < batch; jb++) b[jb] = X[xc + (ja + jb)*nx_padded];
                    
                    for (int ib = 0; ib < batch; ib++) {
                        for (int jb = 0; jb < batch; jb++) {
                            sums[ib][jb] += a[ib]*b[jb];
                        }
                    }
                }
            }

            for (int ib = 0; ib < batch; ib++) {
                for (int jb = 0; jb < batch; jb++) {
                    if ((ia + ib < ny) && (ja + jb < ny) & (ja + jb > ia + ib)) {
                        float sum = 0;
                        for (int p = 0; p < unroll; p++) sum += sums[ib][jb][p];
                        result[ja + jb + (ia + ib)*ny] = sum;
                    }
                }
            }
        }

        #pragma omp for
        for (int i = 0; i<ny; i++) result[i + i*ny] = 1.f;
    }
}
