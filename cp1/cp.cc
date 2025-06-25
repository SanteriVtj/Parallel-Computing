#include <iostream>
using namespace std;
#include <vector>
#include <cmath>

void correlate(int ny, int nx, const float *data, float *result) {
    vector<double> avg_x(ny);
    vector<double> sq_x(ny);
    
    for (int y=0; y<ny; y++) {
        for (int x=0; x<nx; x++) {
            avg_x[y] += static_cast<double>(data[x + y*nx]/nx);
        }
        for (int x=0; x<nx; x++) {
            sq_x[y] += static_cast<double>(pow(data[x + y*nx]-avg_x[y], 2));
        }
    }

    for (int y=0; y<ny; y++){
        for (int x=y+1; x<ny; x++) {
            double num = 0;
            for (int i=0; i<nx; i++) {
                num += (data[i + x*nx]-avg_x[x])*(data[i + y*nx]-avg_x[y]);
            }
            result[x + y*ny] = static_cast<float>(num/sqrt((sq_x[x]*sq_x[y])));
        }
    }
    for (int i=0; i<ny; i++) {
        result[i+i*ny] = 1.f;
    }
}