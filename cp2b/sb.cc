#include <iostream>
#include "cp.h"
#include <iostream>
using namespace std;
#include <vector>
#include <cmath>
#include <numeric>

int main() {
    int ny = 2;
    int nx = 2;

    // float data[nx*ny] = {-1.0,1.0,1.0,-1.0,-1.0,1.0};
    float data[nx*ny] = {-1.0,1.0,-1.0,1.0};
    
    // Simulated image: 4x4 pixels, 3 channels per pixel (RGB)
    // float data[4 * 4] = {
    //     5, 32, 4, 4,
    //     5, 3, 365, 4,
    //     354,11,23,74,
    //     88, 132, 45, 65 
    // };
    float result[nx*ny];

    correlate(ny,nx,data,result);

    // for (int i = 0; i<nx*ny; i++) {
    //     cout << data[i];
    // }
    // cout << "\n";

    // for (int y=0; y<ny; y++) {
    //     for (int x=0; x<nx; x++) {
    //         cout << data[x + y*nx] << " ";
    //     }
    //     cout << "\n";
    // }
    cout << "\ncor\n";
    for (int y=0; y<ny; y++) {
        for (int x=0; x<ny; x++) {
            cout << result[x + y*ny] << " ";
        }
        cout << "\n";
    }

    return 0;
}