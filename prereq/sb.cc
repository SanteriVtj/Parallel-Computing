#include <iostream>
#include "calculate.h"
using namespace std;

int main() {
    int ny = 4;
    int nx = 4;
    
    // Simulated image: 4x4 pixels, 3 channels per pixel (RGB)
    float data[4 * 4 * 3] = {
        // row 0 (4 pixels × 3 components each)
        1,2,3,  4,5,6,  7,8,9,  10,11,12,
        // row 1
        13,14,15, 16,17,18, 19,20,21, 22,23,24,
        // row 2
        25,26,27, 28,29,30, 31,32,33, 34,35,36,
        // row 3
        37,38,39, 40,41,42, 43,44,45, 46,47,48
    };

    int y0 = 1, x0 = 1, y1 = 3, x1 = 3;

    Result result = calculate(ny, nx, data, y0, x0, y1, x1);

    std::cout << "Average color: R=" << result.avg[0]
              << " G=" << result.avg[1]
              << " B=" << result.avg[2] << std::endl;
    return 0;
}