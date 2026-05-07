#include <iostream>
#include "is.hpp"
#include <iostream>
using namespace std;
#include <vector>
#include <cmath>
#include <numeric>

int main() {
    float data[27];
    for (int i = 0; i < 27; i++) data[i] = 1.0f;
    Result r = segment(3, 3, data);
    // cout << "y0=" << r.y0 << " x0=" << r.x0
    //      << " y1=" << r.y1 << " x1=" << r.x1 << "\n";
    return 0;
}