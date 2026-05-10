using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <functional>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

static double window_median(const float *in, int x, int y, int nx, int ny, int hx, int hy) {
  int start_x = max(0, x-hx);
  int start_y = max(0, y-hy);
  int end_x = min(nx, x + hx + 1);
  int end_y = min(ny, y + hy + 1);

  int window_size = (end_x-start_x)*(end_y-start_y);
  vector<double> flat_window(window_size);

  int flat_idx = 0;
  for (int i = start_y; i < end_y; i++) {
    for(int j = start_x; j < end_x; j++) {
      flat_window[flat_idx] = in[j + i*nx];
      flat_idx++;
    }
  }
  auto m = flat_window.begin() + flat_window.size() / 2;
  nth_element(flat_window.begin(), m, flat_window.end());
  double high = *m;
  if (flat_window.size() % 2 == 0) {
    double low = *max_element(flat_window.begin(), m);
    return 0.5*(low +  high);
  }

  return flat_window[flat_window.size()/2];
}

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      out[j + i*nx] = window_median(in, j, i, nx, ny, hx, hy);
    }
  }
}
