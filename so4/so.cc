using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <omp.h>

typedef unsigned long long data_t;

void mergesort(data_t *vec, size_t l, size_t r, size_t th) {
    if (l >= r) return;

    if (r - l < 8192) {
        sort(vec + l, vec + r + 1);
        return;
    }

    size_t m = l + (r - l)/2;

    if (r - l > th) {
        #pragma omp task
        mergesort(vec, l, m, th);
    
        #pragma omp task
        mergesort(vec, m + 1, r, th);
    
        #pragma omp taskwait
    } else {
        mergesort(vec, l, m, th);
        mergesort(vec, m + 1, r, th);
    }
    inplace_merge(vec+l, vec+m+1, vec+r+1);
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    // std::sort(data, data + n);
    if (n <= 1) return;

    int threads = omp_get_max_threads();
    size_t th = max<size_t>(
        50000,
        size_t(n) / (16 * threads)
    );

    #pragma omp parallel
    {
        #pragma omp single
        mergesort(data, 0, n-1, th);
    }
}
