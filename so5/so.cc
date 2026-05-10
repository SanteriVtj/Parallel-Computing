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
#include <random>

typedef unsigned long long data_t;

static thread_local mt19937_64 rng(random_device{}());


void qsort(data_t *vec, size_t l, size_t r, size_t th) {
    if (l >= r) return;
    if (r - l < 2048) {
        sort(vec + l, vec + r + 1);
        return;
    }

    // data_t pivot = vec[l + (r - l)/2];
    uniform_int_distribution<size_t> dist(l, r);
    data_t pivot = vec[dist(rng)];

    auto mid1 = partition(vec + l, vec + r + 1, [pivot](data_t x) {return x < pivot;});
    auto mid2 = partition(mid1, vec + r + 1, [pivot](data_t x) {return x == pivot;});

    size_t m1 = mid1 - vec;
    size_t m2 = mid2 - vec;

    if (r - l > th) {
        #pragma omp task
        {if (m1 > l) qsort(vec, l, m1 - 1, th);}

        #pragma omp task
        {if (m2 <= r) qsort(vec, m2, r, th);}
        
        #pragma omp taskwait
    } else {
        if (m1 > l) qsort(vec, l, m1 - 1, th);
        if (m2 <= r) qsort(vec, m2, r, th);
    }
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    // std::sort(data, data + n);
    if (n<=1) return;
    int threads = omp_get_max_threads();
    size_t th = max<size_t>(
        50000,
        size_t(n) / (16 * threads)
    );
    #pragma omp parallel
    {
        #pragma omp single
        qsort(data, 0, n - 1, th);
    }
}
