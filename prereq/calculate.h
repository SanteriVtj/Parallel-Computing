// calculate.h
#ifndef CALCULATE_H
#define CALCULATE_H

struct Result {
    float avg[3];
};

Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1);

#endif // CALCULATE_H