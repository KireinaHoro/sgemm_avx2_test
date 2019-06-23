#ifndef __MAT_H__
#define __MAT_H__

#include <stdlib.h>
#include <time.h>

#define INIT_RAND srand((uint32_t)time(NULL))
#define NEXT_FLOAT ((float)(((double)rand() / (double)(RAND_MAX)) * 5.0))

typedef float **Mat;

void gen_rand_mat(int32_t rows, int32_t cols, Mat mat);

Mat alloc_mat(int32_t rows, int32_t cols);
Mat alloc_rand_mat(int32_t rows, int32_t cols);

void free_mat(Mat mat, int32_t rows);

#endif // !defined __MAT_H__
