#ifndef __GEMM_H__
#define __GEMM_H__

#include "mat.h"

#define KERNEL_M 4
#define KERNEL_N 24

void sgemm_only_4x24__avx2(int32_t k, const float *a, int32_t a_off,
                           const float *b, int32_t b_off, float *c,
                           int32_t c_off, int32_t ldc);

void sgemm_naive(int32_t m, int32_t k, int32_t n, const Mat a, const Mat b,
                 Mat c);

void sgemm_avx2(int32_t m, int32_t k, int32_t n, const Mat a, const Mat b,
                Mat c, float *a_panel_buf, float *b_panel_buf,
                float *c_flat_buf);

#endif // !defined __GEMM_H__
