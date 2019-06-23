#include <memory.h>
#include <stdint.h>

#include "mat.h"

Mat alloc_mat(int32_t rows, int32_t cols) {
  Mat ret = malloc(sizeof(float *) * rows);
  for (int32_t r = 0; r < rows; ++r) {
    ret[r] = malloc(sizeof(float) * cols);
  }
  return ret;
}

Mat alloc_rand_mat(int32_t rows, int32_t cols) {
  Mat ret = alloc_mat(rows, cols);
  gen_rand_mat(rows, cols, ret);
  return ret;
}

void free_mat(Mat mat, int32_t rows) {
  for (int32_t r = 0; r < rows; ++r) {
    free(mat[r]);
  }
  free(mat);
}

void gen_rand_mat(int32_t rows, int32_t cols, Mat mat) {
  for (int32_t r = 0; r < rows; ++r) {
    for (int32_t c = 0; c < cols; ++c) {
      mat[r][c] = NEXT_FLOAT;
    }
  }
}
