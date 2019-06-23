#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gemm.h"

/* SGEMM, AVX-2 implementation with handcrafted kernel for 4*24.
 * a has m rows, k columns.
 * b has k rows, n columns.
 * c has m rows, n columns.
 * a_panel_buf should have length of at least k * a_tiles * KERNEL_M.
 * b_panel_buf should have length of at least k * b_tiles * KERNEL_N.
 * c_flat_buf should have length of at least a_tiles * KERNEL_M * b_tiles *
 * KERNEL_N.
 */
void sgemm_avx2(int32_t m, int32_t k, int32_t n, const Mat a, const Mat b,
                Mat c, float *a_panel_buf, float *b_panel_buf,
                float *c_flat_buf) {
  int32_t a_tiles = m / KERNEL_M;
  int32_t a_left = m % KERNEL_M;
  int32_t b_tiles = n / KERNEL_N;
  int32_t b_left = n % KERNEL_N;

  if (a_left != 0) {
    ++a_tiles;
  }
  if (b_left != 0) {
    ++b_tiles;
  }

  for (int t = 0; t < a_tiles; ++t) {
    if (a_left != 0 && t == a_tiles - 1) {
      for (int kk = 0; kk < k; ++kk) {
        for (int l = 0; l < a_left; ++l) {
          a_panel_buf[(t * k + kk) * KERNEL_M + l] = a[t * KERNEL_M + l][kk];
        }
        for (int l = a_left; l < KERNEL_M; ++l) {
          a_panel_buf[(t * k + kk) * KERNEL_M + l] = 0;
        }
      }
    } else {
      for (int kk = 0; kk < k; ++kk) {
        for (int l = 0; l < KERNEL_M; ++l) {
          a_panel_buf[(t * k + kk) * KERNEL_M + l] = a[t * KERNEL_M + l][kk];
        }
      }
    }
  }

  /*
  // dump a and a_panel
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      printf("%f ", a[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < a_tiles * KERNEL_M * k; ++i) {
    if (i % 4 == 0) {
      printf(" | ");
    }
    printf("%f ", a_panel_buf[i]);
  }
  printf("\na and a_panel dumped\n");
  */

  for (int t = 0; t < b_tiles; ++t) {
    if (b_left != 0 && t == b_tiles - 1) {
      for (int kk = 0; kk < k; ++kk) {
        for (int l = 0; l < b_left; ++l) {
          b_panel_buf[(t * k + kk) * KERNEL_N + l] = b[kk][t * KERNEL_N + l];
        }
        for (int l = b_left; l < KERNEL_N; ++l) {
          b_panel_buf[(t * k + kk) * KERNEL_N + l] = 0;
        }
      }
    } else {
      for (int kk = 0; kk < k; ++kk) {
        for (int l = 0; l < KERNEL_N; ++l) {
          b_panel_buf[(t * k + kk) * KERNEL_N + l] = b[kk][t * KERNEL_N + l];
        }
      }
    }
  }

  /*
  // dump b and b_panel
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", b[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < b_tiles * KERNEL_N * k; ++i) {
    if (i % 24 == 0) {
      printf(" | ");
    }
    printf("%f ", b_panel_buf[i]);
  }
  printf("\nb and b_panel dumped\n");
  */

  if (a_tiles >= 8) {
    for (int mo = 0; mo < a_tiles / 8; ++mo) {
      for (int no = 0; no < b_tiles; ++no) {
        for (int mio = 0; mio < 8; ++mio) {
          sgemm_only_4x24__avx2(k, a_panel_buf, (mo * 8 + mio) * k * KERNEL_M,
                                b_panel_buf, no * k * KERNEL_N, c_flat_buf,
                                KERNEL_N *
                                    ((mo * 8 + mio) * KERNEL_M * b_tiles + no),
                                b_tiles * KERNEL_N);
        }
      }
    }
  } else {
    for (int m = 0; m < a_tiles; ++m) {
      for (int no = 0; no < b_tiles; ++no) {
        sgemm_only_4x24__avx2(k, a_panel_buf, m * k * KERNEL_M, b_panel_buf,
                              no * k * KERNEL_N, c_flat_buf,
                              KERNEL_N * (m * KERNEL_M * b_tiles + no),
                              b_tiles * KERNEL_N);
      }
    }
  }

  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      c[mm][nn] = c_flat_buf[mm * (b_tiles * KERNEL_N) + nn];
    }
  }

  /*
  // dump c and c_flat
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", c[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < b_tiles * KERNEL_N * a_tiles * KERNEL_M; ++i) {
    printf("%f ", c_flat_buf[i]);
  }
  printf("\nc and c_flat dumped\n");
  */
}
