#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mat.h"

/* SGEMM, naive implementation for reference.
 * a has m rows, k columns.
 * b has k rows, n columns.
 * c has m rows, n columns.
 */
void sgemm_naive(int32_t m, int32_t k, int32_t n, const Mat a, const Mat b,
                 Mat c) {
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        c[mm][nn] += a[mm][kk] * b[kk][nn];
      }
    }
  }
}
