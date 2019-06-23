#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemm.h"
#include "mat.h"

int32_t main(int32_t argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <M> <K> <N>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  const int M = atoi(argv[1]);
  const int K = atoi(argv[2]);
  const int N = atoi(argv[3]);

  printf("Initializing rng...\n");
  INIT_RAND;

  printf("Generating matrices...\n");
  Mat a = alloc_rand_mat(M, K);
  Mat b = alloc_rand_mat(K, N);
  Mat c = alloc_mat(M, N);
  Mat ref = alloc_mat(M, N);

  printf("Calculating buffer allocations for shape (%d, %d, %d)...\n", M, K, N);
  int32_t a_left = M % KERNEL_M;
  int32_t a_alloc;
  if (a_left == 0) {
    a_alloc = M;
  } else {
    a_alloc = (M / KERNEL_M + 1) * KERNEL_M;
  }

  int32_t b_left = N % KERNEL_N;
  int32_t b_alloc;
  if (b_left == 0) {
    b_alloc = N;
  } else {
    b_alloc = (N / KERNEL_N + 1) * KERNEL_N;
  }

  float *a_panel = malloc(sizeof(float) * K * a_alloc);
  float *b_panel = malloc(sizeof(float) * K * b_alloc);
  float *c_flat = malloc(sizeof(float) * a_alloc * b_alloc);
  printf("Allocation sizes: A = (%d, %d), B = (%d, %d)\n", a_alloc, K, K,
         b_alloc);

  // Timed test
  struct timespec start, end;

  printf("Performing naive SGEMM...");
  fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  sgemm_naive(M, K, N, a, b, ref);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 +
                      (end.tv_nsec - start.tv_nsec) / 1000;
  double delta_ms = (double)delta_us / 1000.0;

  printf("  ----- took %.3fms.\n", delta_ms);

  printf("Performing AVX-2 SGEMM...");
  fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  sgemm_avx2(M, K, N, a, b, c, a_panel, b_panel, c_flat);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  delta_us = (end.tv_sec - start.tv_sec) * 1000000 +
             (end.tv_nsec - start.tv_nsec) / 1000;
  delta_ms = (double)delta_us / 1000.0;

  printf("  ----- took %.3fms.\n", delta_ms);

  // Test for correctness
  bool matched = true;
  printf("Verifying results...\n");
  int32_t matched_count = 0;
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      if ((double)fabsf(c[i][j] - ref[i][j]) / ref[i][j] > 1e-6) {
        printf("mismatch: kernel = %.10f, ref = %.10f\n", c[i][j], ref[i][j]);
        matched = false;
      } else {
        matched_count++;
      }
      // assert(c[i][j] == ref[i][j]);
    }
  }

  if (matched) {
    printf("Test succeeded.\n");
  } else {
    printf("Test finished, but mismatch exists.  Mismatch/Total: %d/%d\n",
           M * N - matched_count, M * N);
  }

  printf("Freeing buffers...\n");
  free_mat(a, M);
  free_mat(b, K);
  free_mat(c, M);
  free_mat(ref, M);
  free(a_panel);
  free(b_panel);
  free(c_flat);

  exit(EXIT_SUCCESS);
}
