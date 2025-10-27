#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <cassert>
#include <omp.h>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;

const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);

const uint64_t NX = 66; // 64 interior points + 2 boundary points
const uint64_t NY = 66;
const uint64_t NZ = 66;
const uint64_t TOTAL_SIZE = NX * NY * NZ;

const static double EPSILON = std::numeric_limits<double>::epsilon();

// Original scalar implementation of the 3D 7-point stencil
void stencil_3d_7pt(const double *curr, double *next)
{
  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 1; j < NY - 1; ++j)
    {
      for (int k = 1; k < NZ - 1; ++k)
      {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Precomputation
void stencil_3d_7pt_precompute(const double *curr, double *next)
{
  const uint64_t dim = NY * NZ;
  for (int i = 1; i < NX - 1; ++i)
  {
    const uint64_t tip1 = (i + 1) * dim;
    const uint64_t tim1 = (i - 1) * dim;
    const uint64_t tie1 = i * dim;
    for (int j = 1; j < NY - 1; ++j)
    {
      const uint64_t t1 = tip1 + j * NZ;
      const uint64_t t2 = tim1 + j * NZ;
      const uint64_t t3 = tie1 + (j + 1) * NZ;
      const uint64_t t4 = tie1 + (j - 1) * NZ;
      const uint64_t t5 = tie1 + j * NZ;
      for (int k = 1; k < (int)NZ - 1; ++k)
      {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[t1 + k];
        neighbors_sum += curr[t2 + k];
        neighbors_sum += curr[t3 + k];
        neighbors_sum += curr[t4 + k];
        neighbors_sum += curr[t5 + (k + 1)];
        neighbors_sum += curr[t5 + (k - 1)];
        next[t5 + k] = W_OWN * curr[t5 + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Tiled version of the 3D 7-point stencil
void stencil_3d_7pt_tiled(const double *curr, double *next)
{
  const int BLOCK_SIZE = 64;

  for (int ii = 1; ii < NX - 1; ii += BLOCK_SIZE)
  {
    for (int jj = 1; jj < NY - 1; jj += BLOCK_SIZE)
    {
      for (int kk = 1; kk < NZ - 1; kk += BLOCK_SIZE)
      {
        for (int i = ii; i < ii + BLOCK_SIZE && i < NX - 1; ++i)
        {
          for (int j = jj; j < jj + BLOCK_SIZE && j < NY - 1; ++j)
          {
            for (int k = kk; k < kk + BLOCK_SIZE && k < NZ - 1; ++k)
            {
              double neighbors_sum = 0.0;
              neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
              neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
              neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
              neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
              neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

              next[i * NY * NZ + j * NZ + k] =
                  W_OWN * curr[i * NY * NZ + j * NZ + k] +
                  W_NEIGHBORS * neighbors_sum;
            }
          }
        }
      }
    }
  }
}

// Unrolled version of the 3D 7-point stencil
void stencil_3d_7pt_unrolled(const double *curr, double *next)
{
  const int UNROLL_FACTOR = 4;

  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 1; j < NY - 1; ++j)
    {
      for (int k = 1; k <= (NZ - UNROLL_FACTOR - 1); k += UNROLL_FACTOR)
      {
        double neighbors_sum_0 = 0.0;
        neighbors_sum_0 += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum_0 += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum_0 += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum_0 += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum_0;

        double neighbors_sum_1 = 0.0;
        neighbors_sum_1 += curr[(i + 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum_1 += curr[(i - 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum_1 += curr[i * NY * NZ + (j + 1) * NZ + (k + 1)];
        neighbors_sum_1 += curr[i * NY * NZ + (j - 1) * NZ + (k + 1)];
        neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k + 2)];
        neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k)];

        next[i * NY * NZ + j * NZ + (k + 1)] =
            W_OWN * curr[i * NY * NZ + j * NZ + (k + 1)] +
            W_NEIGHBORS * neighbors_sum_1;

        double neighbors_sum_2 = 0.0;
        neighbors_sum_2 += curr[(i + 1) * NY * NZ + j * NZ + (k + 2)];
        neighbors_sum_2 += curr[(i - 1) * NY * NZ + j * NZ + (k + 2)];
        neighbors_sum_2 += curr[i * NY * NZ + (j + 1) * NZ + (k + 2)];
        neighbors_sum_2 += curr[i * NY * NZ + (j - 1) * NZ + (k + 2)];
        neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 3)];
        neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 1)];

        next[i * NY * NZ + j * NZ + (k + 2)] =
            W_OWN * curr[i * NY * NZ + j * NZ + (k + 2)] +
            W_NEIGHBORS * neighbors_sum_2;

        double neighbors_sum_3 = 0.0;
        neighbors_sum_3 += curr[(i + 1) * NY * NZ + j * NZ + (k + 3)];
        neighbors_sum_3 += curr[(i - 1) * NY * NZ + j * NZ + (k + 3)];
        neighbors_sum_3 += curr[i * NY * NZ + (j + 1) * NZ + (k + 3)];
        neighbors_sum_3 += curr[i * NY * NZ + (j - 1) * NZ + (k + 3)];
        neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 4)];
        neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 2)];

        next[i * NY * NZ + j * NZ + (k + 3)] =
            W_OWN * curr[i * NY * NZ + j * NZ + (k + 3)] +
            W_NEIGHBORS * neighbors_sum_3;
      }
    }
  }
}

// OpenMP parallelized version of the 3D 7-point stencil
void stencil_3d_7pt_omp(const double *curr, double *next)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 1; j < NY - 1; ++j)
    {
      for (int k = 1; k < NZ - 1; ++k)
      {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Tiling + Unrolled version of the 3D 7-point stencil
void stencil_3d_7pt_tiled_unrolled(const double *curr, double *next)
{
  const int BLOCK_SIZE = 16;
  const int UNROLL_FACTOR = 4;

  for (int ii = 1; ii < NX - 1; ii += BLOCK_SIZE)
  {
    for (int jj = 1; jj < NY - 1; jj += BLOCK_SIZE)
    {
      for (int kk = 1; kk < NZ - 1; kk += BLOCK_SIZE)
      {
        for (int i = ii; i < ii + BLOCK_SIZE && i < NX - 1; ++i)
        {
          for (int j = jj; j < jj + BLOCK_SIZE && j < NY - 1; ++j)
          {
            for (int k = kk; k < kk + BLOCK_SIZE - (UNROLL_FACTOR - 1) && k < NZ - 1; k += UNROLL_FACTOR)
            {
              double neighbors_sum_0 = 0.0;
              neighbors_sum_0 += curr[(i + 1) * NY * NZ + j * NZ + k];
              neighbors_sum_0 += curr[(i - 1) * NY * NZ + j * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + (j + 1) * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + (j - 1) * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k - 1)];

              next[i * NY * NZ + j * NZ + k] =
                  W_OWN * curr[i * NY * NZ + j * NZ + k] +
                  W_NEIGHBORS * neighbors_sum_0;

              double neighbors_sum_1 = 0.0;
              neighbors_sum_1 += curr[(i + 1) * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_1 += curr[(i - 1) * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + (j + 1) * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + (j - 1) * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k)];

              next[i * NY * NZ + j * NZ + (k + 1)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 1)] +
                  W_NEIGHBORS * neighbors_sum_1;

              double neighbors_sum_2 = 0.0;
              neighbors_sum_2 += curr[(i + 1) * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_2 += curr[(i - 1) * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + (j + 1) * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + (j - 1) * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 1)];

              next[i * NY * NZ + j * NZ + (k + 2)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 2)] +
                  W_NEIGHBORS * neighbors_sum_2;

              double neighbors_sum_3 = 0.0;
              neighbors_sum_3 += curr[(i + 1) * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_3 += curr[(i - 1) * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + (j + 1) * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + (j - 1) * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 4)];
              neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 2)];

              next[i * NY * NZ + j * NZ + (k + 3)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 3)] +
                  W_NEIGHBORS * neighbors_sum_3;
            }
          }
        }
      }
    }
  }
}

// Precomputation + Tiled + Unrolled version of the 3D 7-point stencil
void stencil_3d_7pt_precompute_tiled_unrolled(const double *curr, double *next)
{
  const int BLOCK_SIZE = 16;
  const int UNROLL_FACTOR = 4;
  const uint64_t dim = NY * NZ;

  for (int ii = 1; ii < NX - 1; ii += BLOCK_SIZE)
  {
    for (int jj = 1; jj < NY - 1; jj += BLOCK_SIZE)
    {
      for (int kk = 1; kk < NZ - 1; kk += BLOCK_SIZE)
      {
        for (int i = ii; i < ii + BLOCK_SIZE && i < NX - 1; ++i)
        {
          const uint64_t tip1 = (i + 1) * dim;
          const uint64_t tim1 = (i - 1) * dim;
          const uint64_t tie1 = i * dim;

          for (int j = jj; j < jj + BLOCK_SIZE && j < NY - 1; ++j)
          {
            const uint64_t t1 = tip1 + j * NZ;
            const uint64_t t2 = tim1 + j * NZ;
            const uint64_t t3 = tie1 + (j + 1) * NZ;
            const uint64_t t4 = tie1 + (j - 1) * NZ;
            const uint64_t t5 = tie1 + j * NZ;

            for (int k = kk; k < kk + BLOCK_SIZE - (UNROLL_FACTOR - 1) && k < NZ - 1; k += UNROLL_FACTOR)
            {
              double neighbors_sum_0 = curr[t1 + k] + curr[t2 + k] +
                                       curr[t3 + k] + curr[t4 + k] +
                                       curr[t5 + (k + 1)] + curr[t5 + (k - 1)];
              next[t5 + k] = W_OWN * curr[t5 + k] + W_NEIGHBORS * neighbors_sum_0;

              double neighbors_sum_1 = curr[t1 + (k + 1)] + curr[t2 + (k + 1)] +
                                       curr[t3 + (k + 1)] + curr[t4 + (k + 1)] +
                                       curr[t5 + (k + 2)] + curr[t5 + k];
              next[t5 + (k + 1)] = W_OWN * curr[t5 + (k + 1)] + W_NEIGHBORS * neighbors_sum_1;

              double neighbors_sum_2 = curr[t1 + (k + 2)] + curr[t2 + (k + 2)] +
                                       curr[t3 + (k + 2)] + curr[t4 + (k + 2)] +
                                       curr[t5 + (k + 3)] + curr[t5 + (k + 1)];
              next[t5 + (k + 2)] = W_OWN * curr[t5 + (k + 2)] + W_NEIGHBORS * neighbors_sum_2;

              double neighbors_sum_3 = curr[t1 + (k + 3)] + curr[t2 + (k + 3)] +
                                       curr[t3 + (k + 3)] + curr[t4 + (k + 3)] +
                                       curr[t5 + (k + 4)] + curr[t5 + (k + 2)];
              next[t5 + (k + 3)] = W_OWN * curr[t5 + (k + 3)] + W_NEIGHBORS * neighbors_sum_3;
            }
          }
        }
      }
    }
  }
}

// OpenMP parallelized + Tiled + Unrolled version of the 3D 7-point stencil
void stencil_3d_7pt_omp_tiled_unrolled(const double *curr, double *next)
{
  const int BLOCK_SIZE = 16;
  const int UNROLL_FACTOR = 4;

#pragma omp parallel for collapse(2)
  for (int ii = 1; ii < NX - 1; ii += BLOCK_SIZE)
  {
    for (int jj = 1; jj < NY - 1; jj += BLOCK_SIZE)
    {
      for (int kk = 1; kk < NZ - 1; kk += BLOCK_SIZE)
      {
        for (int i = ii; i < ii + BLOCK_SIZE && i < NX - 1; ++i)
        {
          for (int j = jj; j < jj + BLOCK_SIZE && j < NY - 1; ++j)
          {
            for (int k = kk; k < kk + BLOCK_SIZE - (UNROLL_FACTOR - 1) && k < NZ - 1; k += UNROLL_FACTOR)
            {
              double neighbors_sum_0 = 0.0;
              neighbors_sum_0 += curr[(i + 1) * NY * NZ + j * NZ + k];
              neighbors_sum_0 += curr[(i - 1) * NY * NZ + j * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + (j + 1) * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + (j - 1) * NZ + k];
              neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_0 += curr[i * NY * NZ + j * NZ + (k - 1)];

              next[i * NY * NZ + j * NZ + k] =
                  W_OWN * curr[i * NY * NZ + j * NZ + k] +
                  W_NEIGHBORS * neighbors_sum_0;

              double neighbors_sum_1 = 0.0;
              neighbors_sum_1 += curr[(i + 1) * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_1 += curr[(i - 1) * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + (j + 1) * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + (j - 1) * NZ + (k + 1)];
              neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_1 += curr[i * NY * NZ + j * NZ + (k)];

              next[i * NY * NZ + j * NZ + (k + 1)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 1)] +
                  W_NEIGHBORS * neighbors_sum_1;

              double neighbors_sum_2 = 0.0;
              neighbors_sum_2 += curr[(i + 1) * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_2 += curr[(i - 1) * NY * NZ + j * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + (j + 1) * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + (j - 1) * NZ + (k + 2)];
              neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_2 += curr[i * NY * NZ + j * NZ + (k + 1)];

              next[i * NY * NZ + j * NZ + (k + 2)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 2)] +
                  W_NEIGHBORS * neighbors_sum_2;

              double neighbors_sum_3 = 0.0;
              neighbors_sum_3 += curr[(i + 1) * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_3 += curr[(i - 1) * NY * NZ + j * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + (j + 1) * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + (j - 1) * NZ + (k + 3)];
              neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 4)];
              neighbors_sum_3 += curr[i * NY * NZ + j * NZ + (k + 2)];

              next[i * NY * NZ + j * NZ + (k + 3)] =
                  W_OWN * curr[i * NY * NZ + j * NZ + (k + 3)] +
                  W_NEIGHBORS * neighbors_sum_3;
            }
          }
        }
      }
    }
  }
}

// OpenMP Parallelize + Precomputation + Tiled + Unrolled version of the 3D 7-point stencil
void stencil_3d_7pt_omp_precompute_tiled_unrolled(const double *curr, double *next)
{
  const int BLOCK_SIZE = 16;
  const int UNROLL_FACTOR = 4;
  const uint64_t dim = NY * NZ;

#pragma omp parallel for collapse(3)
  for (int ii = 1; ii < NX - 1; ii += BLOCK_SIZE)
  {
    for (int jj = 1; jj < NY - 1; jj += BLOCK_SIZE)
    {
      for (int kk = 1; kk < NZ - 1; kk += BLOCK_SIZE)
      {
        for (int i = ii; i < ii + BLOCK_SIZE && i < NX - 1; ++i)
        {
          const uint64_t tip1 = (i + 1) * dim;
          const uint64_t tim1 = (i - 1) * dim;
          const uint64_t tie1 = i * dim;

          for (int j = jj; j < jj + BLOCK_SIZE && j < NY - 1; ++j)
          {
            const uint64_t t1 = tip1 + j * NZ;
            const uint64_t t2 = tim1 + j * NZ;
            const uint64_t t3 = tie1 + (j + 1) * NZ;
            const uint64_t t4 = tie1 + (j - 1) * NZ;
            const uint64_t t5 = tie1 + j * NZ;

            for (int k = kk; k < kk + BLOCK_SIZE - (UNROLL_FACTOR - 1) && k < NZ - 1; k += UNROLL_FACTOR)
            {
              double neighbors_sum_0 = curr[t1 + k] + curr[t2 + k] +
                                       curr[t3 + k] + curr[t4 + k] +
                                       curr[t5 + (k + 1)] + curr[t5 + (k - 1)];
              next[t5 + k] = W_OWN * curr[t5 + k] + W_NEIGHBORS * neighbors_sum_0;

              double neighbors_sum_1 = curr[t1 + (k + 1)] + curr[t2 + (k + 1)] +
                                       curr[t3 + (k + 1)] + curr[t4 + (k + 1)] +
                                       curr[t5 + (k + 2)] + curr[t5 + k];
              next[t5 + (k + 1)] = W_OWN * curr[t5 + (k + 1)] + W_NEIGHBORS * neighbors_sum_1;

              double neighbors_sum_2 = curr[t1 + (k + 2)] + curr[t2 + (k + 2)] +
                                       curr[t3 + (k + 2)] + curr[t4 + (k + 2)] +
                                       curr[t5 + (k + 3)] + curr[t5 + (k + 1)];
              next[t5 + (k + 2)] = W_OWN * curr[t5 + (k + 2)] + W_NEIGHBORS * neighbors_sum_2;

              double neighbors_sum_3 = curr[t1 + (k + 3)] + curr[t2 + (k + 3)] +
                                       curr[t3 + (k + 3)] + curr[t4 + (k + 3)] +
                                       curr[t5 + (k + 4)] + curr[t5 + (k + 2)];
              next[t5 + (k + 3)] = W_OWN * curr[t5 + (k + 3)] + W_NEIGHBORS * neighbors_sum_3;
            }
          }
        }
      }
    }
  }
}

// Scalar Vectorization
void stencil_3d_7pt_scalar_vectorize(const double *curr, double *next)
{
  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 1; j < NY - 1; ++j)
    {
#pragma omp simd
      for (int k = 1; k < NZ - 1; ++k)
      {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

int main()
{
  auto *grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto *grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double *current_grid = grid1;
  double *next_grid = grid2;

  // Original scalar implementation of the 3D 7-point stencil
  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Scalar kernel time: " << duration << " ms" << endl;

  double true_final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << true_final << "\n";
  double true_total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    true_total_sum += current_grid[i];
  }
  cout << "Total sum : " << true_total_sum << "\n";

  cout << "\n----------------------\n\n";

  // Precomputation version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_precompute(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Precompute kernel time: " << duration << " ms" << endl;

  double final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  double total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // Tiled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_tiled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Tiled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // Unrolled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_unrolled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Unrolled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // OpenMP parallelized version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_omp(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "OpenMP kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // Tiling + Unrolled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_tiled_unrolled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Tiling + Unrolled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // Precompute + Tiling + Unrolled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_precompute_tiled_unrolled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Precompute + Tiling + Unrolled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // OpenMP parallelized + Tiled + Unrolled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_omp_tiled_unrolled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "OpenMP + Tiled + Unrolled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // OpenMP parallelized + Precompute + Tiled + Unrolled version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_omp_precompute_tiled_unrolled(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "OpenMP + Precompute + Tiled + Unrolled kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < EPSILON && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  // Scalar + Vectorize version of the 3D 7-point stencil
  grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++)
  {
    stencil_3d_7pt_scalar_vectorize(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Scalar + Vectorize kernel time: " << duration << " ms" << endl;

  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++)
  {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";
  assert(std::fabs(total_sum - true_total_sum) < 1e-9 && "Incorrect total sum");
  assert(std::fabs(final - true_final) < EPSILON && "Incorrect final value");
  cout << "\n----------------------\n\n";

  delete[] grid1;
  delete[] grid2;

  return EXIT_SUCCESS;
}
