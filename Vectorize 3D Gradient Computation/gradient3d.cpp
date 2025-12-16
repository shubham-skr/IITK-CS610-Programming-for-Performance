#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <assert.h>
#include <immintrin.h>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint32_t NX = 128;
const uint32_t NY = 128;
const uint32_t NZ = 128;
const uint64_t TOTAL_SIZE = (NX * NY * NZ);

const uint32_t N_ITERATIONS = 100;
const uint64_t INITIAL_VAL = 1000000;

void scalar_3d_gradient(const uint64_t *A, uint64_t *B)
{
  const uint64_t stride_i = (NY * NZ);
  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 0; j < NY; ++j)
    {
      for (int k = 0; k < NZ; ++k)
      {
        uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
        // A[i+1, j, k]
        int A_right = A[base_idx + stride_i];
        // A[i-1, j, k]
        int A_left = A[base_idx - stride_i];
        B[base_idx] = A_right - A_left;
      }
    }
  }
}

void sse4_3d_gradient(const uint64_t *A, uint64_t *B)
{
  const uint64_t stride_i = (NY * NZ);
  const int sse4_stride = 2; // 2 x 64-bit integers per __m128i
  for (int i = 1; i < NX - 1; ++i)
    for (int j = 0; j < NY; ++j)
      for (int k = 0; k < NZ; k += sse4_stride)
      {
        uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
        __m128i a_right = _mm_load_si128((__m128i *)&A[base_idx + stride_i]);
        __m128i a_left = _mm_load_si128((__m128i *)&A[base_idx - stride_i]);
        __m128i result = _mm_sub_epi64(a_right, a_left);
        _mm_store_si128((__m128i *)&B[base_idx], result);
      }
}

void avx2_3d_gradient(const uint64_t *A, uint64_t *B)
{
  const uint64_t stride_i = NY * NZ;
  const int avx2_stride = 4; // 4 x 64-bit integers per __m256i

  for (int i = 1; i < NX - 1; ++i)
  {
    for (int j = 0; j < NY; ++j)
    {
      for (int k = 0; k < NZ; k += avx2_stride)
      {
        uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
        __m256i a_right = _mm256_loadu_si256((__m256i *)&A[base_idx + stride_i]);
        __m256i a_left = _mm256_loadu_si256((__m256i *)&A[base_idx - stride_i]);
        __m256i result = _mm256_sub_epi64(a_right, a_left);
        _mm256_storeu_si256((__m256i *)&B[base_idx], result);
      }
    }
  }
}

long compute_checksum(const uint64_t *grid)
{
  uint64_t sum = 0;
  for (int i = 1; i < (NX - 1); i++)
  {
    for (int j = 0; j < NY; j++)
    {
      for (int k = 0; k < NZ; k++)
      {
        sum += grid[i * NY * NZ + j * NZ + k];
      }
    }
  }
  return sum;
}

int main()
{
  auto *i_grid = new uint64_t[TOTAL_SIZE];
  for (int i = 0; i < NX; i++)
  {
    for (int j = 0; j < NY; j++)
    {
      for (int k = 0; k < NZ; k++)
      {
        i_grid[i * NY * NZ + j * NZ + k] = (INITIAL_VAL + i +
                                            2 * j + 3 * k);
      }
    }
  }

  // Scalar version
  auto *o_grid1 = new uint64_t[TOTAL_SIZE];
  std::fill_n(o_grid1, TOTAL_SIZE, 0);
  auto start = HR::now();
  for (int iter = 0; iter < N_ITERATIONS; ++iter)
  {
    scalar_3d_gradient(i_grid, o_grid1);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Scalar kernel time (ms): " << duration << "\n";
  uint64_t scalar_checksum = compute_checksum(o_grid1);
  cout << "Checksum: " << scalar_checksum << "\n";
  cout << "\n";

  // SSE4 Version
  auto *o_grid2 = new uint64_t[TOTAL_SIZE];
  std::fill_n(o_grid2, TOTAL_SIZE, 0);

  start = HR::now();
  for (int iter = 0; iter < N_ITERATIONS; ++iter)
  {
    sse4_3d_gradient(i_grid, o_grid2);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "SSE4 kernel time (ms): " << duration << "\n";
  uint64_t sse4_checksum = compute_checksum(o_grid2);
  cout << "Checksum: " << sse4_checksum << "\n";
  assert(scalar_checksum == sse4_checksum);
  cout << "\n";

  // AVX2 Version
  auto *o_grid3 = new uint64_t[TOTAL_SIZE];
  std::fill_n(o_grid3, TOTAL_SIZE, 0);
  start = HR::now();
  for (int iter = 0; iter < N_ITERATIONS; ++iter)
  {
    avx2_3d_gradient(i_grid, o_grid3);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "AVX2 kernel time (ms): " << duration << "\n";
  uint64_t avx2_checksum = compute_checksum(o_grid3);
  cout << "Checksum: " << avx2_checksum << "\n";
  assert(scalar_checksum == avx2_checksum);
  cout << "\n";

  delete[] i_grid;
  delete[] o_grid1;
  delete[] o_grid2;
  delete[] o_grid3;

  return EXIT_SUCCESS;
}
