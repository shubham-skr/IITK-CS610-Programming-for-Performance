#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <string>
#include <cmath>

#define THRESHOLD (1e-9)

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define TILE_SIZE 8

const uint64_t N = 512;

__global__ void shmem_kernel(const double *in, double *out, uint64_t N)
{
  __shared__ double tile[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];

  int i = blockIdx.x * TILE_SIZE + threadIdx.x + 1;
  int j = blockIdx.y * TILE_SIZE + threadIdx.y + 1;
  int k = blockIdx.z * TILE_SIZE + threadIdx.z + 1;

  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;
  int tz = threadIdx.z + 1;

  if (i < N && j < N && k < N)
    tile[tx][ty][tz] = in[i * N * N + j * N + k];

  if (threadIdx.x == 0 && i > 0)
    tile[0][ty][tz] = in[(i - 1) * N * N + j * N + k];
  if (threadIdx.x == TILE_SIZE - 1 && i < N - 1)
    tile[TILE_SIZE + 1][ty][tz] = in[(i + 1) * N * N + j * N + k];
  if (threadIdx.y == 0 && j > 0)
    tile[tx][0][tz] = in[i * N * N + (j - 1) * N + k];
  if (threadIdx.y == TILE_SIZE - 1 && j < N - 1)
    tile[tx][TILE_SIZE + 1][tz] = in[i * N * N + (j + 1) * N + k];
  if (threadIdx.z == 0 && k > 0)
    tile[tx][ty][0] = in[i * N * N + j * N + (k - 1)];
  if (threadIdx.z == TILE_SIZE - 1 && k < N - 1)
    tile[tx][ty][TILE_SIZE + 1] = in[i * N * N + j * N + (k + 1)];

  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
  {
    out[i * N * N + j * N + k] = 0.8 * (tile[tx - 1][ty][tz] + tile[tx + 1][ty][tz] +
                                        tile[tx][ty - 1][tz] + tile[tx][ty + 1][tz] +
                                        tile[tx][ty][tz - 1] + tile[tx][ty][tz + 1]);
  }
}

__host__ void stencil(const double *in, double *out)
{
  for (uint64_t i = 1; i < (N - 1); i++)
  {
    for (uint64_t j = 1; j < (N - 1); j++)
    {
      for (uint64_t k = 1; k < (N - 1); k++)
      {
        out[i * N * N + j * N + k] =
            0.8 *
            (in[(i - 1) * N * N + j * N + k] + in[(i + 1) * N * N + j * N + k] +
             in[i * N * N + (j - 1) * N + k] + in[i * N * N + (j + 1) * N + k] +
             in[i * N * N + j * N + (k - 1)] + in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const double *w_ref, const double *w_opt, const uint64_t size)
{
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++)
  {
    for (uint64_t j = 0; j < size; j++)
    {
      for (uint64_t k = 0; k < size; k++)
      {
        double this_diff =
            w_ref[i * N * N + j * N + k] - w_opt[i * N * N + j * N + k];
        if (std::fabs(this_diff) > THRESHOLD)
        {
          numdiffs++;
          if (std::fabs(this_diff) > maxdiff)
          {
            maxdiff = std::fabs(this_diff);
          }
        }
      }
    }
  }

  if (numdiffs > 0)
  {
    cout << "ERROR: " << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  }
  else
  {
    cout << "SUCCESS: No differences found between base and test versions.\n";
  }
}

int main()
{
  uint64_t NUM_ELEMS = (N * N * N);
  uint64_t SIZE_BYTES = NUM_ELEMS * sizeof(double);

  cout << "3D Stencil N=" << N << " (" << SIZE_BYTES / (1024.0 * 1024.0) << " MB)" << endl;

  auto *h_in = new double[NUM_ELEMS];
  auto *h_out_cpu = new double[NUM_ELEMS];
  auto *h_out_gpu = new double[NUM_ELEMS];

  srand(42);
  for (uint64_t i = 0; i < NUM_ELEMS; i++)
  {
    h_in[i] = static_cast<double>(rand() % 100);
  }
  std::fill_n(h_out_cpu, NUM_ELEMS, 0.0);
  std::fill_n(h_out_gpu, NUM_ELEMS, 0.0);

  // CPU Baseline
  cout << "\n--- CPU Baseline ---" << endl;
  auto cpu_start = HR::now();
  stencil(h_in, h_out_cpu);
  auto cpu_end = HR::now();
  auto duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
  cout << "Stencil time on CPU: " << duration << " ms\n";

  // GPU Setup
  double *d_in, *d_out;
  cudaCheckError(cudaMalloc((void **)&d_in, SIZE_BYTES));
  cudaCheckError(cudaMalloc((void **)&d_out, SIZE_BYTES));

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float kernel_time = 0.0f;

  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));

  // --- (ii) Shared Memory Kernel ---
  cout << "\n--- (ii) Shared Memory Kernel ---" << endl;
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
  dim3 blockDim(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (N + blockDim.z - 1) / blockDim.z);

  cudaEventRecord(start);
  shmem_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
  cudaCheckError(cudaGetLastError());
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernel_time, start, end);
  cout << "Shmem kernel time: " << kernel_time << " ms\n";

  cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
  check_result(h_out_cpu, h_out_gpu, N);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(d_in);
  cudaFree(d_out);

  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;

  return EXIT_SUCCESS;
}
