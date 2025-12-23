#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>

#define cudaCheckError(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

__global__ void grid_kernel(const double *b30, const double *a120, double kk,
                            uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                            uint32_t s6, uint32_t s7, uint32_t s8, uint32_t s9, uint32_t s10,
                            double *out_pts, unsigned long long int *out_count, uint64_t total_points)
{
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_points)
    return;

  uint64_t tmp = gid;
  uint32_t r1 = tmp % s1;
  tmp /= s1;
  uint32_t r2 = tmp % s2;
  tmp /= s2;
  uint32_t r3 = tmp % s3;
  tmp /= s3;
  uint32_t r4 = tmp % s4;
  tmp /= s4;
  uint32_t r5 = tmp % s5;
  tmp /= s5;
  uint32_t r6 = tmp % s6;
  tmp /= s6;
  uint32_t r7 = tmp % s7;
  tmp /= s7;
  uint32_t r8 = tmp % s8;
  tmp /= s8;
  uint32_t r9 = tmp % s9;
  tmp /= s9;
  uint32_t r10 = tmp % s10;

  double x1 = b30[0] + (double)r1 * b30[2];
  double x2 = b30[3] + (double)r2 * b30[5];
  double x3 = b30[6] + (double)r3 * b30[8];
  double x4 = b30[9] + (double)r4 * b30[11];
  double x5 = b30[12] + (double)r5 * b30[14];
  double x6 = b30[15] + (double)r6 * b30[17];
  double x7 = b30[18] + (double)r7 * b30[20];
  double x8 = b30[21] + (double)r8 * b30[23];
  double x9 = b30[24] + (double)r9 * b30[26];
  double x10 = b30[27] + (double)r10 * b30[29];

  double q[10];

  for (int g = 0; g < 10; ++g)
  {
    const double *c = a120 + g * 12;
    double d = a120[g * 12 + 10];
    q[g] = fabs(c[0] * x1 + c[1] * x2 + c[2] * x3 + c[3] * x4 + c[4] * x5 +
                c[5] * x6 + c[6] * x7 + c[7] * x8 + c[8] * x9 + c[9] * x10 - d);
  }

  bool ok = true;
  for (int g = 0; g < 10; ++g)
  {
    double ey = a120[g * 12 + 11];
    double e = kk * ey;
    if (q[g] > e)
    {
      ok = false;
      break;
    }
  }

  if (ok)
  {
    unsigned long long int idx =
        atomicAdd(out_count, (unsigned long long int)1);
    uint64_t base = idx * 10ull;
    out_pts[base + 0] = x1;
    out_pts[base + 1] = x2;
    out_pts[base + 2] = x3;
    out_pts[base + 3] = x4;
    out_pts[base + 4] = x5;
    out_pts[base + 5] = x6;
    out_pts[base + 6] = x7;
    out_pts[base + 7] = x8;
    out_pts[base + 8] = x9;
    out_pts[base + 9] = x10;
  }
}

int main()
{
  double a[120];
  double b[30];

  FILE *fp = fopen("disp.txt", "r");
  if (!fp)
  {
    fprintf(stderr, "Error: could not open disp.txt\n");
    return 1;
  }
  for (int i = 0; i < 120; ++i)
  {
    if (fscanf(fp, "%lf", &a[i]) != 1)
    {
      fprintf(stderr, "Error reading disp.txt\n");
      return 1;
    }
  }
  fclose(fp);

  FILE *fg = fopen("grid.txt", "r");
  if (!fg)
  {
    fprintf(stderr, "Error: could not open grid.txt\n");
    return 1;
  }
  for (int i = 0; i < 30; ++i)
  {
    if (fscanf(fg, "%lf", &b[i]) != 1)
    {
      fprintf(stderr, "Error reading grid.txt\n");
      return 1;
    }
  }
  fclose(fg);

  int s1 = floor((b[1] - b[0]) / b[2]);
  int s2 = floor((b[4] - b[3]) / b[5]);
  int s3 = floor((b[7] - b[6]) / b[8]);
  int s4 = floor((b[10] - b[9]) / b[11]);
  int s5 = floor((b[13] - b[12]) / b[14]);
  int s6 = floor((b[16] - b[15]) / b[17]);
  int s7 = floor((b[19] - b[18]) / b[20]);
  int s8 = floor((b[22] - b[21]) / b[23]);
  int s9 = floor((b[25] - b[24]) / b[26]);
  int s10 = floor((b[28] - b[27]) / b[29]);

  uint64_t total = 1ull * s1 * s2 * s3 * s4 * s5 * s6 * s7 * s8 * s9 * s10;

  double kk = 0.3;

  double *d_a, *d_b, *d_out;
  unsigned long long int *d_count;
  cudaCheckError(cudaMalloc(&d_a, 120 * sizeof(double)));
  cudaCheckError(cudaMalloc(&d_b, 30 * sizeof(double)));
  cudaCheckError(cudaMalloc(&d_count, sizeof(unsigned long long int)));

  uint64_t MAX_POINTS_ALLOC = std::min(total, (uint64_t)1 << 26);
  cudaCheckError(cudaMalloc(&d_out, MAX_POINTS_ALLOC * 10 * sizeof(double)));

  cudaCheckError(cudaMemcpy(d_a, a, 120 * sizeof(double), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_b, b, 30 * sizeof(double), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_count, 0, sizeof(unsigned long long int)));

  const int THREADS = 256;
  uint64_t blocks = (total + THREADS - 1) / THREADS;
  if (blocks > 2147483647)
  {
    fprintf(stderr, "Too many total points..\n");
    return 1;
  }
  float kernel_time = 0.0f;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  grid_kernel<<<(int)blocks, THREADS>>>(d_b, d_a, kk,
                                        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                        d_out, d_count, total);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaGetLastError());
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);

  std::cout << "Kernel time: " << kernel_time / 1000 << " s\n";

  unsigned long long int h_count = 0;
  cudaCheckError(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

  if (h_count > MAX_POINTS_ALLOC)
    h_count = MAX_POINTS_ALLOC;

  double *h_out = (double *)malloc(h_count * 10 * sizeof(double));
  if (h_count > 0)
    cudaCheckError(cudaMemcpy(h_out, d_out, h_count * 10 * sizeof(double), cudaMemcpyDeviceToHost));

  FILE *fout = fopen("results-v1.txt", "w");
  if (!fout)
  {
    fprintf(stderr, "Error creating results-v1.txt\n");
  }
  else
  {
    for (uint64_t i = 0; i < h_count; ++i)
    {
      for (int j = 0; j < 10; ++j)
        fprintf(fout, "%lf\t", h_out[i * 10 + j]);
      fprintf(fout, "\n");
    }
    fclose(fout);
  }

  printf("Valid points: %llu\n", (unsigned long long)h_count);

  free(h_out);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  cudaFree(d_count);

  return 0;
}
