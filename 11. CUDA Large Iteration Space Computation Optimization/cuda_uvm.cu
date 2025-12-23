#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>

#define cudaCheckError(ans)                   \
    {                                         \
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

using HR = std::chrono::high_resolution_clock;

__constant__ double d_a_const[120];

__global__ void grid_kernel_uvm(
    const double *b30, double kk,
    uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
    uint32_t s6, uint32_t s7, uint32_t s8, uint32_t s9, uint32_t s10,
    double *out_pts,
    uint64_t *out_indices,
    unsigned long long int *out_count,
    uint64_t total_points)
{
    __shared__ double s_b[30];

    unsigned int tid = threadIdx.x;
    if (tid < 30)
    {
        s_b[tid] = b30[tid];
    }
    __syncthreads();

    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + tid;
    if (gid >= total_points)
        return;

    uint64_t tmp = gid;
    uint32_t r10 = tmp % s10;
    tmp /= s10;
    uint32_t r9 = tmp % s9;
    tmp /= s9;
    uint32_t r8 = tmp % s8;
    tmp /= s8;
    uint32_t r7 = tmp % s7;
    tmp /= s7;
    uint32_t r6 = tmp % s6;
    tmp /= s6;
    uint32_t r5 = tmp % s5;
    tmp /= s5;
    uint32_t r4 = tmp % s4;
    tmp /= s4;
    uint32_t r3 = tmp % s3;
    tmp /= s3;
    uint32_t r2 = tmp % s2;
    tmp /= s2;
    uint32_t r1 = tmp;

    double x1 = s_b[0] + (double)r1 * s_b[2];
    double x2 = s_b[3] + (double)r2 * s_b[5];
    double x3 = s_b[6] + (double)r3 * s_b[8];
    double x4 = s_b[9] + (double)r4 * s_b[11];
    double x5 = s_b[12] + (double)r5 * s_b[14];
    double x6 = s_b[15] + (double)r6 * s_b[17];
    double x7 = s_b[18] + (double)r7 * s_b[20];
    double x8 = s_b[21] + (double)r8 * s_b[23];
    double x9 = s_b[24] + (double)r9 * s_b[26];
    double x10 = s_b[27] + (double)r10 * s_b[29];

    bool ok = true;

    for (int g = 0; g < 10; ++g)
    {
        int base = g * 12;

        double q = fabs(
            d_a_const[base + 0] * x1 +
            d_a_const[base + 1] * x2 +
            d_a_const[base + 2] * x3 +
            d_a_const[base + 3] * x4 +
            d_a_const[base + 4] * x5 +
            d_a_const[base + 5] * x6 +
            d_a_const[base + 6] * x7 +
            d_a_const[base + 7] * x8 +
            d_a_const[base + 8] * x9 +
            d_a_const[base + 9] * x10 -
            d_a_const[base + 10]);

        double e = kk * d_a_const[base + 11];

        if (q > e)
        {
            ok = false;
            break;
        }
    }

    if (ok)
    {
        unsigned long long int idx = atomicAdd(out_count, 1ULL);

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

        out_indices[idx] = gid;
    }
}

struct ResultEntry
{
    uint64_t index;
    double values[10];

    bool operator<(const ResultEntry &other) const
    {
        return index < other.index;
    }
};

int main()
{
    double a[120];
    double b[30];

    int device;
    cudaCheckError(cudaGetDevice(&device));

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
            fprintf(stderr, "Error reading disp.txt at index %d\n", i);
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
            fprintf(stderr, "Error reading grid.txt at index %d\n", i);
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

    double *uvm_b;
    double *uvm_out;
    uint64_t *uvm_indices;
    unsigned long long int *uvm_count;

    uint64_t MAX_POINTS_ALLOC = std::min(total, (uint64_t)1 << 26);

    cudaCheckError(cudaMallocManaged(&uvm_b, 30 * sizeof(double)));
    cudaCheckError(cudaMallocManaged(&uvm_out, MAX_POINTS_ALLOC * 10 * sizeof(double)));
    cudaCheckError(cudaMallocManaged(&uvm_indices, MAX_POINTS_ALLOC * sizeof(uint64_t)));
    cudaCheckError(cudaMallocManaged(&uvm_count, sizeof(unsigned long long int)));

    for (int i = 0; i < 30; i++)
    {
        uvm_b[i] = b[i];
    }
    *uvm_count = 0;

    cudaCheckError(cudaMemcpyToSymbol(d_a_const, a, 120 * sizeof(double)));

    cudaCheckError(cudaMemAdvise(uvm_b, 30 * sizeof(double),
                                 cudaMemAdviseSetReadMostly, device));

    cudaCheckError(cudaMemAdvise(uvm_out, MAX_POINTS_ALLOC * 10 * sizeof(double),
                                 cudaMemAdviseSetPreferredLocation, device));
    cudaCheckError(cudaMemAdvise(uvm_indices, MAX_POINTS_ALLOC * sizeof(uint64_t),
                                 cudaMemAdviseSetPreferredLocation, device));
    cudaCheckError(cudaMemAdvise(uvm_count, sizeof(unsigned long long int),
                                 cudaMemAdviseSetPreferredLocation, device));

    cudaCheckError(cudaMemAdvise(uvm_out, MAX_POINTS_ALLOC * 10 * sizeof(double),
                                 cudaMemAdviseSetAccessedBy, device));
    cudaCheckError(cudaMemAdvise(uvm_indices, MAX_POINTS_ALLOC * sizeof(uint64_t),
                                 cudaMemAdviseSetAccessedBy, device));

    cudaCheckError(cudaMemPrefetchAsync(uvm_b, 30 * sizeof(double), device, 0));

    cudaCheckError(cudaMemPrefetchAsync(uvm_out, MAX_POINTS_ALLOC * 10 * sizeof(double),
                                        device, 0));
    cudaCheckError(cudaMemPrefetchAsync(uvm_indices, MAX_POINTS_ALLOC * sizeof(uint64_t),
                                        device, 0));
    cudaCheckError(cudaMemPrefetchAsync(uvm_count, sizeof(unsigned long long int),
                                        device, 0));

    cudaCheckError(cudaDeviceSynchronize());

    const int THREADS = 256;
    uint64_t blocks = (total + THREADS - 1) / THREADS;

    if (blocks > INT_MAX)
    {
        fprintf(stderr, "Error: Too many blocks required (%llu). Use chunking.\n",
                (unsigned long long)blocks);
        return 1;
    }

    auto t0 = HR::now();

    grid_kernel_uvm<<<(int)blocks, THREADS>>>(
        uvm_b, kk,
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
        uvm_out, uvm_indices, uvm_count, total);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    auto t1 = HR::now();

    unsigned long long int h_count = *uvm_count;

    if (h_count > MAX_POINTS_ALLOC)
    {
        h_count = MAX_POINTS_ALLOC;
    }

    if (h_count > 0)
    {
        cudaCheckError(cudaMemPrefetchAsync(uvm_indices, h_count * sizeof(uint64_t),
                                            cudaCpuDeviceId, 0));
        cudaCheckError(cudaMemPrefetchAsync(uvm_out, h_count * 10 * sizeof(double),
                                            cudaCpuDeviceId, 0));
        cudaCheckError(cudaDeviceSynchronize());
    }

    std::vector<ResultEntry> results(h_count);

    for (uint64_t i = 0; i < h_count; i++)
    {
        results[i].index = uvm_indices[i];
        for (int j = 0; j < 10; j++)
        {
            results[i].values[j] = uvm_out[i * 10 + j];
        }
    }

    std::sort(results.begin(), results.end());

    FILE *fout = fopen("results-v3.txt", "w");
    if (!fout)
    {
        fprintf(stderr, "Error creating results-v3.txt\n");
    }
    else
    {
        for (uint64_t i = 0; i < h_count; ++i)
        {
            fprintf(fout, "%lf\t", results[i].values[0]);
            fprintf(fout, "%lf\t", results[i].values[1]);
            fprintf(fout, "%lf\t", results[i].values[2]);
            fprintf(fout, "%lf\t", results[i].values[3]);
            fprintf(fout, "%lf\t", results[i].values[4]);
            fprintf(fout, "%lf\t", results[i].values[5]);
            fprintf(fout, "%lf\t", results[i].values[6]);
            fprintf(fout, "%lf\t", results[i].values[7]);
            fprintf(fout, "%lf\t", results[i].values[8]);
            fprintf(fout, "%lf\n", results[i].values[9]);
        }
        fclose(fout);
    }

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("GPU time: %.2f s\n", elapsed);
    printf("result pnts: %llu\n", (unsigned long long)h_count);

    cudaCheckError(cudaFree(uvm_b));
    cudaCheckError(cudaFree(uvm_out));
    cudaCheckError(cudaFree(uvm_indices));
    cudaCheckError(cudaFree(uvm_count));

    return 0;
}
