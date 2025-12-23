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

#define cudaCheckError(ans)                   \
    {                                         \
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

__global__ void pinned_kernel(const double *in,
                              double *out,
                              uint64_t N)
{
    int k_base = (blockIdx.x * TILE_SIZE + threadIdx.x) * 2 + 1;
    int j = blockIdx.y * TILE_SIZE + threadIdx.y + 1;
    int i = blockIdx.z * TILE_SIZE + threadIdx.z + 1;

    __shared__ double tile[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE * 2 + 2];

    int tx = threadIdx.x * 2 + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int k = k_base;
    if (i < N && j < N && k < N)
        tile[tz][ty][tx] = in[i * N * N + j * N + k];
    else
        tile[tz][ty][tx] = 0.0;

    k = k_base + 1;
    if (i < N && j < N && k < N)
        tile[tz][ty][tx + 1] = in[i * N * N + j * N + k];
    else
        tile[tz][ty][tx + 1] = 0.0;

    if (threadIdx.x == 0)
    {
        int k_left = k_base - 1;
        if (k_left >= 0 && i < N && j < N)
            tile[tz][ty][0] = in[i * N * N + j * N + k_left];
        else
            tile[tz][ty][0] = 0.0;
    }

    if (threadIdx.x == TILE_SIZE - 1)
    {
        int k_right = k_base + 2;
        if (k_right < N && i < N && j < N)
            tile[tz][ty][tx + 2] = in[i * N * N + j * N + k_right];
        else
            tile[tz][ty][tx + 2] = 0.0;
    }

    if (threadIdx.y == 0)
    {
        int jm = j - 1;
        if (jm >= 0 && i < N)
        {
            if (k_base < (int)N)
                tile[tz][0][tx] = in[i * N * N + jm * N + k_base];
            else
                tile[tz][0][tx] = 0.0;
            if ((k_base + 1) < N)
                tile[tz][0][tx + 1] = in[i * N * N + jm * N + (k_base + 1)];
            else
                tile[tz][0][tx + 1] = 0.0;
        }
        else
        {
            tile[tz][0][tx] = 0.0;
            tile[tz][0][tx + 1] = 0.0;
        }
    }

    if (threadIdx.y == TILE_SIZE - 1)
    {
        int jp = j + 1;
        if (jp < N && i < N)
        {
            if (k_base < N)
                tile[tz][TILE_SIZE + 1][tx] = in[i * N * N + jp * N + k_base];
            else
                tile[tz][TILE_SIZE + 1][tx] = 0.0;
            if ((k_base + 1) < N)
                tile[tz][TILE_SIZE + 1][tx + 1] = in[i * N * N + jp * N + (k_base + 1)];
            else
                tile[tz][TILE_SIZE + 1][tx + 1] = 0.0;
        }
        else
        {
            tile[tz][TILE_SIZE + 1][tx] = 0.0;
            tile[tz][TILE_SIZE + 1][tx + 1] = 0.0;
        }
    }

    if (threadIdx.z == 0)
    {
        int im = i - 1;
        if (im >= 0 && j < N)
        {
            if (k_base < N)
                tile[0][ty][tx] = in[im * N * N + j * N + k_base];
            else
                tile[0][ty][tx] = 0.0;
            if ((k_base + 1) < N)
                tile[0][ty][tx + 1] = in[im * N * N + j * N + (k_base + 1)];
            else
                tile[0][ty][tx + 1] = 0.0;
        }
        else
        {
            tile[0][ty][tx] = 0.0;
            tile[0][ty][tx + 1] = 0.0;
        }
    }

    if (threadIdx.z == TILE_SIZE - 1)
    {
        int ip = i + 1;
        if (ip < N && j < N)
        {
            if (k_base < N)
                tile[TILE_SIZE + 1][ty][tx] = in[ip * N * N + j * N + k_base];
            else
                tile[TILE_SIZE + 1][ty][tx] = 0.0;
            if ((k_base + 1) < N)
                tile[TILE_SIZE + 1][ty][tx + 1] = in[ip * N * N + j * N + (k_base + 1)];
            else
                tile[TILE_SIZE + 1][ty][tx + 1] = 0.0;
        }
        else
        {
            tile[TILE_SIZE + 1][ty][tx] = 0.0;
            tile[TILE_SIZE + 1][ty][tx + 1] = 0.0;
        }
    }

    __syncthreads();

    k = k_base;
    int local_tx = tx;
    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
        double sum =
            tile[tz - 1][ty][local_tx] + tile[tz + 1][ty][local_tx] +
            tile[tz][ty - 1][local_tx] + tile[tz][ty + 1][local_tx] +
            tile[tz][ty][local_tx - 1] + tile[tz][ty][local_tx + 1];

        out[i * N * N + j * N + k] = 0.8 * sum;
    }

    k = k_base + 1;
    local_tx = tx + 1;
    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
        double sum =
            tile[tz - 1][ty][local_tx] + tile[tz + 1][ty][local_tx] +
            tile[tz][ty - 1][local_tx] + tile[tz][ty + 1][local_tx] +
            tile[tz][ty][local_tx - 1] + tile[tz][ty][local_tx + 1];

        out[i * N * N + j * N + k] = 0.8 * sum;
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
    double *h_in_gpu, *h_out_gpu;
    cudaCheckError(cudaHostAlloc((void **)&h_in_gpu, SIZE_BYTES, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void **)&h_out_gpu, SIZE_BYTES, cudaHostAllocDefault));

    srand(42);
    for (uint64_t i = 0; i < NUM_ELEMS; i++)
    {
        h_in[i] = static_cast<double>(rand() % 100);
        h_in_gpu[i] = h_in[i];
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

    // --- (iv) Pinned Kernel ---
    cout << "\n--- (iv) Pinned Kernel ---" << endl;
    cudaEvent_t cstart, cend;
    cudaEventCreate(&cstart);
    cudaEventCreate(&cend);
    cudaEventRecord(cstart);
    cudaCheckError(cudaMemcpy(d_in, h_in_gpu, SIZE_BYTES, cudaMemcpyHostToDevice));
    cudaEventRecord(cend);
    cudaEventSynchronize(cend);

    float copy_host2Device = 0.0f;
    cudaEventElapsedTime(&copy_host2Device, cstart, cend);
    cout << "Host to Device copy time (pinned):" << copy_host2Device << " ms\n";

    cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
    dim3 blockDim(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N - 2 + TILE_SIZE * 2 - 1) / (TILE_SIZE * 2),
        (N - 2 + TILE_SIZE - 1) / TILE_SIZE,
        (N - 2 + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    pinned_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    cudaCheckError(cudaGetLastError());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float kernel_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, start, end);
    cout << "Pinned kernel time: " << kernel_time << " ms\n";

    cudaEventRecord(cstart);
    cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
    cudaEventRecord(cend);
    cudaEventSynchronize(cend);
    float copy_device2Host = 0.0f;
    cudaEventElapsedTime(&copy_device2Host, cstart, cend);
    cout << "Device to Host copy time (pinned):" << copy_device2Host << " ms\n";

    check_result(h_out_cpu, h_out_gpu, N);

    cudaEventDestroy(cstart);
    cudaEventDestroy(cend);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in_gpu);
    cudaFreeHost(h_out_gpu);

    delete[] h_in;

    return EXIT_SUCCESS;
}
