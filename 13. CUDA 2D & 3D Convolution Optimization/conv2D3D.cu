#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>
#include <cstdio>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

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

const uint64_t N = (1 << 10);

__constant__ float filter2D[25];

__constant__ float filter3D[125];

__global__ void kernel2D_basic(float *input, float *output, int size, int filter_size)
{
    int filterrad = filter_size / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size)
    {
        float sum = 0.0f;
        int count = 0;

        for (int fy = -filterrad; fy <= filterrad; fy++)
        {
            for (int fx = -filterrad; fx <= filterrad; fx++)
            {
                int ix = x + fx;
                int iy = y + fy;
                if (ix >= 0 && ix < size && iy >= 0 && iy < size)
                {
                    sum += input[ix + iy * size];
                    count++;
                }
            }
        }
        output[x + y * size] = sum / count;
    }
}

__global__ void kernel2D_opt(float *input, float *output, int size, int filter_size)
{
    extern __shared__ float tile[];
    int filterrad = filter_size / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bx_dim = blockDim.x;
    int by_dim = blockDim.y;

    int x = bx * bx_dim + tx;
    int y = by * by_dim + ty;

    int tilew = bx_dim + 2 * filterrad;
    int tileh = by_dim + 2 * filterrad;

    int startx = bx * bx_dim - filterrad;
    int starty = by * by_dim - filterrad;

    for (int j = ty; j < tileh; j += by_dim)
    {
        for (int i = tx; i < tilew; i += bx_dim)
        {
            int gx = startx + i;
            int gy = starty + j;
            float val = 0.0f;
            if (gx >= 0 && gx < size && gy >= 0 && gy < size)
            {
                val = input[gx + gy * size];
            }
            tile[i + j * tilew] = val;
        }
    }
    __syncthreads();

    if (x < size && y < size)
    {
        float sum = 0.0f;
        int count = 0;
        for (int fy = -filterrad; fy <= filterrad; fy++)
        {
            for (int fx = -filterrad; fx <= filterrad; fx++)
            {
                int gx = x + fx;
                int gy = y + fy;
                if (gx >= 0 && gx < size && gy >= 0 && gy < size)
                {
                    int xl = gx - startx;
                    int yl = gy - starty;
                    sum += tile[xl + yl * tilew];
                    count++;
                }
            }
        }
        output[x + y * size] = (count > 0) ? (sum / count) : 0.0f;
    }
}

__global__ void kernel3D_basic(float *input, float *output, int size, int filter_size)
{
    int filterrad = filter_size / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < size && y < size && z < size)
    {
        float sum = 0.0f;
        int count = 0;

        for (int fz = -filterrad; fz <= filterrad; fz++)
        {
            for (int fy = -filterrad; fy <= filterrad; fy++)
            {
                for (int fx = -filterrad; fx <= filterrad; fx++)
                {
                    int ix = x + fx;
                    int iy = y + fy;
                    int iz = z + fz;
                    if (ix >= 0 && ix < size && iy >= 0 && iy < size && iz >= 0 && iz < size)
                    {
                        sum += input[ix + size * (iy + size * iz)];
                        count++;
                    }
                }
            }
        }
        output[x + size * (y + size * z)] = sum / count;
    }
}

__global__ void kernel3D_opt(float *input, float *output, int size, int filter_size)
{
    extern __shared__ float tile3D[];
    int filterrad = filter_size / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int bx_dim = blockDim.x;
    int by_dim = blockDim.y;
    int bz_dim = blockDim.z;

    int x = bx * bx_dim + tx;
    int y = by * by_dim + ty;
    int z = bz * bz_dim + tz;

    int tile_x = bx_dim + 2 * filterrad;
    int tile_y = by_dim + 2 * filterrad;
    int tile_z = bz_dim + 2 * filterrad;

    int startx = bx * bx_dim - filterrad;
    int starty = by * by_dim - filterrad;
    int start_z = bz * bz_dim - filterrad;

    for (int kz = tz; kz < tile_z; kz += bz_dim)
    {
        for (int jy = ty; jy < tile_y; jy += by_dim)
        {
            for (int ix = tx; ix < tile_x; ix += bx_dim)
            {
                int gx = startx + ix;
                int gy = starty + jy;
                int gz = start_z + kz;
                float val = 0.0f;
                if (gx >= 0 && gx < size && gy >= 0 && gy < size && gz >= 0 && gz < size)
                {
                    val = input[gx + size * (gy + size * gz)];
                }
                int idx = ix + tile_x * (jy + tile_y * kz);
                tile3D[idx] = val;
            }
        }
    }
    __syncthreads();

    if (x < size && y < size && z < size)
    {
        float sum = 0.0f;
        int count = 0;
        for (int fz = -filterrad; fz <= filterrad; fz++)
        {
            for (int fy = -filterrad; fy <= filterrad; fy++)
            {
                for (int fx = -filterrad; fx <= filterrad; fx++)
                {
                    int gx = x + fx;
                    int gy = y + fy;
                    int gz = z + fz;
                    if (gx >= 0 && gx < size && gy >= 0 && gy < size && gz >= 0 && gz < size)
                    {
                        int xl = gx - startx;
                        int yl = gy - starty;
                        int tz_local = gz - start_z;
                        int idx = xl + tile_x * (yl + tile_y * tz_local);
                        sum += tile3D[idx];
                        count++;
                    }
                }
            }
        }
        output[x + size * (y + size * z)] = (count > 0) ? (sum / count) : 0.0f;
    }
}

void check_result2D(const float *w_ref, const float *w_opt, int size)
{
    double maxdiff = 0.0;
    int numdiffs = 0;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            double this_diff = fabs(w_ref[i + size * j] - w_opt[i + size * j]);
            if (this_diff > THRESHOLD)
            {
                numdiffs++;
                if (this_diff > maxdiff)
                {
                    maxdiff = this_diff;
                }
            }
        }
    }

    if (numdiffs > 0)
    {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
             << "; Max Diff = " << maxdiff << endl;
    }
    else
    {
        cout << "No differences found between base and optimized 2D versions\n";
    }
}

void check_result3D(const float *w_ref, const float *w_opt, int size)
{
    double maxdiff = 0.0;
    int numdiffs = 0;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                double this_diff =
                    fabs(w_ref[i + size * (j + size * k)] - w_opt[i + size * (j + size * k)]);
                if (this_diff > THRESHOLD)
                {
                    numdiffs++;
                    if (this_diff > maxdiff)
                    {
                        maxdiff = this_diff;
                    }
                }
            }
        }
    }

    if (numdiffs > 0)
    {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
             << "; Max Diff = " << maxdiff << endl;
    }
    else
    {
        cout << "No differences found between base and optimized 3D versions\n";
    }
}

void print2D(const float *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.6f\t", A[i * size + j]);
        }
        cout << "\n";
    }
    cout << "\n";
}

void print3D(const float *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            for (int k = 0; k < size; ++k)
            {
                printf("%.6f\t", A[i * size * size + j * size + k]);
            }
            cout << "\n";
        }
        cout << "\n";
    }
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
    {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main()
{
    srand(42);
    int filter_size = 3;
    int filterrad = filter_size / 2;

    cout << "2D Convolution\n";

    float *hin2D = new float[N * N];
    float *houtbasic2D = new float[N * N];
    float *houtopt2D = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        hin2D[i] = rand() / (float)RAND_MAX;
    }

    float *din2D, *doutbasic2D, *doutopt2D;
    cudaCheckError(cudaMalloc(&din2D, N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&doutbasic2D, N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&doutopt2D, N * N * sizeof(float)));
    cudaCheckError(cudaMemcpy(din2D, hin2D, N * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemset(doutbasic2D, 0, N * N * sizeof(float)));
    cudaCheckError(cudaMemset(doutopt2D, 0, N * N * sizeof(float)));

    dim3 block2D(16, 16);
    dim3 grid2D((N + block2D.x - 1) / block2D.x, (N + block2D.y - 1) / block2D.y);

    double tstart = rtclock();
    kernel2D_basic<<<grid2D, block2D>>>(din2D, doutbasic2D, N, filter_size);
    cudaCheckError(cudaDeviceSynchronize());
    double tend = rtclock();
    float kernel_time_basic2D = 1000.0 * (tend - tstart);
    cout << "Kernel2D_basic time (ms): " << kernel_time_basic2D << "\n";
    cudaCheckError(cudaMemcpy(houtbasic2D, doutbasic2D, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    int halo2D = 2 * filterrad;
    int tilew2D = block2D.x + halo2D;
    int tileh2D = block2D.y + halo2D;
    size_t shared_size2D = tilew2D * tileh2D * sizeof(float);

    tstart = rtclock();
    kernel2D_opt<<<grid2D, block2D, shared_size2D>>>(din2D, doutopt2D, N, filter_size);
    cudaCheckError(cudaDeviceSynchronize());
    tend = rtclock();
    float kernel_time_opt2D = 1000.0 * (tend - tstart);
    cout << "Kernel2D_opt time (ms): " << kernel_time_opt2D << "\n";
    cudaCheckError(cudaMemcpy(houtopt2D, doutopt2D, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    check_result2D(houtbasic2D, houtopt2D, N);

    float speedup2D = kernel_time_basic2D / kernel_time_opt2D;
    cout << "2D Speedup: " << speedup2D << "x\n\n";

    cout << "3D Convolution\n";

    float *h_input3D = new float[N * N * N];
    float *h_output_basic3D = new float[N * N * N];
    float *h_output_opt3D = new float[N * N * N];

    for (int i = 0; i < N * N * N; i++)
    {
        h_input3D[i] = rand() / (float)RAND_MAX;
    }

    float *d_input3D, *d_output_basic3D, *d_output_opt3D;
    cudaCheckError(cudaMalloc(&d_input3D, N * N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_output_basic3D, N * N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_output_opt3D, N * N * N * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_input3D, h_input3D, N * N * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemset(d_output_basic3D, 0, N * N * N * sizeof(float)));
    cudaCheckError(cudaMemset(d_output_opt3D, 0, N * N * N * sizeof(float)));

    dim3 block3D(8, 8, 8);
    dim3 grid3D((N + block3D.x - 1) / block3D.x,
                (N + block3D.y - 1) / block3D.y,
                (N + block3D.z - 1) / block3D.z);

    tstart = rtclock();
    kernel3D_basic<<<grid3D, block3D>>>(d_input3D, d_output_basic3D, N, filter_size);
    cudaCheckError(cudaDeviceSynchronize());
    tend = rtclock();
    float kernel_time_basic3D = 1000.0 * (tend - tstart);
    cout << "Kernel3D_basic time (ms): " << kernel_time_basic3D << "\n";
    cudaCheckError(cudaMemcpy(h_output_basic3D, d_output_basic3D, N * N * N * sizeof(float), cudaMemcpyDeviceToHost));

    int tile_x3D = block3D.x + halo2D;
    int tile_y3D = block3D.y + halo2D;
    int tile_z3D = block3D.z + halo2D;
    size_t shared_size3D = tile_x3D * tile_y3D * tile_z3D * sizeof(float);

    tstart = rtclock();
    kernel3D_opt<<<grid3D, block3D, shared_size3D>>>(d_input3D, d_output_opt3D, N, filter_size);
    cudaCheckError(cudaDeviceSynchronize());
    tend = rtclock();
    float kernel_time_opt3D = 1000.0 * (tend - tstart);
    cout << "Kernel3D_opt time (ms): " << kernel_time_opt3D << "\n";
    cudaCheckError(cudaMemcpy(h_output_opt3D, d_output_opt3D, N * N * N * sizeof(float), cudaMemcpyDeviceToHost));

    check_result3D(h_output_basic3D, h_output_opt3D, N);

    float speedup3D = kernel_time_basic3D / kernel_time_opt3D;
    cout << "3D Speedup: " << speedup3D << "x\n\n";

    delete[] hin2D;
    delete[] houtbasic2D;
    delete[] houtopt2D;
    delete[] h_input3D;
    delete[] h_output_basic3D;
    delete[] h_output_opt3D;

    cudaFree(din2D);
    cudaFree(doutbasic2D);
    cudaFree(doutopt2D);
    cudaFree(d_input3D);
    cudaFree(d_output_basic3D);
    cudaFree(d_output_opt3D);

    return EXIT_SUCCESS;
}
