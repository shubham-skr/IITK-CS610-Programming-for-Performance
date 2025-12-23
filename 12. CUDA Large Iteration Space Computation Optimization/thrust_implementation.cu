#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

using HR = std::chrono::high_resolution_clock;

__constant__ double d_a[120];
__constant__ double d_b[30];

struct CheckPointFunctor
{
    double kk;
    uint32_t s[10];

    __device__ bool operator()(uint64_t gid) const
    {
        uint64_t tmp = gid;
        uint32_t r10 = tmp % s[9];
        tmp /= s[9];
        uint32_t r9 = tmp % s[8];
        tmp /= s[8];
        uint32_t r8 = tmp % s[7];
        tmp /= s[7];
        uint32_t r7 = tmp % s[6];
        tmp /= s[6];
        uint32_t r6 = tmp % s[5];
        tmp /= s[5];
        uint32_t r5 = tmp % s[4];
        tmp /= s[4];
        uint32_t r4 = tmp % s[3];
        tmp /= s[3];
        uint32_t r3 = tmp % s[2];
        tmp /= s[2];
        uint32_t r2 = tmp % s[1];
        tmp /= s[1];
        uint32_t r1 = tmp;

        double x1 = d_b[0] + r1 * d_b[2];
        double x2 = d_b[3] + r2 * d_b[5];
        double x3 = d_b[6] + r3 * d_b[8];
        double x4 = d_b[9] + r4 * d_b[11];
        double x5 = d_b[12] + r5 * d_b[14];
        double x6 = d_b[15] + r6 * d_b[17];
        double x7 = d_b[18] + r7 * d_b[20];
        double x8 = d_b[21] + r8 * d_b[23];
        double x9 = d_b[24] + r9 * d_b[26];
        double x10 = d_b[27] + r10 * d_b[29];

        for (int g = 0; g < 10; g++)
        {
            int base = g * 12;
            double q = fabs(
                d_a[base + 0] * x1 + d_a[base + 1] * x2 + d_a[base + 2] * x3 + d_a[base + 3] * x4 +
                d_a[base + 4] * x5 + d_a[base + 5] * x6 + d_a[base + 6] * x7 + d_a[base + 7] * x8 +
                d_a[base + 8] * x9 + d_a[base + 9] * x10 - d_a[base + 10]);
            if (q > kk * d_a[base + 11])
                return false;
        }
        return true;
    }
};

int main()
{
    double a[120], b[30];

    FILE *fp = fopen("disp.txt", "r");
    if (!fp)
    {
        printf("Error opening disp.txt\n");
        return 1;
    }
    for (int i = 0; i < 120; i++)
    {
        if (fscanf(fp, "%lf", &a[i]) != 1)
        {
            printf("Error reading disp.txt\n");
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    FILE *fg = fopen("grid.txt", "r");
    if (!fg)
    {
        printf("Error opening grid.txt\n");
        return 1;
    }
    for (int i = 0; i < 30; i++)
    {
        if (fscanf(fg, "%lf", &b[i]) != 1)
        {
            printf("Error reading grid.txt\n");
            fclose(fg);
            return 1;
        }
    }
    fclose(fg);

    cudaMemcpyToSymbol(d_a, a, sizeof(a));
    cudaMemcpyToSymbol(d_b, b, sizeof(b));

    uint32_t s[10];
    s[0] = floor((b[1] - b[0]) / b[2]);
    s[1] = floor((b[4] - b[3]) / b[5]);
    s[2] = floor((b[7] - b[6]) / b[8]);
    s[3] = floor((b[10] - b[9]) / b[11]);
    s[4] = floor((b[13] - b[12]) / b[14]);
    s[5] = floor((b[16] - b[15]) / b[17]);
    s[6] = floor((b[19] - b[18]) / b[20]);
    s[7] = floor((b[22] - b[21]) / b[23]);
    s[8] = floor((b[25] - b[24]) / b[26]);
    s[9] = floor((b[28] - b[27]) / b[29]);

    uint64_t total = 1ULL;
    for (int i = 0; i < 10; i++)
        total *= s[i];

    CheckPointFunctor pred;
    pred.kk = 0.3;
    for (int i = 0; i < 10; i++)
        pred.s[i] = s[i];

    const uint64_t CHUNK = 200000000ULL;

    thrust::host_vector<uint64_t> h_valid;

    auto t0 = HR::now();

    for (uint64_t offset = 0; offset < total; offset += CHUNK)
    {
        uint64_t this_chunk = std::min(CHUNK, total - offset);

        thrust::counting_iterator<uint64_t> first(offset);
        thrust::counting_iterator<uint64_t> last(offset + this_chunk);

        thrust::device_vector<uint64_t> tmp(this_chunk);

        auto end_it =
            thrust::copy_if(thrust::device, first, last, tmp.begin(), pred);

        size_t count = end_it - tmp.begin();

        thrust::host_vector<uint64_t> h_chunk(count);
        thrust::copy_n(tmp.begin(), count, h_chunk.begin());

        h_valid.insert(h_valid.end(), h_chunk.begin(), h_chunk.end());
    }

    auto t1 = HR::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::sort(h_valid.begin(), h_valid.end());

    printf("Total valid = %zu\n", h_valid.size());
    printf("GPU time: %.6f s\n", elapsed);

    FILE *fout = fopen("results-v4.txt", "w");

    for (uint64_t gid : h_valid)
    {
        uint64_t tmp = gid;
        uint32_t r10 = tmp % s[9];
        tmp /= s[9];
        uint32_t r9 = tmp % s[8];
        tmp /= s[8];
        uint32_t r8 = tmp % s[7];
        tmp /= s[7];
        uint32_t r7 = tmp % s[6];
        tmp /= s[6];
        uint32_t r6 = tmp % s[5];
        tmp /= s[5];
        uint32_t r5 = tmp % s[4];
        tmp /= s[4];
        uint32_t r4 = tmp % s[3];
        tmp /= s[3];
        uint32_t r3 = tmp % s[2];
        tmp /= s[2];
        uint32_t r2 = tmp % s[1];
        tmp /= s[1];
        uint32_t r1 = tmp;

        double x1 = b[0] + r1 * b[2];
        double x2 = b[3] + r2 * b[5];
        double x3 = b[6] + r3 * b[8];
        double x4 = b[9] + r4 * b[11];
        double x5 = b[12] + r5 * b[14];
        double x6 = b[15] + r6 * b[17];
        double x7 = b[18] + r7 * b[20];
        double x8 = b[21] + r8 * b[23];
        double x9 = b[24] + r9 * b[26];
        double x10 = b[27] + r10 * b[29];

        fprintf(fout,
                "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);
    }

    fclose(fout);

    return 0;
}
