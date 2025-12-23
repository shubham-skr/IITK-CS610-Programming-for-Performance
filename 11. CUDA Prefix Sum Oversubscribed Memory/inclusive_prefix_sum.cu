#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <numeric>
#include <iterator>
#include <cstdlib>
using namespace std;

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

const uint64_t N = 1 << 30;
const int BLOCK_SIZE = 1024;

__global__ void block_prefix_sum(uint64_t *data, uint64_t *blockSums, uint64_t n)
{
    __shared__ uint64_t temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    uint64_t blockStart = (uint64_t)blockIdx.x * (uint64_t)BLOCK_SIZE;
    uint64_t gid = blockStart + tid;

    uint64_t val = (gid < n) ? data[gid] : 0ULL;
    temp[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1)
    {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE)
            temp[idx] += temp[idx - offset];
        __syncthreads();
    }

    if (tid == 0)
        temp[BLOCK_SIZE - 1] = 0ULL;
    __syncthreads();

    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1)
    {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE)
        {
            uint64_t t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    uint64_t exclusive = temp[tid];
    uint64_t inclusive = exclusive + val;

    if (gid < n)
        data[gid] = inclusive;

    uint64_t remaining = (n > blockStart) ? (n - blockStart) : 0;
    uint64_t block_valid = (remaining < (uint64_t)BLOCK_SIZE) ? (uint64_t)remaining : (uint64_t)BLOCK_SIZE;

    if (blockSums && block_valid > 0 && tid == (int)block_valid - 1)
    {
        blockSums[blockIdx.x] = inclusive;
    }
}

__global__ void add_offsets(uint64_t *data, const uint64_t *blockOffsets,
                            uint64_t globalOffset, uint64_t n)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n)
    {
        uint64_t offset = globalOffset;
        if (blockIdx.x > 0)
            offset += blockOffsets[blockIdx.x];
        data[gid] += offset;
    }
}

__host__ void check_result(const uint64_t *w_ref, const uint64_t *w_opt,
                           const uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        if (w_ref[i] != w_opt[i])
        {
            cout << "Mismatch at index " << i << ": ref=" << w_ref[i]
                 << ", opt=" << w_opt[i] << "\n";
            assert(false);
        }
    }
    cout << "No differences found between base and test versions\n";
}

__host__ void inclusive_prefix_sum(const uint64_t *input, uint64_t *output)
{
    output[0] = input[0];
    for (uint64_t i = 1; i < N; i++)
    {
        output[i] = output[i - 1] + input[i];
    }
}

__host__ void cte_sum(const uint64_t *h_in, uint64_t *h_out_gpu)
{
    cout << "\n-----CTE SUM-----\n";
    size_t freeMem = 0, totalMem = 0;
    cudaCheckError(cudaMemGetInfo(&freeMem, &totalMem));

    uint64_t chunk_elems = min((uint64_t)(freeMem / 8 / sizeof(uint64_t)), N);
    if (chunk_elems < BLOCK_SIZE)
        chunk_elems = BLOCK_SIZE;

    cout << "Free GPU memory: " << (freeMem / 1024 / 1024) << " MB\n";
    cout << "Chunk size = " << chunk_elems << " elements ("
         << (chunk_elems * sizeof(uint64_t) / 1024 / 1024) << " MB)\n";

    uint64_t maxBlocks = (chunk_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int NUM_STREAMS = 1;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        cudaCheckError(cudaStreamCreate(&streams[s]));

    uint64_t *d_data[NUM_STREAMS];
    uint64_t *d_blockSums[NUM_STREAMS];
    uint64_t *d_blockOffsets[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        cudaCheckError(cudaMalloc((void **)&d_data[s], chunk_elems * sizeof(uint64_t)));
        cudaCheckError(cudaMalloc((void **)&d_blockSums[s], maxBlocks * sizeof(uint64_t)));
        cudaCheckError(cudaMalloc((void **)&d_blockOffsets[s], maxBlocks * sizeof(uint64_t)));
    }

    uint64_t *h_blockSums_pinned[NUM_STREAMS];
    uint64_t *h_blockOffsets_pinned[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        cudaCheckError(cudaHostAlloc((void **)&h_blockSums_pinned[s],
                                     maxBlocks * sizeof(uint64_t), cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc((void **)&h_blockOffsets_pinned[s],
                                     maxBlocks * sizeof(uint64_t), cudaHostAllocDefault));
    }

    uint64_t globalOffset = 0;
    uint64_t processed = 0;

    auto start_all = chrono::high_resolution_clock::now();
    float total_kernel_time = 0.0f, kernel_time = 0.0f;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    size_t chunkIdx = 0;
    while (processed < N)
    {
        uint64_t this_chunk = min(chunk_elems, N - processed);
        int s = chunkIdx % NUM_STREAMS;
        cudaStream_t stream = streams[s];

        uint64_t numBlocks = (this_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaCheckError(cudaMemcpyAsync(d_data[s], h_in + processed,
                                       this_chunk * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice, stream));
        kernel_time = 0.0f;
        cudaEventRecord(start);
        block_prefix_sum<<<(int)numBlocks, BLOCK_SIZE, 0, stream>>>(d_data[s], d_blockSums[s], this_chunk);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&kernel_time, start, end);
        total_kernel_time += kernel_time;

        cudaCheckError(cudaMemcpyAsync(h_blockSums_pinned[s], d_blockSums[s],
                                       numBlocks * sizeof(uint64_t),
                                       cudaMemcpyDeviceToHost, stream));

        cudaCheckError(cudaStreamSynchronize(stream));

        if (numBlocks > 0)
        {
            h_blockOffsets_pinned[s][0] = 0;
            for (size_t b = 1; b < numBlocks; ++b)
            {
                h_blockOffsets_pinned[s][b] = h_blockOffsets_pinned[s][b - 1] + h_blockSums_pinned[s][b - 1];
            }
        }

        cudaCheckError(cudaMemcpyAsync(d_blockOffsets[s], h_blockOffsets_pinned[s],
                                       numBlocks * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice, stream));

        kernel_time = 0.0f;
        cudaEventRecord(start);
        add_offsets<<<(int)numBlocks, BLOCK_SIZE, 0, stream>>>(
            d_data[s], d_blockOffsets[s], globalOffset, this_chunk);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&kernel_time, start, end);
        total_kernel_time += kernel_time;

        cudaCheckError(cudaMemcpyAsync(h_out_gpu + processed, d_data[s],
                                       this_chunk * sizeof(uint64_t),
                                       cudaMemcpyDeviceToHost, stream));

        cudaCheckError(cudaStreamSynchronize(stream));

        globalOffset = h_out_gpu[processed + this_chunk - 1];

        processed += this_chunk;
        chunkIdx++;
        cout << "Processed " << processed << " / " << N << "\r" << flush;
    }

    cout << "\nAll chunks processed.\n";
    auto end_all = chrono::high_resolution_clock::now();

    cout << "Total GPU wall time: "
         << chrono::duration<double, milli>(end_all - start_all).count() << " ms\n";
    cout << "Total Kernel time: " << total_kernel_time << "ms\n";

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        cudaCheckError(cudaFree(d_data[s]));
        cudaCheckError(cudaFree(d_blockSums[s]));
        cudaCheckError(cudaFree(d_blockOffsets[s]));
        cudaCheckError(cudaFreeHost(h_blockSums_pinned[s]));
        cudaCheckError(cudaFreeHost(h_blockOffsets_pinned[s]));
        cudaCheckError(cudaStreamDestroy(streams[s]));
    }
}

__host__ void uvm_sum(uint64_t *uvm_data)
{
    cout << "\n-----UVM SUM-----\n";
    int device;
    cudaCheckError(cudaGetDevice(&device));
    uint64_t numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint64_t *uvm_blockSums = nullptr;
    uint64_t *uvm_blockOffsets = nullptr;
    cudaCheckError(cudaMallocManaged(&uvm_blockSums, numBlocks * sizeof(uint64_t)));
    cudaCheckError(cudaMallocManaged(&uvm_blockOffsets, numBlocks * sizeof(uint64_t)));
    auto start_gpu = chrono::high_resolution_clock::now();

    // NO BENEFITS OBSERVED SO COMMENTED OUT
    // cudaCheckError(cudaMemAdvise(uvm_data, N * sizeof(uint64_t),
    //                              cudaMemAdviseSetPreferredLocation, device));

    // cudaCheckError(cudaMemAdvise(uvm_data, N * sizeof(uint64_t),
    //                              cudaMemAdviseSetAccessedBy, device));

    // cudaCheckError(cudaMemAdvise(uvm_blockSums, numBlocks * sizeof(uint64_t),
    //                              cudaMemAdviseSetPreferredLocation, device));

    // cudaCheckError(cudaMemAdvise(uvm_blockOffsets, numBlocks * sizeof(uint64_t),
    //                              cudaMemAdviseSetPreferredLocation, device));

    cudaCheckError(cudaMemPrefetchAsync(uvm_data, N * sizeof(uint64_t), device, 0));
    cudaCheckError(cudaMemPrefetchAsync(uvm_blockSums, numBlocks * sizeof(uint64_t), device, 0));
    cudaCheckError(cudaMemPrefetchAsync(uvm_blockOffsets, numBlocks * sizeof(uint64_t), device, 0));

    cudaCheckError(cudaDeviceSynchronize());

    float total_kernel_time = 0.0f, kernel_time = 0.0f;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    kernel_time = 0.0f;
    cudaEventRecord(start);
    block_prefix_sum<<<numBlocks, BLOCK_SIZE>>>(uvm_data, uvm_blockSums, N);
    cudaCheckError(cudaGetLastError());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel_time, start, end);
    total_kernel_time += kernel_time;
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemPrefetchAsync(uvm_blockSums, numBlocks * sizeof(uint64_t),
                                        cudaCpuDeviceId, 0));
    cudaCheckError(cudaMemPrefetchAsync(uvm_blockOffsets, numBlocks * sizeof(uint64_t),
                                        cudaCpuDeviceId, 0));
    cudaCheckError(cudaDeviceSynchronize());

    uvm_blockOffsets[0] = 0;
    for (uint64_t b = 1; b < numBlocks; b++)
    {
        uvm_blockOffsets[b] = uvm_blockOffsets[b - 1] + uvm_blockSums[b - 1];
    }

    cudaCheckError(cudaMemPrefetchAsync(uvm_blockOffsets, numBlocks * sizeof(uint64_t),
                                        device, 0));
    cudaCheckError(cudaDeviceSynchronize());

    kernel_time = 0.0f;
    cudaEventRecord(start);
    add_offsets<<<numBlocks, BLOCK_SIZE>>>(uvm_data, uvm_blockOffsets, 0, N);
    cudaCheckError(cudaGetLastError());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel_time, start, end);
    total_kernel_time += kernel_time;
    cudaCheckError(cudaDeviceSynchronize());

    auto end_gpu = chrono::high_resolution_clock::now();
    auto gpu_time = chrono::duration<double, milli>(end_gpu - start_gpu).count();

    cout << "Total GPU wall time: " << gpu_time << " ms\n";
    cout << "Total kernel time: " << total_kernel_time << " ms\n";

    cudaCheckError(cudaMemPrefetchAsync(uvm_data, N * sizeof(uint64_t),
                                        cudaCpuDeviceId, 0));
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaFree(uvm_blockSums));
    cudaCheckError(cudaFree(uvm_blockOffsets));
}

int main()
{
    cout << "Inclusive Prefix Sum for N = " << N << " elements.\n";

    uint64_t *h_in = nullptr;
    uint64_t *h_out_cpu = nullptr;
    uint64_t *h_out_gpu = nullptr;

    cudaCheckError(cudaHostAlloc((void **)&h_in, N * sizeof(uint64_t), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void **)&h_out_gpu, N * sizeof(uint64_t), cudaHostAllocDefault));

    h_out_cpu = new uint64_t[N];

    for (size_t i = 0; i < N; ++i)
        h_in[i] = 1;

    // CPU reference
    cout << "Computing CPU reference...\n";
    auto t0 = chrono::high_resolution_clock::now();
    inclusive_prefix_sum(h_in, h_out_cpu);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "CPU scan done, total time = "
         << chrono::duration<double, milli>(t1 - t0).count()
         << " ms\n";

    cte_sum(h_in, h_out_gpu);

    cout << "Verifying results...\n";
    check_result(h_out_cpu, h_out_gpu, N);

    uint64_t *uvm_data = nullptr;
    cudaCheckError(cudaMallocManaged(&uvm_data, N * sizeof(uint64_t)));
    for (size_t i = 0; i < N; ++i)
        uvm_data[i] = 1;
    uvm_sum(uvm_data);

    cout << "\nVerifying results...\n";
    check_result(h_out_cpu, uvm_data, N);

    cudaCheckError(cudaFree(uvm_data));
    cudaCheckError(cudaFreeHost(h_in));
    cudaCheckError(cudaFreeHost(h_out_gpu));
    delete[] h_out_cpu;

    return 0;
}
