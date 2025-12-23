# Inclusive Prefix Sum on Large Inputs using CUDA

## Implementation

• Two distinct CUDA implementations were developed:  
1. A Copy-Then-Execute (CTE) model using explicit chunk-based memory management without Unified Virtual Memory (UVM).  
2. A UVM-based model using `cudaMallocManaged()` along with `cudaMemAdvise()` and `cudaMemPrefetchAsync()`.

Both versions implement the Blelloch parallel scan algorithm for block-level computation, and then combine block results to form the global inclusive prefix sum.

• The algorithm performs the scan in two hierarchical levels:  
– **Intra-block scan:** Each CUDA block performs an inclusive scan using shared memory via an up-sweep and down-sweep phase, following Blelloch’s method.  
– **Inter-block offset addition:** Each block’s total sum is stored in a global array (`blockSums`), which is later scanned on the CPU to compute offsets for subsequent blocks. A second kernel (`add offsets()`) then applies these offsets to obtain the final global prefix sums.

• **Copy-Then-Execute (CTE) Implementation**  
The CTE model explicitly handles data transfers between host and device. Since the input size can exceed GPU memory, it divides the input into manageable chunks determined dynamically based on available GPU memory (`cudaMemGetInfo()`). Each chunk is processed sequentially using the following pipeline:

1. Copy the current chunk from host to GPU using `cudaMemcpyAsync()`.  
2. Launch `block prefix sum()` to compute intra-block scans.  
3. Copy `blockSums[]` back to the host and perform a CPU-based prefix sum to compute inter-block offsets.  
4. Copy the offsets back to the GPU and launch `add offsets()` to finalize the scan.  
5. Copy the completed chunk back to host memory.

Pinned (page-locked) host memory was used for all transfer buffers (`cudaHostAlloc()`).

The code implemented streams but only one CUDA stream was used in this implementation because the algorithm process chunks sequentially and the current chunk output is dependent on previous chunk, so having multiple streams fails here and produce no performance benefit.

• **Unified Virtual Memory (UVM) Implementation**  
The second implementation uses `cudaMallocManaged()` to allocate a single unified memory region shared between CPU and GPU. This eliminates explicit copies, with the CUDA driver automatically managing data migration between host and device. To minimize page fault overhead and control data placement, several UVM optimization strategies were used:

– **Preferred location:** `cudaMemAdviseSetPreferredLocation` was applied to keep large data arrays resident on the GPU.  
– **Access hints:** `cudaMemAdviseSetAccessedBy` was used to ensure the data remained accessible from both CPU and GPU without unnecessary migration.  
– **Prefetching:** `cudaMemPrefetchAsync()` was used to proactively migrate data to the GPU before kernel launch and back to CPU memory after computation for verification.

The computation pipeline remains the same:

1. Block-wise scan using `block prefix sum()`.  
2. CPU prefix sum on `blockSums` to compute offsets.  
3. Offset addition via `add offsets()`.

• **Timing and Performance Measurement:**  
Execution times were measured at multiple levels for accurate analysis:

– **CPU timing:** The sequential prefix sum was timed using the C++ `std::chrono::high resolution clock`, providing the baseline performance.  
– **Total GPU wall time:** Each GPU-based implementation (CTE and UVM) measured total end-to-end execution using `std::chrono::high resolution clock`, capturing data transfer, kernel launches, and synchronization overheads.  
– **Kernel execution time:** Individual kernel executions (`block prefix sum()` and `add offsets()`) were measured using CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`).

• In the CTE version, total kernel time was accumulated across all chunks: 
total kernel time =
Σ_{i=1}^{nchunks} (time(i)_block prefix + time(i)_add offsets)

while the total wall-clock time included memory transfer and CPU offset computation.

• In the UVM version, two kernel events were recorded (for intra-block and offset-add phases), and total time included the impact of prefetching and data migration.

---

## Results

All experiments were repeated for both a small input size (N = 2¹⁵ = 32768) that fully fits in GPU memory, and a very large input size (N = 2³⁰ = 1,073,741,824), where the elements were of type `uint64_t` (changed from given `uint32_t`), that significantly oversubscribes the GPU memory. Measured Unified Virtual Memory (UVM) model progressively with no optimizations, `cudaMemAdvise()`, prefetching, and the combination of both.

**Table 2: Performance comparison of CTE and UVM prefix sum implementations for small and large input sizes.**

| Configuration                  | N    | Total GPU Time (ms) | Kernel Time (ms) |
|--------------------------------|------|---------------------|------------------|
| CPU Baseline                   | 2¹⁵  | 0.212               | –                |
| CTE Model                      | 2¹⁵  | 0.301               | 0.140            |
| UVM (no optimizations)         | 2¹⁵  | 0.925               | 0.821            |
| CPU Baseline                   | 2³⁰  | 6801.13             | –                |
| CTE Model                      | 2³⁰  | 2341.27             | 110.52           |
| UVM (no optimizations)         | 2³⁰  | 1997.22             | 1983.61          |
| UVM + cudaMemAdvise            | 2³⁰  | 2278.29             | 1851.93          |
| UVM + Prefetching              | 2³⁰  | 867.73              | 103.25           |
| UVM + MemAdvise + Prefetch     | 2³⁰  | 1216.20             | 103.28           |

---

## Observations and Analysis

• **Small Input Size (N = 2¹⁵):**  
For inputs that comfortably fit within GPU memory, the Copy–Then–Execute (CTE) model achieves the best overall performance. It completes the scan in only ≈ 0.30 ms, slightly faster than the UVM version (≈ 0.93 ms). The kernel times for both versions are similar, but the UVM configuration incurs significant page-fault-driven migration overhead, which dominates the total execution time.

• **Oversubscribed Input Size (N = 2³⁰):**  
For large inputs that exceed GPU memory capacity, the behavior changes drastically.

• **CTE Model:**  
The CTE implementation processes the input in chunks determined by available GPU memory. Although the kernel itself is fast (110 ms total), the chunking pipeline requires: multiple H2D and D2H transfers, CPU-based prefix sum over block sums, synchronization after every chunk, leading to a total execution time of 2341 ms. Even though CTE avoids page faults, the repeated PCIe transfers become the bottleneck.

• **UVM Without Optimizations:**  
UVM without any hints performs poorly for large inputs. The total kernel time (1983 ms) nearly equals the total program time (1997 ms), showing that the computation is dominated by on-demand page migration (page faults).

• **Effect of Individual UVM Optimizations:**  
Setting preferred location and access hints reduces unnecessary migrations, but page faults still occur. Enabling only `cudaMemPrefetchAsync()` dramatically improves performance: the kernel time drops to 103 ms (same as CTE), demonstrating that prefetching moves data to the GPU in bulk. Prefetching provides the single largest performance boost, making UVM faster than CTE in oversubscription scenarios. Adding MemAdvise to prefetching does not improve performance further and in fact introduces small management overheads.




