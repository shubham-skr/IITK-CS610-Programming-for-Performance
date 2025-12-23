# Parallelization of a 10D Grid Search Computation using CUDA

## Implementations

1. **Vanilla CUDA Port:**  
The first implementation is a direct CUDA port of the provided CPU code. The primary objective of this version is correctness, with no optimizations applied.

2. **Performance-Optimized CUDA Port:**  
This version improves on the vanilla port by applying several GPU-specific optimizations: placing invariant coefficients in constant memory, using block-local shared memory for frequently-read grid parameters, processing the large 10-D iteration space in chunks, and recording candidate indices to allow host-side ordering.

### High-level flow

(a) Read `disp.txt` (120 coefficients) and `grid.txt` (30 grid parameters) on the host.  
(b) Copy the 120 coefficients into device constant memory (`d_a_const`) and the 30 grid parameters to device global memory.  
(c) Compute the total number of grid points: `total = s1 · s2 · · · s10`.  
(d) Iterate over the 1D iteration-space in host-controlled chunks: launch the optimized kernel for each chunk, then collect results.  
(e) Copy back GPU results (point values and indices), sort on CPU by the original linear index to reproduce the original ordering, and write output.

### Key optimizations

• Coefficients in constant memory  
• Shared-memory copy of grid parameters  
• Chunking the global 10-D space. The host loops over the flattened index range in fixed-size chunks (`CHUNK_SIZE`), launching a kernel for each chunk.  
• Early exit and minimal per-thread work  
• Atomic accumulation of results (unordered) + indices  
• Host-side sorting by index  

3. **Unified Memory (UVM) Implementation with Memory Advises and Prefetching:**  
The third version re-implemented the 10D grid–search algorithm using Unified Memory (UVM) via `cudaMallocManaged()` along with `cudaMemAdvise()` and `cudaMemPrefetchAsync()`. Compared to the previous versions, this design removes all explicit H2D/D2H copies and allows the CUDA driver to manage data movement on demand.

4. **Thrust-Based Implementation:**  
In this version, the grid-search computation is parallelized using the Thrust library. This implementation replaces the original 10-level nested grid-search loops with a fully parallel, high-level CUDA approach by using a counting iterator to represent the entire search space and `thrust::copy_if` to filter only those grid points that satisfy all ten constraints. The loop bounds and coefficients from `disp.txt` and `grid.txt` are moved to constant memory for fast read-only access, and a custom predicate functor reconstructs the corresponding `(x1 … x10)` coordinates for each global index and evaluates the constraints exactly as in the CPU version.

---

## Results

**Table 3: Kernel execution times for the three implementations.**

| Version | Kernel Time (s) |
|--------|----------------|
| (i) Vanilla CUDA Port | 90.84 |
| (ii) Optimized CUDA Version | 26.28 |
| (iii) UVM Version (with MemAdvise + Prefetch) | 26.25 |
| (iv) Thrust Implementation | 38.60 |

---

## Observations and Analysis

• The vanilla GPU implementation (Part 1) maps each point in the 10D grid space to a unique CUDA thread. Although this version is functionally correct, its performance is significantly limited due to repeated loads of the 120 constraint coefficients from global memory, repeated loads of grid parameters for every thread, no caching or reuse of shared constants and others, leading to high kernel time.

• The optimized version (Part 2) reduces execution time to 26.28 s, a speedup of about 3.45× over the vanilla port. This comes from several targeted GPU optimizations: use of constant memory, shared-memory caching, and chunked execution, which improves scheduling efficiency.

• The UVM version (Part 3) achieves a nearly identical kernel runtime. With both memadvise and prefetching applied, UVM nearly eliminates paging overhead, thus matching the performance of the fully optimized explicit–copy version.

• However, the performance of the Thrust implementation is slower than the manually optimized CUDA versions because `copy_if` internally performs multiple passes (flagging, prefix-scan, scattering) and introduces overhead related to iterator abstraction and large temporary allocations. This implementation using the Thrust approach cannot exploit low-level kernel fusion, warp-level optimizations, or shared-memory usage, resulting in performance that sits between the naive and optimized CUDA implementations, though a more advanced Thrust-based implementation could potentially improve this and remains an avenue for future exploration.
