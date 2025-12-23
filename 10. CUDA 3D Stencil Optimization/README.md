# CUDA 7-Point 3D Stencil Optimization

## Problem Description
7-point 3D stencil computation on a 64×64×64 grid.

## Implementation

1. **Naive Kernel:** The naive version serves as the baseline implementation. It performs  
   convolution using only global memory without any data reuse optimization. Each thread  
   computes the convolution for a single grid point by directly reading all required neighbor-  
   ing values from global memory.

2. **Shared Memory Kernel:** The shared-memory version introduces a 3D tiling opti-  
   mization to reduce redundant global memory reads. A 3D tile of size (TILE SIZE + 2)³  
   is allocated in shared memory to store both the current block and its neighboring cells.

3. **Optimized Shared Memory Kernel:** The optimized version extends the shared-  
   memory approach with two major enhancements:  
   • **Index Permutation:** In earlier versions, the thread mapping followed the order  
     (i, j, k) corresponding to (x, y, z) dimensions. This mapping was modified to (k, j, i),  
     effectively interchanging the axes to make the k-dimension the fastest varying index.  
   • **Loop Unrolling by Two:** Each thread is designed to compute two consecutive  
     output points along the k-dimension (i.e., two slices per thread).

4. **Pinned Memory Version:** The pinned-memory version uses the same optimized kernel  
   as above but replaces pageable host memory with page-locked (pinned) host memory using  
   `cudaHostAlloc`. The kernel execution time, Host-to-Device (H2D) copy time, and D2H  
   copy time were measured separately to highlight transfer-time improvements over the  
   standard pageable memory implementation.

---

## Results

**Table 1: Execution time comparison for different CUDA kernel implementations (N = 512).**

| Implementation                     | H2D Copy (ms) | Kernel Time (ms) | D2H Copy (ms) |
|-----------------------------------|---------------|------------------|---------------|
| CPU Baseline                      | –             | 340.00           | –             |
| Naive Kernel (Global Memory)      | –             | 20.52            | –             |
| Shared Memory (Tile = 8)          | –             | 18.12            | –             |
| Optimized Kernel (Pageable)       | 242.65        | 5.31             | 251.42        |
| Pinned Memory (Page-Locked)       | 131.43        | 5.34             | 135.04        |

---

## Observations and Analysis

• The baseline CPU implementation took approximately 340 ms, whereas the naive GPU  
  kernel achieved a 16.6× speedup purely due to parallel execution on the GPU cores. Sub-  
  sequent optimizations improved both computational throughput and memory efficiency,  
  with the shared-memory and optimized kernels achieving substantial performance gains.

• The naive implementation directly accessed global memory for all neighboring elements,  
  leading to redundant memory transactions. By introducing shared memory tiling in the  
  second version, redundant global reads were eliminated within a thread block. Each tile  
  was loaded cooperatively into on-chip shared memory, allowing data reuse across multiple  
  threads. This reduced global memory bandwidth pressure and led to a 10–12% reduction  
  in kernel time (from 20.52 ms to 18.12 ms), and other tile sizes were also inefficient.

• For shared memory kernel, the kernel was tested on varying tile sizes - {1, 2, 4, 8}. The  
  best performance was found in tile size = 8. For tile size = 8, the kernel time was 18.12 ms,  
  as compared to the kernel time of 23.89 ms on tile size = 4.

• The optimized kernel achieved the largest reduction in compute time. Reordering the  
  mapping from (i, j, k) to (k, j, i) allowed threads in a warp to access contiguous memory  
  locations, improving memory coalescing and reducing transaction overhead. Each thread  
  computed two consecutive k-indices, increasing arithmetic intensity and instruction-level  
  parallelism. Together, these optimizations results in a nearly 3.5× speedup over the  
  shared-memory version.

• The pinned-memory version used page-locked host buffers for host-to-device (H2D) and  
  device-to-host (D2H) transfers. Pinned memory allows the GPU’s DMA engine to access  
  host memory directly, enabling higher PCIe bandwidth utilization. This reduced the  
  transfer times significantly:  
  – H2D copy time reduced from 242.65 ms (pageable) to 131.43 ms.  
  – D2H copy time reduced from 251.42 ms (pageable) to 135.04 ms.  

  The kernel execution time remained nearly identical (around 5.3 ms) since computation  
  is unaffected by host memory type. Though, only CPU stencil computation saw a rise  
  in time from 340 ms to 450 ms. However, the overall runtime for the device improved by  
  approximately 45% due to faster data transfer.
