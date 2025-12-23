# Optimized 2D and 3D Convolution using CUDA

## Implementation

1. **2D Convolution – Basic 2D Kernel:**  
It performs convolution using only global memory without any form of caching or tiling. Each thread computes the average value for a single grid point by directly reading all neighboring elements from global memory.

2. **2D Convolution – Optimized 2D Kernel:**  
The optimized 2D kernel introduces several GPU memory hierarchy optimizations to improve performance:

• **Shared Memory Tiling:** Each thread block loads a tile of the input grid, including halo regions, into shared memory.  
• **Constant Memory for Filter:** The convolution filter coefficients are stored in constant memory.  
• **Loop Unrolling:** The convolution loops over filter dimensions are unrolled.  
• **Boundary Handling via Halos:** Threads cooperatively load valid input data into shared memory, substituting zeros for out-of-bound indices.

3. **3D Convolution – Basic 3D Kernel:**  
The basic 3D convolution kernel extends the naive 2D approach to three dimensions. The kernel employs three nested loops corresponding to the x, y, and z filter dimensions and averages the valid neighboring values.

4. **3D Convolution – Optimized 3D Kernel:**  
The optimized 3D kernel enhances the basic version by leveraging CUDA’s shared and constant memory for improved memory reuse and computational efficiency:

• **3D Shared Memory Tiling:** Each block of threads loads a 3D tile (including halos) from global memory into shared memory.  
• **Constant Memory for Filter Storage:** The 3D convolution filter is stored in device constant memory.  
• **Loop Unrolling:** Inner loops over the filter dimensions are unrolled.  
• **Cooperative Data Loading and Synchronization:** Threads within a block cooperatively load the full shared tile, using `syncthreads()`.

---

## Results

**Table 4: Execution time and speedup comparison of basic and optimized convolution kernels where N = 1024.**

| Kernel | Basic Time (ms) | Optimized Time (ms) |
|-------|-----------------|--------------------|
| 2D Convolution | 1.7178 | 0.0839 |
| 3D Convolution | 77.797 | 63.293 |

---

## Observations and Analysis

• The optimized 2D convolution kernel achieved a significant speedup of approximately 20.46× over the naive version. This substantial improvement is primarily due to the effective use of shared memory tiling, which reduces redundant global memory accesses by allowing threads within a block to reuse neighboring data elements. Additionally, the use of constant memory for filter coefficients and loop unrolling further improved instruction throughput.

• For the 3D convolution kernel, the optimized version achieved a speedup of approximately 1.23×. This is primarily attributed to the much larger memory footprint of 3D data and shared-memory tiles, which increase shared-memory pressure and require more extensive memory accesses per thread, reducing the relative benefit of caching and tiling.

• No numerical differences were observed between the outputs of the basic and optimized kernels for either 2D or 3D cases, confirming functional correctness of the optimized implementations.
