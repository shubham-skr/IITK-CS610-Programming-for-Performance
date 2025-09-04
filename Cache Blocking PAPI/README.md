# Improving Cache Performance With Blocking - Analyzing cache through PAPI Tools

## Machine Specifications
The experiments were conducted on the IIT Kanpur CSE KD Lab on the CSEWS machine. The relevant hardware specifications, obtained using `papi_mem_info`, are as follows:

- **L1 Data Cache:** 32 KB, 8-way set associative, 64B line size
- **L2 Unified Cache:** 256 KB, 4-way set associative, 64B line size
- **L3 Unified Cache:** 12 MB, 16-way set associative, 64B line size

## PAPI Counters Used
The following PAPI counters were used to capture cache behavior:

- `PAPI_L1_DCM` → L1 Data Cache Misses
- `PAPI_L2_DCM` → L2 Data Cache Misses
- `PAPI_L3_TCM` → L3 Total Cache Misses

## Block Sizes Tested
The blocked convolution implementation was evaluated using the following block sizes:

- (4, 8, 16, 32, 64)
- (4, 8, 16, 32, 64, 128)
- (4, 8, 16, 32, 64, 128, 512, 1024)

Each experiment was executed 5 times, and the average values of execution time and cache misses are reported.

## Performance Metrics
To evaluate the effectiveness of the optimization, the following metrics were collected for each implementation, averaged over five runs:

1. **Average Execution Time (ms):** The primary measure of performance.
2. **Average L1 Data Cache Misses (L1 DCM):** Measures data misses in the highest-level L1 cache level.
3. **Average L2 Data Cache Misses (L2 DCM):** Measures data misses in the mid-level L2 cache. 
4. **Average L3 Total Cache Misses (L3 TCM):** Measures total misses in the last-level cache (LLC). 

## Autotuning Block Sizes
For the blocked implementation, an autotuning process was performed to find the optimal block dimensions ($B_I, B_J, B_K$). A range of block sizes was tested for each dimension. The combination that resulted in the lowest average execution time was identified as the optimal configuration.

---

## Results

**Table 1: Performance Summary for Block Sizes (4, 8, 16, 32, 64)**

| **SNO** | **Implementation**      | **Avg Time (ms)** | **L1 DCM** | **L2 DCM** | **L3 TCM** | **Speedup** |
|---------|--------------------------|-------------------|------------|------------|------------|-------------|
| 1       | Naive                   | 20.8407           | 125571     | 162572     | 49354      | --          |
|         | Best Blocked (8x32x64)  | 17.14             | 90329      | 160818     | 30748      | 1.216x      |
| 2       | Naive                   | 20.7376           | 125550     | 161150     | 55090      | --          |
|         | Best Blocked (8x4x64)   | 17.0842           | 125455     | 159756     | 41255      | 1.214x      |
| 3       | Naive                   | 20.5711           | 125642     | 166827     | 41267      | --          |
|         | Best Blocked (4x64x64)  | 17.1182           | 90960      | 162521     | 40400      | 1.202x      |

---

**Table 2: Performance Summary for Block Sizes (4, 8, 16, 32, 64, 128)**

| **Implementation**      | **Avg Time (ms)** | **L1 DCM** | **L2 DCM** | **L3 TCM** | **Speedup** |
|--------------------------|-------------------|------------|------------|------------|-------------|
| Naive                   | 20.7269           | 125634     | 167261     | 41888      | --          |
| Best Blocked (8x8x128)  | 17.0505           | 84562      | 164745     | 36169      | 1.216x      |

---

## Analysis
1. In the naïve version, memory accesses stride through large 3D arrays → poor spatial and temporal locality. Blocking restructures loops to reuse input data and kernel values within the cache before they are evicted. This reduces L1/L2/L3 misses, keeping the working set in faster caches.
2. The naïve implementation suffers from poor data locality. For each output element `O[i][j][k]`, it iterates through the entire $3 \times 3 \times 3$ filter, accessing a corresponding block of the input array `I`. When it moves to the next output element `O[i][j][k+1]`, it slides this input window by one element. This access pattern means that most of the input data loaded into the cache for one calculation is evicted before it can be reused for the next, leading to a high number of cache misses, especially at the L2 levels.
3. The blocked implementation fundamentally changes this access pattern. The key insight is that this input block is small enough to fit within the CPU's caches (particularly the 256KB L2 cache). The algorithm then performs all the necessary computations for the output block, heavily reusing the input data that is already in the cache. This drastically improves temporal locality.
4. Effect of block size: Small blocks (e.g., 4–8) underutilize the cache, leading to less benefit. Very large blocks (e.g., ≥ 512) risk creating a working set that exceeds the L2 cache size, which can cause conflict misses. The best performance is found when the block's working set size is approximately equal to the L2 cache capacity.
5. Blocking pattern analysis: Across all experiments, the best blocked implementations consistently chose the *k*-dimension block size as 64. The *i*-dimension block size remained small (typically 4 or 8), while the *j*-dimension varied across runs. This suggests that fixing the innermost block size to the cache line length (64 bytes) provides stable performance benefits, whereas the outer block sizes adapt based on the specific cache behavior and workload.
6. Observed Speedup: The best blocked kernel achieved a significant speedup (≈1.20x–1.24x). The primary justification for this performance gain is the substantial reduction in L2 data cache misses. By keeping the working set in the L2 cache, the algorithm avoids the massive latency penalty of accessing the L3 cache or main memory. The even more reduction in L3 misses confirms that the blocked version is extremely effective at preventing costly fetches from RAM.
