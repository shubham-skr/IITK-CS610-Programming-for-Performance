# Optimize Stencil Computation

## Transformations Applied

The following transformations were incrementally applied:

1.  **Scalar baseline kernel:** Original kernel without any transformation.
2.  **Precomputation:** Precompute loop indices to reduce repeated multiplications inside the innermost loops.
3.  **Tiling (Blocking):** Exploit cache locality by blocking the i, j, and k loops.
4.  **Unrolling:** Manually unroll innermost loop by factor of 4.
5.  **OpenMP parallelization:** Parallelize outer loops with `#pragma omp parallel for collapse(n)`.
6.  **Compound transformations:** Combinations such as precompute + tiling + unrolling, OpenMP + tiling + unrolling, and OpenMP + precompute + tiling + unrolling.
7.  **Vectorization:** Scalar kernel was not being vectorized automatically by the compiler. By placing `#pragma omp simd` immediately above the innermost k loop, the scalar kernel achieved significant vectorization.

---

## Performance Results

**Table 1:** Execution times of different kernel transformations (in milliseconds).

| Kernel | Time (ms) |
| :--- | ---: |
| Scalar | 24 |
| Precomputation | 20 |
| Tiled | 20 |
| Unrolled | 19 |
| OpenMP | 8 |
| Tiling + Unrolled | 20 |
| Precompute + Tiling + Unrolled | 18 |
| OpenMP + Tiling + Unrolled | 8 |
| OpenMP + Precompute + Tiling + Unrolled | 6 |
| Scalar + Vectorization | 11 |

---

## Observations and Analysis

* **Precomputation** reduces repeated multiplications and slightly improves the scalar kernel performance (from 24 ms to 20 ms).
* **Tiling** improves cache locality. Combined with unrolling, the kernel time decreased from 24 ms (scalar) to 18–20 ms.
* **Loop unrolling** provides a small performance boost due to reduced loop overhead and better instruction-level parallelism.
* **OpenMP parallelization** gives the largest speedup, reducing kernel time to 6–8 ms for compound transformations.
* **Vectorization** of the scalar kernel with `#pragma omp simd` significantly improves its performance (24 ms &rarr; 12 ms) without any other transformation.
* **Optimal block size:** For compound transformations, 16 gave best results. For individual tiling without other optimizations, 64 was slightly better (1–3 ms difference).
