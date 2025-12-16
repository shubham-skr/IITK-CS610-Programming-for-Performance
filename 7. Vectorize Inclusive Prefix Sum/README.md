# Optimizing Prefix Sum with SIMD (SSE4/AVX2) and OpenMP

## Implementations

1.  **Sequential (Reference) Version:** A straightforward scalar implementation that accumulates sums sequentially.
2.  **OpenMP Version:** Utilizes `#pragma omp simd` with inclusive scan reduction to exploit vectorization and parallelism.
3.  **SSE4 Version:** Uses 128-bit SIMD vectors to compute the prefix sum for 4 integers at a time via tree reduction.
4.  **AVX2 Version:** Extends the idea of SSE4 to 256-bit vectors, processing 8 integers at a time with shifts, adds, and lane propagation.

---

## Performance Results

Execution times of inclusive prefix sum implementations (in microseconds).

| Implementation | Time (us) |
| :--- | :--- |
| Serial Version | 105 |
| OpenMP Version | 106 |
| SSE4 Version | 11 |
| AVX2 Version | 17 |

---

## Observations and Analysis

* The **SSE4** version achieves the fastest execution time due to 128-bit vectorized processing and efficient tree reduction.
* The **AVX2** version is slightly slower than SSE4 for this specific input size because of the lane extraction and broadcast overheads in AVX2, despite processing 8 integers per vector.
* For **smaller** problem sizes ($N = 1 \ll 16$, or $2^{16}$), the **OpenMP** version was inconsistent, 60% of time taking longer than the sequential version ($\approx +10$ us), and occasionally slightly faster ($\approx -2$ us), showing that parallelization overhead dominates for small loops.
    <br>For **larger** problem sizes ($N = 1 \ll 24$, or $2^{24}$), **OpenMP** consistently outperformed the sequential version, reducing execution time from 18000 us to 16000 us, though still slightly slower than the fully vectorized versions. OpenMP was slower than the SSE4 and AVX2 vectorized versions, which consistently took 8200 us and 8500 us, respectively.
* Overall, explicit **SIMD** via **SSE4** and **AVX2** provides a **6–10x** speedup over the scalar sequential implementation.
