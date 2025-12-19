# High-Dimensional Grid Search Optimization 

## 1. Loop Transformation 

### Applied Optimizations

1. **I/O Buffering Attempted but Not Fully Practical**  
   Initially, all output values were accumulated in a dynamically allocated memory buffer to reduce repeated `fprintf` calls inside the innermost loop. While this approach allowed the computation to complete faster (72s), the resulting output file could not be opened afterwards. In contrast, writing directly using `fprintf` in each iteration produced a file that opened correctly, though at slightly higher execution time (77s). Thus, buffering was not practical due to the huge memory requirement and file-write issues.

2. **Loop Invariant Code Motion (LICM)**  
   Several computations inside nested loops were invariant across one or more loop levels. Such expressions were hoisted to the appropriate outer loops. Example: Terms like `c11 * x1 - d1` were computed once per x1-loop instead of recomputing at deeper levels. This reduced redundant floating-point operations, improving arithmetic efficiency and cache utilization.

3. **Strength Reduction**  
   Multiplicative expressions involving loop indices were replaced by equivalent additions. This reduced the computational cost of repeated multiplications. For example:  base = i * stride, was replaced with incremental updates:  base += stride which avoids repeated multiplication in every iteration.

4. **Short-Circuit Constraint Evaluation**  
Each constraint check was restructured to evaluate sequentially with early exits. If any constraint exceeded its threshold (`if (qk > ek) continue;`) the remaining constraints for that iteration were skipped. This avoided unnecessary computations and reduced the average number of checks per loop iteration.

5. **Incremental Constraint Computation**  
Constraint equations were computed incrementally using precomputed partial sums (`p1_k = p1_{k-1} + c1k * xk`) instead of recomputing full expressions at every level. This eliminated redundant arithmetic operations and improved cache efficiency.

---

### Performance Results

**Table: Execution times of original vs modified in seconds**

| Version                         | Execution Time (seconds) |
|---------------------------------|--------------------------|
| Original (Baseline)             | 110                      |
| All feasible optimizations applied | 77                   |

---

### Observations and Analysis

- Using LICM, strength reduction, incremental updates, and short-circuit evaluation (without buffering) provided the best practical performance, reducing execution time from 110s to 77s.
- Changing the function prototype from multiple scalar parameters to array-based arguments slightly degraded performance, likely due to additional pointer dereferencing and reduced opportunities for compile-time optimization and inlining.
- The compiler was already auto-vectorizing some parts of the innermost loop; adding `#pragma omp simd` did not yield further improvement.
- Since there are no data dependencies in the loop, loop transformations such as interchange provided no additional performance benefit.
- Unrolling the innermost loop by 2 times actually increased the execution time to nearly 100 seconds, likely due to instruction cache pressure and increased overhead in the innermost loop.

---

## 2. OpenMP

The sequential version of the code was parallelized using OpenMP to leverage multi-core execution. A `#pragma omp parallel for ordered schedule(dynamic) reduction(+:pnts)` directive was applied over the outer loops. All loop variables and temporary constraint variables (`x1--x10`, `q1--q10`) were declared as private, while the point counter (`pnts`) was accumulated using a reduction clause to ensure correctness. Output operations were placed inside an `omp ordered` section to maintain deterministic and consistent printing order across threads when needed. Loop collapsing of the outermost 4 loops (`collapse(4)`) with `ordered(4)` reduces scheduling overhead and improves load balancing, resulting in better performance.

The submitted program corresponds to the version with ordered output and not the unordered variant.

---

### Performance Results

**Table 5: Execution times of sequential and OpenMP versions**

| Version                                         | Time (seconds) |
|-------------------------------------------------|----------------|
| Original (Baseline)                             | 110.0          |
| Optimized Sequential Version                   | 77.0           |
| OpenMP Parallel Version (ordered, no collapse) | 15.0           |
| OpenMP Parallel Version (ordered, collapse(4)) | 10.0           |
| OpenMP Parallel Version (unordered, collapse(9)) | 8.0          |

---

### Observations and Analysis

- Parallelization achieved a speedup of ∼7–8× over the optimized sequential version when using loop collapsing and dynamic scheduling.
- Using ordered without collapsing gave 15s runtime, whereas collapsing the outermost 4 loops with `ordered(4)` reduced runtime to 10s.
- Dynamic scheduling consistently performed better than guided, reducing runtime by approximately 2–5s compared to guided scheduling for the same loop structure.
- The scalability with ordered is primarily limited by serialization of printing operations. Excluding file I/O from the parallel region or buffering outputs per thread could further improve speedup.
- Collapsing loops reduces scheduling overhead and improves load balancing for irregular iteration workloads caused by early exits from constraint checks.

