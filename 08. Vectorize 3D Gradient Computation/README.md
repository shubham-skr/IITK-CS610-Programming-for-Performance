# Vectorization of the 3D Gradient Kernel Using SSE4 and AVX2

## Problem Description

This problem focuses on optimizing a 3D gradient kernel by vectorizing a scalar implementation using SIMD intrinsics. The goal is to implement and compare three versions of the kernel—Scalar, SSE4, and AVX2—and analyze the performance improvements achieved through vectorization.

The gradient is computed along the *i*-dimension as:

\[
B[i, j, k] = A[i + 1, j, k] - A[i - 1, j, k]
\]

## Implementations

### Scalar Version
A straightforward 3D gradient kernel that computes the difference along the *i*-dimension. No vectorization or parallelism is applied.

### SSE4 Version
Uses 128-bit SIMD registers (`__m128i`) to compute differences of two 64-bit integers at a time. Loads and stores are aligned, and subtraction is performed using `_mm_sub_epi64`.

### AVX2 Version
Uses 256-bit SIMD registers (`__m256i`) to compute differences of four 64-bit integers at a time. Loads and stores are unaligned for safety, and subtraction is performed using `_mm256_sub_epi64`.

## Performance Results

**Table 3: Execution times of scalar, SSE4, and AVX2 3D gradient kernels**

| Implementation | Time (ms) |
|----------------|-----------|
| Scalar Version | 106       |
| SSE4 Version   | 80        |
| AVX2 Version   | 74        |

## Observations and Speedups

- The SSE4 version achieves a speedup of approximately 1.33 times over the scalar implementation.

- The AVX2 version achieves a speedup of approximately 1.43 times over the scalar implementation.

- AVX2 outperforms SSE4 because it can process four 64-bit integers per vector instead of two.

- The vectorized versions benefit from SIMD parallelism along the innermost dimension (*k*) of the array, improving throughput without changing the underlying algorithm.
