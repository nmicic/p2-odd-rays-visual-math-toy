# Theta CUDA - File Descriptions

**Version**: v1.2
**Date**: November 2025

> **Warning / Disclaimer**
>
> This repository contains an AI-assisted exploratory visualization project.
> It is educational and experimental, intended for curiosity, illustration, and math-art.
> It is not peer-reviewed, not mathematical research, and not presented as a theorem or verified result.
>
> No practical evaluation, performance assessment, or engineering analysis has been performed.
> Nothing in this repository should be interpreted as providing system behavior, optimization, or functional guarantees.
>
> Visual patterns produced by this code are artifacts of the construction, not conclusions or claims.
>
> This project is archived for transparency and exploration only.

---

## What This Is

A toy CUDA program that runs the theta_key integer mapping on a GPU. This is purely for exploration and visualization of the 2-adic decomposition—not for any engineering purpose.

**This is NOT a benchmark.** No performance claims, comparisons, or evaluations are made.

---

## Files

| File | Description |
|------|-------------|
| `theta_cuda_benchmark_v1.2.cu` | Toy CUDA code (not for production) |
| `theta_cuda_v1.2.cuh` | Experimental header (not for production) |
| `THETA_SPEC_v1.2.md` | Mathematical definitions only |
| `../python/theta_toolkit.py` | CPU toy implementation |

---

## How to Run (Toy Only)

### Compile

```bash
nvcc -O3 -arch=sm_89 theta_cuda_benchmark_v1.2.cu -o theta_bench
```

### Run

```bash
./theta_bench 10 64 1 0
```

> **Warning**: The output is for visualization/exploration only. Do not interpret any
> numbers as performance metrics, benchmarks, or quality assessments.

---

## Understanding Theta (Mathematical Definition Only)

Every positive integer `n` decomposes uniquely:
```
n = 2^v2(n) × core(n)
```

Where:
- `v2(n)` = count of trailing zeros (2-adic valuation)
- `core(n)` = odd part after removing all factors of 2

**theta_key** = bit-reverse of the odd core

### Example
```
n = 12 = 0b1100
v2(12) = 2 (two trailing zeros)
core(12) = 3 = 0b11
theta_key(12) = bit_reverse(0b11, 2) = 0b11 = 3
```

This is a mathematical definition only. No claims are made about its utility.

---

## Disclaimer

This is a **toy hobby project**. It is **not research** and **not peer-reviewed**.

**Not for any practical use:**
- Not for benchmarking — output numbers are not validated metrics
- Not cryptographic — deterministic, trivially reversible
- Not for security — do not use to protect secrets
- Not for hashing — do not use for routing, load balancing, or operational systems
- Not for production — use established, vetted libraries for real work
- Not validated — no claims of correctness, uniformity, or fitness for any purpose

---

## See Also

- [`THETA_SPEC_v1.2.md`](THETA_SPEC_v1.2.md) — Mathematical definitions
- [`../README.md`](../README.md) — Repository overview
