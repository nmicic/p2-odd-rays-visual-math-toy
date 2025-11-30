# Theta Toolkit - Experimental Toy (CUDA & CPU)

**Version**: v1.2
**Date**: November 2025
**Repository**: https://github.com/nmicic/p2-odd-rays-visual-math-toy/

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

## Overview

A toy integer encoding based on 2-adic decomposition, explored for curiosity. Every positive integer `n` decomposes uniquely as `n = 2^v2(n) × core(n)`, where `theta_key(n) = bit_reverse(core(n))`.

This is a reversible mapping explored purely as a mathematical curiosity. No engineering applications are intended or implied.

---

## Files

### Mathematical Definitions

| File | Description |
|------|-------------|
| `cuda/THETA_SPEC_v1.2.md` | Mathematical definitions only |

### Python (Toy Reference)

| File | Description |
|------|-------------|
| `python/theta_toolkit.py` | CPU toy implementation |

### CUDA (Experimental Toy)

| File | Description |
|------|-------------|
| `cuda/theta_cuda_v1.2.cuh` | Experimental CUDA header (not for production) |
| `cuda/theta_cuda_benchmark_v1.2.cu` | Toy code to run the integer mapping |
| `cuda/THETA_CUDA_BENCHMARK_README_v1.2` | File descriptions |

---

## Example Usage (Toy Only)

### Python

```python
from python.theta_toolkit import theta_key, v2, odd_core

n = 12
print(f"n={n}, v2={v2(n)}, core={odd_core(n)}, theta_key={theta_key(n)}")
# n=12, v2=2, core=3, theta_key=3
```

### CUDA

```bash
# Compile
nvcc -O3 -arch=sm_89 cuda/theta_cuda_benchmark_v1.2.cu -o theta_bench

# Run (toy example, not a benchmark)
./theta_bench 10 64 1 0
```

### In Your Code (NOT RECOMMENDED)

> **Warning**: This code is for visualization/exploration only. Do not use in production.

```cpp
#include "theta_cuda_v1.2.cuh"

// Example only - NOT for production use
__global__ void example_kernel(uint32_t* data, uint32_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = theta::theta_key(data[idx]);
    }
}
```

---

## Version History

- **v1.2**: Added ray embedding visualization
- **v1.1**: Internal changes
- **v1.0**: Initial toy release

---

## Disclaimer

This is a **toy hobby project**. It is **not research** and **not peer-reviewed**.

**Not for any practical use:**
- Not cryptographic — deterministic, trivially reversible
- Not for security — do not use to protect secrets
- Not for hashing — do not use for routing, load balancing, or operational systems
- Not for production — use established, vetted libraries for real work
- Not validated — no claims of correctness, uniformity, or fitness for any purpose

Portions of this code include AI-assisted generation (ChatGPT, Claude). All work reviewed by the author.
