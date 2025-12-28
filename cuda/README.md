<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Nenad Micic <nenad@micic.be>
-->

# Theta Experimental CUDA

**Version**: v10, 2025
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

This directory contains V10 of the theta order exploration CUDA implementation. The primary goal is exploring theta order in integers using GPU-accelerated enumeration, with fast composite detection using bitwise operations (no division).

---

## Files

### theta_order_complete_v10.cu

Main CUDA implementation for scanning odd integers organized by "theta order" (angular position on a square perimeter derived from binary representation).

**Key features:**
- Shell-based integer enumeration (power-of-2 ranges)
- Fast composite pre-filters using bit-chunk periodicity:
  - mod 3 (popcount-based)
  - mod 5 (byte sum)
  - mod 7 (9-bit chunks)
  - mod 11 (10-bit chunks)
  - mod 13 (12-bit chunks) - V10 new
- Miller-Rabin primality test (deterministic for 32/64-bit)
- Per-segment statistics aggregation
- Anomaly leaderboard tracking (V10 new)
- Filter-pass-rate tracking (V10 new)

**Requirements:** NVIDIA GPU with compute capability sm_86+ (for native atomicAdd on doubles)

**Build:**
```bash
nvcc -O3 -arch=sm_86 theta_order_complete_v10.cu -o theta_order_complete_v10
```

---

### theta_order_batch_v10.sh

Batch runner script for V10 with multiple scan modes.

**Modes:**
- `scan32` - Full 32-bit theta scan
- `scan64` - Full 64-bit theta scan
- `zoom64` - Zoom into specific segment range
- `batch` - Batch mode with checkpoints
- `deep` - Deep 64-bit scan to high shell numbers
- `compare` - Compare V9 vs V10 results

**Usage:**
```bash
./theta_order_batch_v10.sh scan32 8 28      # 32-bit, 256 segments, shells 2-28
./theta_order_batch_v10.sh scan64 10 40     # 64-bit, 1024 segments, shells 2-40
./theta_order_batch_v10.sh compare          # Compare V9 vs V10
./theta_order_batch_v10.sh help             # Show usage
```

---

### theta_order_complete_v10_spec.md

Full specification document serving as a regeneration anchor for the CUDA code.

**Contents:**
- **Part 1:** Reverse prompt (V10 changes, algorithm descriptions)
- **Part 2:** High-Level Design (HLD)
- **Part 3:** Low-Level Design (LLD)
- **Part 4:** Functional specification with examples
- **Appendix A:** Key algorithm code samples
- **Appendix B:** Bit-chunk periodicity mathematical foundation
- **Appendix C:** Complete formula set for theta coordinates

---

