<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Nenad Micic <nenad@micic.be>
-->

# theta_order_complete_v10 - Full Specification

## Document Purpose

This document serves as a **regeneration anchor** for `theta_order_complete_v10.cu`. It contains:
1. A reverse prompt to regenerate the exact code
2. Complete algorithm explanations
3. V10-specific changes and bug fixes inherited from V9
4. High-Level Design (HLD) and Low-Level Design (LLD)
5. Appendix on bit-chunk periodicity for composite detection without division

**Target audience:** Future LLM models reviewing or extending this code.

**Primary goal:** Exploring theta order in integers. Primes are an interesting property but the main value is the efficient enumeration framework and fast composite detection using bitwise operations.

---

# PART 1: REVERSE PROMPT (Regeneration Anchor)

## Problem Restatement

Create a CUDA program that scans all odd integers within a specified range of "shells" (power-of-2 ranges), testing each for primality using Miller-Rabin, but **first applying fast bitwise composite filters** to reject obvious composites before the expensive primality test. The integers are organized by "theta order" (angular position on a square perimeter when mapped via a specific coordinate system), and results are aggregated per angular segment.

## V10 Changes from V9

### New Features

1. **Mod 13 Filter:**
   - Mathematical basis: 2^12 ≡ 1 (mod 13)
   - Uses 12-bit chunks, 6 iterations for 64-bit
   - Impact: Additional ~2.5% rejection, total ~61% filter rate

2. **Filter-Pass-Rate Tracking (`passed_to_mr`):**
   - New counter for candidates that passed all filters and reached Miller-Rabin
   - Useful for analyzing filter efficiency across segments

3. **Anomaly Leaderboard:**
   - Tracks top 20 densest and sparsest segments
   - Outputs to separate CSV file
   - Updated at checkpoints and end of run

4. **Small Prime Guard Updated:**
   - Now includes 13: `if (odd <= 13)` before filter cascade

### Inherited Bug Fixes (from V9)

1. **MOD5_LUT Size (CRITICAL):** LUT size 262 (indices 0-261 valid)
2. **Paired-stat Race Condition:** Removed theta_key fields from atomic updates
3. **Small Prime Guard (CRITICAL):** `if (odd <= 13)` guard before filter cascade (V10: extended from 11 to 13)
4. **Mod 7/11/13 Final Check (CRITICAL):** Use `(sum % p) == 0` for final check instead of LUT
5. **atomicAdd for Doubles (sm_86+):** Native support instead of CAS loop
6. **Mod 7 with 9-bit Chunks:** 8 iterations instead of 11

## Key Algorithms in Plain Language

### 1. Shell-Based Integer Enumeration
- **Shell k** contains all integers from `2^k` to `2^(k+1) - 1`
- Within each shell, odd integers are enumerated as: `odd = 2^k + 2*j + 1` for `j = 0, 1, ..., 2^(k-1) - 1`
- This gives exactly `2^(k-1)` odd integers per shell

### 2. Theta Key Computation (Angular Position)
- Each odd integer maps to an angular position on a square perimeter
- **Theta key formula:** For odd number `n` in shell `k`:
  ```
  frac = n XOR (1 << k)   // Remove leading bit, get fractional position
  theta_key = frac << (ref_shell - k)   // Scale to reference shell
  ```
- Theta key is an integer in range `[0, 2^ref_shell)` representing angular position

### 3. Segment Assignment
- The angular space is divided into `2^segment_exp` segments
- **Segment formula:**
  ```
  segment = theta_key >> (ref_shell - segment_exp)
  ```
- Each thread checks if its odd number belongs to the target segment

### 4. Fast Composite Pre-Filters (V10)

All filters use the **bit-chunk periodicity** principle: for prime `p`, if `2^m ≡ 1 (mod p)`, divisibility can be tested by summing m-bit chunks without division.

#### 4.1 Divisibility by 3 (Popcount-based)
**Mathematical basis:**
```
2 ≡ -1 (mod 3)
Therefore: 2^k ≡ (-1)^k (mod 3)
For n = Σ bₖ × 2^k:
  n mod 3 = |popcount(even_bits) - popcount(odd_bits)| mod 3
```

**Implementation:**
```c
even_bits = n & 0x5555555555555555ULL;  // Bits at positions 0,2,4,6,...
odd_bits = (n & 0xAAAAAAAAAAAAAAAAULL) >> 1;  // Bits at positions 1,3,5,7,...
diff = popcount(even_bits) - popcount(odd_bits);
// diff is in range [-32, +32] for 64-bit
return MOD3_LUT[abs(diff)] == 0;  // LUT avoids expensive % 3
```

#### 4.2 Divisibility by 5 (Byte Sum)
**Mathematical basis:**
```
2^8 = 256 ≡ 1 (mod 5)
Therefore: n mod 5 = (sum of all bytes) mod 5
```

**Implementation:**
```c
sum = byte[0] + byte[1] + ... + byte[7];  // 8 bytes for 64-bit
// sum max = 8 × 255 = 2040
sum = (sum & 0xFF) + (sum >> 8);  // Reduce: max 255 + 7 = 262
return MOD5_LUT[sum];
```

#### 4.3 Divisibility by 7 (9-bit Chunks)
**Mathematical basis:**
```
2^9 = 512 ≡ 1 (mod 7)
Therefore: n mod 7 = (sum of 9-bit chunks) mod 7
```

**Implementation:**
```c
sum = ((n >>  0) & 0x1FF) + ((n >>  9) & 0x1FF) +
      ((n >> 18) & 0x1FF) + ((n >> 27) & 0x1FF) +
      ((n >> 36) & 0x1FF) + ((n >> 45) & 0x1FF) +
      ((n >> 54) & 0x1FF) + ((n >> 63) & 0x01);
// Reduce with 9-bit folding
sum = (sum & 0x1FF) + (sum >> 9);
sum = (sum & 0x1FF) + (sum >> 9);
sum = (sum & 0x1FF) + (sum >> 9);
return (sum % 7) == 0;  // Use modulo, not LUT
```

#### 4.4 Divisibility by 11 (10-bit Chunks)
**Mathematical basis:**
```
2^10 = 1024 ≡ 1 (mod 11)
Therefore: n mod 11 = (sum of 10-bit chunks) mod 11
```

**Implementation:**
```c
sum = ((n >>  0) & 0x3FF) + ((n >> 10) & 0x3FF) +
      ((n >> 20) & 0x3FF) + ((n >> 30) & 0x3FF) +
      ((n >> 40) & 0x3FF) + ((n >> 50) & 0x3FF) +
      ((n >> 60) & 0x0F);
sum = (sum & 0x3FF) + (sum >> 10);
sum = (sum & 0x3FF) + (sum >> 10);
sum = (sum & 0x3FF) + (sum >> 10);
return (sum % 11) == 0;
```

#### 4.5 Divisibility by 13 (V10 NEW: 12-bit Chunks)
**Mathematical basis:**
```
2^12 = 4096 ≡ 1 (mod 13)
Therefore: n mod 13 = (sum of 12-bit chunks) mod 13
```

**Implementation:**
```c
__device__ __forceinline__ bool is_div_by_13_v10(uint64_t n) {
    // 64-bit = 5 full 12-bit chunks + 4 bits = 6 additions
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFFF) + ((n >> 36) & 0xFFF) +
                   ((n >> 48) & 0xFFF) + ((n >> 60) & 0x0F);

    // sum max = 5 × 4095 + 15 = 20490
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4099
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095

    return (sum % 13) == 0;
}

__device__ __forceinline__ bool is_div_by_13_v10_32(uint32_t n) {
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFF);

    sum = (sum & 0xFFF) + (sum >> 12);
    sum = (sum & 0xFFF) + (sum >> 12);

    return (sum % 13) == 0;
}
```

**Cost:** ~25 cycles (6 shifts + 6 masks + 5 adds + 2 folds + 1 modulo)

### 5. Miller-Rabin Primality Test
- **32-bit:** 3 witnesses {2, 7, 61} - deterministic for all 32-bit integers
- **64-bit:** 12 witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} - deterministic for all 64-bit integers
- Uses modular exponentiation with 128-bit intermediate products

## Invariants & Assumptions

1. **All tested numbers are odd** - even numbers filtered before enumeration
2. **Shell range is valid:** `2 <= min_shell <= max_shell <= 62` (64-bit) or `<= 31` (32-bit)
3. **Segment count is power of 2:** `num_segments = 2^segment_exp`
4. **Reference shell equals max_shell:** This ensures maximum theta key precision
5. **LUT sizes are correct:**
   - MOD3_LUT[65] for diff in [-32, +32]
   - MOD5_LUT[262] for byte sum ≤ 261
   - MOD7_LUT[74] for 9-bit sum ≤ 73
   - MOD11_LUT[67] for 10-bit sum ≤ 66
6. **Filters are applied in order:** mod 3, then mod 5, then mod 7, then mod 11, then mod 13 (most to least rejecting)
7. **Small prime guard:** `if (odd <= 13)` skip filters and go directly to MR
8. **Miller-Rabin witnesses are deterministic** for the given bit ranges
9. **Requires sm_86+ GPU** for atomicAdd on doubles

## Interfaces & I/O Contract

### Command Line Interface
```
theta_order_complete_v10 <mode> [options]

Modes:
  scan32    Full 32-bit theta scan (shells 2-31)
  scan64    Full 64-bit theta scan (shells 2-62)
  zoom64    Zoom into specific segment range
  batch     Batch mode with checkpoints

Options:
  --min-shell S       Minimum shell (default: 2)
  --max-shell S       Maximum shell (default: 31 or 62)
  --segment-exp E     Number of segments = 2^E (default: 10)
  --start-seg N       Starting segment (default: 0)
  --end-seg N         Ending segment (default: all)
  --output FILE       Output CSV file
  --anomaly FILE      Anomaly leaderboard CSV file (V10 NEW)
  --checkpoint FILE   Checkpoint file for resume
  --angles            Include angle columns in CSV
  --debug             Show debug information
  -q                  Quiet mode
  -v                  Verbose mode
```

### Output CSV Columns
```
segment, shell_config, shell_seen, total, primes, density,
filtered, filt3, filt5, filt7, filt11, filt13, passed_mr, mr_pct, filter_pct,
avg_shell, pop_range, avg_pop,
min_prime, max_prime, first_theta_prime, last_theta_prime,
twins, mod6_1, mod6_5, time_ms
[optional: seg_start_deg, seg_end_deg, prime angles...]
```

**V10 changes:** Added `filt13`, `passed_mr`, `mr_pct` columns.

### Anomaly CSV Columns (V10 NEW)
```
type,rank,segment,density,primes,total,filter_rate
dense,1,42,0.052345678901,12345,235678,61.5
dense,2,17,0.051234567890,...
sparse,1,789,0.041234567890,...
```

### Error Cases
- Invalid shell range → stderr message, exit 1
- CUDA kernel error → stderr with error string, exit 1
- Cannot open output file → stderr, exit 1
- Signal interrupt (SIGINT/SIGTERM) → graceful shutdown, checkpoint saved

---

# PART 2: HIGH-LEVEL DESIGN (HLD)

## Purpose

This system explores **theta order in integers** - a geometric coordinate system derived from binary representation. The primary value is:
1. Efficient integer space enumeration using GPU-friendly bitwise operations
2. Fast composite rejection without division using bit-chunk periodicity
3. Analysis framework for anomaly detection

## Key Algorithm: Overall Flow

```
For each angular segment s in [start_segment, end_segment):
    Initialize statistics for segment s

    For each shell k in [min_shell, max_shell]:
        For each j in [0, 2^(k-1)) in parallel (GPU threads):
            odd = 2^k + 2*j + 1

            # Compute angular position
            frac = odd XOR (1 << k)
            theta_key = frac << (ref_shell - k)
            seg = theta_key >> (ref_shell - segment_exp)

            If seg != target_segment: continue

            # V10: Small prime guard (includes 13)
            If odd <= 13: skip filters, go to MR

            # V10: Fast composite pre-filter cascade
            If is_div_by_3(odd): filtered_by_3++; continue
            If is_div_by_5(odd): filtered_by_5++; continue
            If is_div_by_7(odd): filtered_by_7++; continue
            If is_div_by_11(odd): filtered_by_11++; continue
            If is_div_by_13(odd): filtered_by_13++; continue  # V10 NEW

            # Passed all filters
            passed_to_mr++  # V10 NEW

            # Full primality test
            If is_prime(odd):
                prime_count++
                Update min/max/first/last statistics
                Check for twin prime

    Reduce warp-local statistics
    Atomic update global statistics
    Update anomaly leaderboard  # V10 NEW
    Write CSV row
```

## Data Structure Choices

### SegmentStats Structure (V10)
```c
struct SegmentStats {
    uint64_t segment_id;
    uint32_t min_shell, max_shell;
    uint32_t seen_min_shell, seen_max_shell;
    uint64_t total_tested, prime_count;
    uint64_t filtered_count;
    uint64_t filtered_by_3, filtered_by_5, filtered_by_7;
    uint64_t filtered_by_11;
    uint64_t filtered_by_13;  // V10 NEW
    uint64_t passed_to_mr;    // V10 NEW
    double shell_sum, popcount_sum;
    uint32_t min_popcount, max_popcount;
    uint64_t min_prime, max_prime;
    uint64_t first_theta_key, first_theta_prime;
    uint64_t last_theta_key, last_theta_prime;
    uint64_t twin_count, mod6_1, mod6_5;
    float time_ms;
};
```

### Anomaly Leaderboard (V10 NEW)
```c
#define ANOMALY_LEADERBOARD_SIZE 20

struct AnomalyEntry {
    uint64_t segment_id;
    double density;
    uint64_t prime_count;
    uint64_t total_tested;
    double filter_rate;
};

// Top 20 densest and sparsest segments
AnomalyEntry dense_leaders[ANOMALY_LEADERBOARD_SIZE];
AnomalyEntry sparse_leaders[ANOMALY_LEADERBOARD_SIZE];
```

### Constant Memory LUTs
```c
__constant__ uint8_t MOD3_LUT[65];   // For diff in [-32, +32]
__constant__ uint8_t MOD5_LUT[262];  // For byte sum ≤ 261
__constant__ uint8_t MOD7_LUT[74];   // For 9-bit sum ≤ 73 (not used in final check)
__constant__ uint8_t MOD11_LUT[67];  // For 10-bit sum ≤ 66 (not used in final check)
```

## Version History

| Version | Key Changes | Status |
|---------|-------------|--------|
| V1-V4 | Early development, various theta key formulas | Superseded |
| V5 | Working 32-bit and 64-bit, simple iteration | Superseded |
| V6 | Checkpoint, added resume capability | Superseded |
| V7 | Added mod 3/5 filters | Superseded |
| V8 | LUT mod3, byte mod5, 6-bit mod7 | Superseded |
| V9 | Bug fixes, atomicAdd, 9-bit mod7, mod11, ~58% rejection | Superseded |
| V10 | Mod 13 filter, passed_to_mr, anomaly leaderboard, ~61% rejection | **Current** |

---

# PART 3: LOW-LEVEL DESIGN (LLD)

## File Structure

```
theta_order_complete_v10.cu     # Single-file CUDA implementation
├── Includes and defines
├── Constant memory declarations (witnesses, LUTs)
│   ├── MOD3_LUT[65]
│   ├── MOD5_LUT[262]
│   ├── MOD7_LUT[74]
│   └── MOD11_LUT[67]
├── Device functions
│   ├── compute_shell_{32,64}
│   ├── is_div_by_3_v10{,_32}
│   ├── is_div_by_5_v10{,_32}
│   ├── is_div_by_7_v10{,_32}
│   ├── is_div_by_11_v10{,_32}
│   ├── is_div_by_13_v10{,_32}     # V10 NEW
│   ├── fast_composite_check_v10{,_32}
│   ├── get_theta_key_{32,64}
│   ├── mul128, mulmod64, powmod64
│   ├── is_prime_{32,64}
├── Kernel functions
│   ├── angular_segment_kernel_32
│   └── angular_segment_kernel_64
├── Host functions
│   ├── init_stats
│   ├── perimeter_t_to_degrees
│   ├── segment_angle_range
│   ├── theta_key_to_degrees
│   ├── get_theta_key_host
│   ├── prime_to_degrees
│   ├── update_anomaly_leaderboard   # V10 NEW
│   ├── write_csv_header
│   ├── write_csv_row
│   ├── write_anomaly_csv            # V10 NEW
│   ├── save_checkpoint
│   ├── load_checkpoint
│   └── print_usage
└── main()
```

## Kernel Control Flow (V10)

```
angular_segment_kernel_32(target_segment, num_segments, segment_exp,
                          min_shell, max_shell, ref_shell, stats)
│
├── Initialize local accumulators (all thread-local)
│   local_primes, local_total, local_filtered
│   local_filt3, local_filt5, local_filt7, local_filt11, local_filt13  # V10: Added filt13
│   local_passed_mr  # V10 NEW
│   local_shell_min/max, local_pop_min/max, local_shell_sum, local_pop_sum
│   local_min_prime, local_max_prime
│   local_first_theta_prime, local_last_theta_prime (with theta_keys)
│   local_twins, local_m1, local_m5
│
├── FOR shell in [min_shell, max_shell]:
│   │
│   ├── FOR j in [tid, 2^(shell-1)) with stride:
│   │   odd = (1 << shell) | (2*j + 1)
│   │   frac = odd ^ (1 << shell)
│   │   theta_key = frac << (ref_shell - shell)
│   │   seg = theta_key >> (ref_shell - segment_exp)
│   │
│   │   IF seg != target_segment: CONTINUE
│   │
│   │   local_total++
│   │
│   │   # V10: Small prime guard (includes 13)
│   │   IF odd <= 13:
│   │       local_passed_mr++
│   │       prime = is_prime_32(odd)
│   │   ELSE IF is_div_by_3_v10_32(odd):
│   │       local_filtered++; local_filt3++; CONTINUE
│   │   ELSE IF is_div_by_5_v10_32(odd):
│   │       local_filtered++; local_filt5++; CONTINUE
│   │   ELSE IF is_div_by_7_v10_32(odd):
│   │       local_filtered++; local_filt7++; CONTINUE
│   │   ELSE IF is_div_by_11_v10_32(odd):
│   │       local_filtered++; local_filt11++; CONTINUE
│   │   ELSE IF is_div_by_13_v10_32(odd):           # V10 NEW
│   │       local_filtered++; local_filt13++; CONTINUE
│   │   ELSE:
│   │       local_passed_mr++  # V10 NEW
│   │       prime = is_prime_32(odd)
│
├── WARP REDUCTION (for off = 16, 8, 4, 2, 1):
│   Use __shfl_down_sync to reduce all local accumulators
│   Special handling for min/max (compare and swap)
│
└── IF lane 0 of warp (threadIdx.x % 32 == 0):
    ├── atomicAdd for counters (including filtered_by_13, passed_to_mr)
    ├── atomicMin/Max for shell/popcount ranges
    ├── atomicCAS loop for min/max primes
    └── atomicAdd(&stats->shell_sum, ...) # Native atomicAdd for doubles
```

## Filter Cascade Summary (V10)

```
if (odd <= 13)        → skip filters, go to MR
else if (mod 3)       → reject (~33.3% of odds)
else if (mod 5)       → reject (~13.3% of remaining)
else if (mod 7)       → reject (~5.7% of remaining)
else if (mod 11)      → reject (~3.0% of remaining)
else if (mod 13)      → reject (~2.5% of remaining) ← V10 NEW
else                  → run Miller-Rabin
```

**Expected total rejection:** ~61-62% of odd composites

---

# PART 4: FUNCTIONAL SPECIFICATION

## Practical Usage Examples

### Example 1: Quick verification run
```bash
# Build (requires sm_86 or higher for atomicAdd on doubles)
nvcc -O3 -arch=sm_86 theta_order_complete_v10.cu -o theta_order_complete_v10

# Run small test
./theta_order_complete_v10 scan32 --max-shell 15 --segment-exp 4
```

### Example 2: Full scan with anomaly tracking
```bash
./theta_order_complete_v10 scan64 \
    --segment-exp 8 --max-shell 40 \
    --output theta_angular_64_v10.csv \
    --anomaly anomaly_64_v10.csv
```

### Example 3: Batch script
```bash
./theta_order_batch_v10.sh scan32 8 28
./theta_order_batch_v10.sh scan64 10 40
./theta_order_batch_v10.sh compare  # Compare V9 vs V10
```

## Benchmarks

### Filter Breakdown (V10 typical)

| Filter | Rejection | Cumulative |
|--------|-----------|------------|
| mod 3 | 33.3% | 33.3% |
| mod 5 | 13.3% | 46.6% |
| mod 7 | 5.7% | 52.3% |
| mod 11 | 3.0% | 55.3% |
| mod 13 | 2.5% | 57.8% |

**Note:** Exact percentages vary by range. The cumulative rejection before MR is ~58-62%.

### Performance Expectations

| Metric | V9 | V10 | Change |
|--------|-----|-----|--------|
| Filter rejection | ~58% | ~61% | +3% |
| MR calls | ~42% | ~39% | -3% |
| Filter cycles | ~90 | ~115 | +25 |
| Net throughput | baseline | ~same | negligible |

The additional ~25 cycles for mod 13 is offset by ~3% fewer MR calls (~2000 cycles each).

## Validation

V10 should produce identical prime counts to V9 (mod 13 filter is purely additive).

To verify:
```bash
./theta_order_complete_v9 scan32 --segment-exp 8 --max-shell 28 --output v9.csv
./theta_order_complete_v10 scan32 --segment-exp 8 --max-shell 28 --output v10.csv

# Compare prime counts
diff <(cut -d, -f5 v9.csv) <(cut -d, -f5 v10.csv)  # Should be empty
```

---

# APPENDIX A: Key Algorithm Code Samples (V10)

## A.1 Mod 13 Divisibility (V10 NEW)

```c
__device__ __forceinline__ bool is_div_by_13_v10(uint64_t n) {
    // 12-bit chunks: 2^12 = 4096 ≡ 1 (mod 13)
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFFF) + ((n >> 36) & 0xFFF) +
                   ((n >> 48) & 0xFFF) + ((n >> 60) & 0x0F);

    // sum max = 5 × 4095 + 15 = 20490
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4099
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095

    return (sum % 13) == 0;
}

__device__ __forceinline__ bool is_div_by_13_v10_32(uint32_t n) {
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFF);

    sum = (sum & 0xFFF) + (sum >> 12);
    sum = (sum & 0xFFF) + (sum >> 12);

    return (sum % 13) == 0;
}
```

## A.2 Small Prime Guard in Kernel (V10)

```c
// In both 32-bit and 64-bit kernels:
if (odd <= 13) {
    // Small primes: 3, 5, 7, 11, 13 - don't filter, go straight to MR
    local_passed_mr++;
    prime = is_prime_32(odd);  // or is_prime_64(odd) for 64-bit
} else if (is_div_by_3_v10_32(odd)) {
    filtered = true;
    local_filt3++;
} else if (is_div_by_5_v10_32(odd)) {
    filtered = true;
    local_filt5++;
} else if (is_div_by_7_v10_32(odd)) {
    filtered = true;
    local_filt7++;
} else if (is_div_by_11_v10_32(odd)) {
    filtered = true;
    local_filt11++;
} else if (is_div_by_13_v10_32(odd)) {
    filtered = true;
    local_filt13++;
} else {
    local_passed_mr++;
    prime = is_prime_32(odd);
}
```

## A.3 Mod 7 Divisibility (9-bit Chunks)

```c
__device__ __forceinline__ bool is_div_by_7_v10(uint64_t n) {
    uint32_t sum = ((n >>  0) & 0x1FF) + ((n >>  9) & 0x1FF) +
                   ((n >> 18) & 0x1FF) + ((n >> 27) & 0x1FF) +
                   ((n >> 36) & 0x1FF) + ((n >> 45) & 0x1FF) +
                   ((n >> 54) & 0x1FF) + ((n >> 63) & 0x01);

    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);

    return (sum % 7) == 0;
}
```

## A.4 Mod 11 Divisibility (10-bit Chunks)

```c
__device__ __forceinline__ bool is_div_by_11_v10(uint64_t n) {
    uint32_t sum = ((n >>  0) & 0x3FF) + ((n >> 10) & 0x3FF) +
                   ((n >> 20) & 0x3FF) + ((n >> 30) & 0x3FF) +
                   ((n >> 40) & 0x3FF) + ((n >> 50) & 0x3FF) +
                   ((n >> 60) & 0x0F);

    sum = (sum & 0x3FF) + (sum >> 10);
    sum = (sum & 0x3FF) + (sum >> 10);
    sum = (sum & 0x3FF) + (sum >> 10);

    return (sum % 11) == 0;
}
```

---

# APPENDIX B: Bit-Chunk Periodicity and "Chunk-Sum" Divisibility Tests

This appendix explains the mathematical basis behind the "sum fixed-size bit chunks" divisibility filters used in theta_order_complete_v10.cu, such as:
- mod 7 using 9-bit chunks (2^9 ≡ 1 (mod 7))
- mod 11 using 10-bit chunks (2^10 ≡ 1 (mod 11))
- mod 13 using 12-bit chunks (2^12 ≡ 1 (mod 13))

The key idea is that if a power of two has period 1 modulo p, then powers of two repeat every m bits, and a number becomes congruent to the sum of its m-bit chunks.

---

## B.1 The Basic Statement

Let p be an odd prime, and let m be a positive integer such that:

```
2^m ≡ 1 (mod p)
```

Write an integer n in base 2^m (i.e. in m-bit chunks):

```
n = Σ(i=0 to t) c_i × 2^(im)
```

where each chunk c_i satisfies 0 ≤ c_i < 2^m and is exactly the i-th m-bit block of n.

Then:

```
n ≡ Σ(i=0 to t) c_i (mod p)
```

So to test n ≡ 0 (mod p), you can test whether the chunk sum is ≡ 0 (mod p).

---

## B.2 Proof (the "trace")

Starting from the chunk decomposition:

```
n = Σ(i=0 to t) c_i × 2^(im)
```

Reduce modulo p:

```
n ≡ Σ(i=0 to t) c_i × 2^(im) (mod p)
```

Now use the periodicity assumption 2^m ≡ 1 (mod p). Then:

```
2^(im) = (2^m)^i ≡ 1^i ≡ 1 (mod p)
```

Substitute back:

```
n ≡ Σ(i=0 to t) c_i × 1 ≡ Σ(i=0 to t) c_i (mod p)
```

Done.

---

## B.3 "Folding" is Just Applying the Same Congruence Repeatedly

In CUDA you typically:
1. compute a first chunk sum (a handful of masks/shifts/adds),
2. then fold the sum to keep it small.

That folding step is not a heuristic—it is literally re-applying the same congruence.

**Example: for m = 12 (mod 13 filter):**

Let S be the initial sum of 12-bit chunks. Write S itself as:

```
S = a + b × 2^12
```

where:
- a = S mod 2^12 (low 12 bits)
- b = floor(S / 2^12) (everything above)

Then mod p (here p=13):

```
S ≡ a + b × 2^12 ≡ a + b × 1 ≡ a + b (mod 13)
```

So the fold operation:
```c
S = (S & ((1<<m)-1)) + (S >> m)
```

preserves S (mod p) whenever 2^m ≡ 1 (mod p).

You can fold repeatedly until the value is in a small range, then do a final % p (or LUT).

---

## B.4 Why This is Called "Bit-Chunk Periodicity"

The condition 2^m ≡ 1 (mod p) means that powers of two repeat every m steps modulo p:

```
2^(k+m) ≡ 2^k × 2^m ≡ 2^k × 1 ≡ 2^k (mod p)
```

So, modulo p, the weights of bits repeat with period m—which is exactly why "group bits into m-bit blocks and sum the blocks" works.

More generally, the multiplicative order of 2 modulo p is the smallest m such that 2^m ≡ 1 (mod p). Any multiple of that order also satisfies the condition.

---

## B.5 Examples Used in V10

### mod 7 with 9-bit chunks

**Claim:** 2^9 = 512 ≡ 1 (mod 7)

**Reason:** 512 = 7×73 + 1.

So with m=9, any n satisfies:

```
n ≡ Σ (9-bit chunks of n) (mod 7)
```

In code (conceptually):
- chunk mask = 0x1FF (9 bits)
- sum chunks: ((n>>0)&mask) + ((n>>9)&mask) + ...
- fold: sum = (sum & mask) + (sum >> 9) repeated a few times
- final test: sum % 7 == 0

### mod 11 with 10-bit chunks

```
2^10 = 1024 ≡ 1 (mod 11)
```

since 1024 = 11×93 + 1.

So:

```
n ≡ Σ (10-bit chunks of n) (mod 11)
```

### mod 13 with 12-bit chunks

```
2^12 = 4096 ≡ 1 (mod 13)
```

since 4096 = 13×315 + 1.

So:

```
n ≡ Σ (12-bit chunks of n) (mod 13)
```

V10 implementation matches this directly:
- 64-bit is 5 full 12-bit chunks + 4 leftover bits → 6 additions
- then 12-bit folding
- then % 13 as final guard

---

## B.6 General Recipe: Designing a Chunk-Sum Filter for a Small Prime

To build a chunk-sum divisibility test for a prime p:

1. **Find an m** such that 2^m ≡ 1 (mod p).
   - This m is a multiple of ord_p(2) (the order of 2 mod p).
   - Often you choose m so chunking is convenient (e.g. 8, 9, 10, 12, 16), but only works if the congruence holds.

2. **Implement:**
   - mask = (1<<m) - 1
   - sum = Σ ((n >> (i*m)) & mask) for enough chunks to cover your word size
   - fold: sum = (sum & mask) + (sum >> m) a few times
   - final: check sum % p == 0 or use a LUT once sum is small enough

3. **Validate** with randomized testing:
   - ensure is_div_by_p_chunk(n) matches (n % p == 0) for many random n, plus adversarial bit patterns (all ones, alternating masks, powers of two, etc.).

---

## B.7 Practical Notes for GPU Implementation

- The first chunk-sum pass is predictable: fixed number of shifts/masks/adds.
- Folding is branch-free and quickly reduces sum to a small bound.
- The final % p can be:
  - kept as % p (simple, often good enough once sum is small), or
  - replaced with a small LUT if sum is provably within a small maximum.

For V10-style sums, folding typically keeps sum < 2^m, and after 1–3 folds it becomes tiny relative to the original 64-bit n.

---

## B.8 Important Limitation

This method depends on:

```
2^m ≡ 1 (mod p)
```

If that does not hold, you cannot collapse 2^(im) to 1, and the chunk sum is not congruent to n (mod p).

In that case, you can sometimes use a related trick with small periodic coefficients (e.g. alternating sums), but it becomes a different filter family.

---

## B.9 One-Line Takeaway

**If 2^m ≡ 1 (mod p), then "base 2^m" digits (m-bit chunks) add up to the same residue mod p.**
That's why your 9/10/12-bit chunk filters are mathematically exact, not heuristic.

---

# APPENDIX C: Complete Formula Set

## C.1 Decomposition

```
v2 = ctz(n)                     // trailing zeros
core = n >> v2                  // odd core
shell_n = floor(log2(n))        // shell of n
k = floor(log2(core))           // shell of core
```

## C.2 X Coordinate

```
x_num = 2^shell_n - 1           // all-ones mask
x_den = 2^(shell_n - 1)
x = x_num / x_den = 2 - 2^(1-shell_n)
```

## C.3 Slope (Simplified)

```
slope_num = 3×core - 2^(k+2)
slope_den = 2^k
slope = slope_num / slope_den

GPU (no multiply!):
    t = core + (core << 1)        // 3 × core
    slope_num = t - (1 << (k+2))  // 3×core - 2^(k+2)
```

## C.4 Y Coordinate (Shift-Subtract, No Multiply)

```
y_num = (slope_num << shell_n) - slope_num
y_den = 2^(k + shell_n - 1)
y = y_num / y_den
```

## C.5 Theta Key

```
frac = odd XOR (1 << k)
theta_key = frac << (ref_shell - k)
segment = theta_key >> (ref_shell - segment_exp)
```

---

*Document version: 1.0*
*Generated for: theta_order_complete_v10.cu*
