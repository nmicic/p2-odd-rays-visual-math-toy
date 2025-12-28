// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Nenad Micic <nenad@micic.be>
//
// theta_order_complete_v10.cu - Extended Filters + Anomaly Tracking
//
// ============================================================================
// V10: EXTENDED FILTERS + ANOMALY TRACKING
// ============================================================================
//
// V10 changes over V9:
//   1. FEATURE: Added mod 13 filter with 12-bit chunks (6 iterations)
//   2. FEATURE: Track passed_to_mr count (filter-pass-rate metric)
//   3. FEATURE: Anomaly leaderboard CSV output (top/bottom segments by density)
//
// Design goals:
//   - Low complexity, high gain
//   - Low regression risk (additive changes only)
//   - Minimal performance impact (mod 13 only runs on ~41% of candidates)
//   - Machine-native: pure bitwise integer operations
//
// Expected rejection: ~61-62% of odd composites (vs ~58% in V9)
//
// ============================================================================
// COMPILATION
// ============================================================================
//
//   nvcc -O3 -arch=sm_86 theta_order_complete_v10.cu -o theta_order_complete_v10
//
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <signal.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL

// V10: Anomaly leaderboard size
#define ANOMALY_LEADERBOARD_SIZE 20

// Global flag for graceful shutdown
volatile sig_atomic_t shutdown_requested = 0;

void signal_handler(int sig) {
    shutdown_requested = 1;
    fprintf(stderr, "\nShutdown requested (signal %d). Finishing current segment...\n", sig);
}

// ============================================================================
// Miller-Rabin witnesses
// ============================================================================

__constant__ uint64_t MR_WITNESSES_3[3] = {2, 7, 61};
__constant__ uint64_t MR_WITNESSES_12[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

// ============================================================================
// V10: Lookup tables for fast modular arithmetic
// ============================================================================

// MOD3_LUT[i] = i % 3 for i = 0..64
__constant__ uint8_t MOD3_LUT[65] = {
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1
};

// MOD5_LUT[i] = (i % 5 == 0) for i = 0..261
__constant__ uint8_t MOD5_LUT[262] = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 0-19
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 20-39
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 40-59
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 60-79
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 80-99
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 100-119
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 120-139
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 140-159
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 160-179
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 180-199
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 200-219
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 220-239
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 240-259
    1, 0                                                          // 260-261
};

// ============================================================================
// Bit operations
// ============================================================================

__device__ __forceinline__ uint32_t compute_shell_32(uint32_t n) {
    if (n == 0) return 0;
    return 31 - __clz(n);
}

__device__ __forceinline__ uint32_t compute_shell_64(uint64_t n) {
    if (n == 0) return 0;
    return 63 - __clzll(n);
}

// ============================================================================
// V10: Optimized Fast Composite Filters
// ============================================================================

// Divisibility by 3: LUT-based
__device__ __forceinline__ bool is_div_by_3_v10(uint64_t n) {
    uint64_t even_bits = n & 0x5555555555555555ULL;
    uint64_t odd_bits = (n & 0xAAAAAAAAAAAAAAAAULL) >> 1;

    int even_count = __popcll(even_bits);
    int odd_count = __popcll(odd_bits);
    int diff = even_count - odd_count;

    int absdiff = (diff ^ (diff >> 31)) - (diff >> 31);
    return MOD3_LUT[absdiff] == 0;
}

__device__ __forceinline__ bool is_div_by_3_v10_32(uint32_t n) {
    uint32_t even_bits = n & 0x55555555U;
    uint32_t odd_bits = (n & 0xAAAAAAAAU) >> 1;

    int even_count = __popc(even_bits);
    int odd_count = __popc(odd_bits);
    int diff = even_count - odd_count;

    int absdiff = (diff ^ (diff >> 31)) - (diff >> 31);
    return MOD3_LUT[absdiff] == 0;
}

// Divisibility by 5: Byte sum
__device__ __forceinline__ bool is_div_by_5_v10(uint64_t n) {
    uint32_t sum = ((n >>  0) & 0xFF) + ((n >>  8) & 0xFF) +
                   ((n >> 16) & 0xFF) + ((n >> 24) & 0xFF) +
                   ((n >> 32) & 0xFF) + ((n >> 40) & 0xFF) +
                   ((n >> 48) & 0xFF) + ((n >> 56) & 0xFF);

    sum = (sum & 0xFF) + (sum >> 8);
    return MOD5_LUT[sum];
}

__device__ __forceinline__ bool is_div_by_5_v10_32(uint32_t n) {
    uint32_t sum = ((n >>  0) & 0xFF) + ((n >>  8) & 0xFF) +
                   ((n >> 16) & 0xFF) + ((n >> 24) & 0xFF);

    sum = (sum & 0xFF) + (sum >> 8);
    return MOD5_LUT[sum];
}

// Divisibility by 7 using 9-bit chunks
// Mathematical basis: 2^9 = 512 ≡ 1 (mod 7)
__device__ __forceinline__ bool is_div_by_7_v10(uint64_t n) {
    uint32_t sum = ((n >>  0) & 0x1FF) + ((n >>  9) & 0x1FF) +
                   ((n >> 18) & 0x1FF) + ((n >> 27) & 0x1FF) +
                   ((n >> 36) & 0x1FF) + ((n >> 45) & 0x1FF) +
                   ((n >> 54) & 0x1FF) +
                   ((n >> 63) & 0x01);

    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);

    return (sum % 7) == 0;
}

__device__ __forceinline__ bool is_div_by_7_v10_32(uint32_t n) {
    uint32_t sum = ((n >>  0) & 0x1FF) + ((n >>  9) & 0x1FF) +
                   ((n >> 18) & 0x1FF) + ((n >> 27) & 0x1F);

    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);
    sum = (sum & 0x1FF) + (sum >> 9);

    return (sum % 7) == 0;
}

// Divisibility by 11 using 10-bit chunks
// Mathematical basis: 2^10 = 1024 ≡ 1 (mod 11)
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

__device__ __forceinline__ bool is_div_by_11_v10_32(uint32_t n) {
    uint32_t sum = ((n >>  0) & 0x3FF) + ((n >> 10) & 0x3FF) +
                   ((n >> 20) & 0x3FF) + ((n >> 30) & 0x03);

    sum = (sum & 0x3FF) + (sum >> 10);
    sum = (sum & 0x3FF) + (sum >> 10);

    return (sum % 11) == 0;
}

// V10 NEW: Divisibility by 13 using 12-bit chunks
// Mathematical basis: 2^12 = 4096 ≡ 1 (mod 13)
// 64-bit = 5 full 12-bit chunks + 4 bits = 6 additions
__device__ __forceinline__ bool is_div_by_13_v10(uint64_t n) {
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFFF) + ((n >> 36) & 0xFFF) +
                   ((n >> 48) & 0xFFF) + ((n >> 60) & 0x0F);

    // sum max = 5 × 4095 + 15 = 20490
    // Reduce using 12-bit folding: 2^12 ≡ 1 (mod 13)
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095 + 4 = 4099
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095 + 0 = 4095

    // Use modulo for final check (safe for all values)
    return (sum % 13) == 0;
}

__device__ __forceinline__ bool is_div_by_13_v10_32(uint32_t n) {
    // 32-bit = 2 full 12-bit chunks + 8 bits = 3 additions
    uint32_t sum = ((n >>  0) & 0xFFF) + ((n >> 12) & 0xFFF) +
                   ((n >> 24) & 0xFF);

    // sum max = 2 × 4095 + 255 = 8445
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095 + 2 = 4097
    sum = (sum & 0xFFF) + (sum >> 12);  // max 4095 + 0 = 4095

    return (sum % 13) == 0;
}

// ============================================================================
// Theta key computation
// ============================================================================

__device__ __forceinline__ uint64_t get_theta_key_64(uint64_t odd, uint32_t ref_shell) {
    if (odd < 4) return 0;
    uint32_t shell = 63 - __clzll(odd);
    if (shell < 2) return 0;
    uint64_t frac = odd ^ (1ULL << shell);
    if (shell <= ref_shell) {
        return frac << (ref_shell - shell);
    } else {
        return frac >> (shell - ref_shell);
    }
}

__device__ __forceinline__ uint32_t get_theta_key_32(uint32_t odd, uint32_t ref_shell) {
    if (odd < 4) return 0;
    uint32_t shell = 31 - __clz(odd);
    if (shell < 2) return 0;
    uint32_t frac = odd ^ (1U << shell);
    if (shell <= ref_shell) {
        return frac << (ref_shell - shell);
    } else {
        return frac >> (shell - ref_shell);
    }
}

// ============================================================================
// Miller-Rabin primality tests
// ============================================================================

__device__ __forceinline__ void mul128(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi, lo;
    mul128(a, b, &hi, &lo);
    if (hi == 0) return lo % m;
    uint64_t result = lo % m;
    uint64_t factor = hi % m;
    for (int i = 0; i < 64; i++) factor = (factor << 1) % m;
    return (result + factor) % m;
}

__device__ uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, m);
        exp >>= 1;
        base = mulmod64(base, base, m);
    }
    return result;
}

__device__ bool is_prime_32(uint32_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;

    uint32_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    for (int i = 0; i < 3; i++) {
        uint64_t a = MR_WITNESSES_3[i];
        if (a >= n) continue;
        uint64_t x = 1, base = a % n;
        uint32_t exp = d;
        while (exp > 0) {
            if (exp & 1) x = (x * base) % n;
            exp >>= 1;
            base = (base * base) % n;
        }
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = (x * x) % n;
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

__device__ bool is_prime_64(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
    if (n < THRESHOLD_32BIT) return is_prime_32((uint32_t)n);

    uint64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    for (int i = 0; i < 12; i++) {
        uint64_t a = MR_WITNESSES_12[i];
        if (a >= n) continue;
        uint64_t x = powmod64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = mulmod64(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// ============================================================================
// V10: Statistics structure
// ============================================================================

struct SegmentStats {
    uint64_t segment_id;

    uint32_t min_shell;
    uint32_t max_shell;
    uint32_t seen_min_shell;
    uint32_t seen_max_shell;

    uint64_t total_tested;
    uint64_t prime_count;
    uint64_t filtered_count;
    uint64_t filtered_by_3;
    uint64_t filtered_by_5;
    uint64_t filtered_by_7;
    uint64_t filtered_by_11;
    uint64_t filtered_by_13;  // V10 NEW
    uint64_t passed_to_mr;    // V10 NEW: count of candidates that reached MR

    double shell_sum;
    uint32_t min_popcount;
    uint32_t max_popcount;
    double popcount_sum;

    uint64_t min_prime;
    uint64_t max_prime;
    uint64_t first_theta_key;
    uint64_t first_theta_prime;
    uint64_t last_theta_key;
    uint64_t last_theta_prime;

    uint64_t twin_count;
    uint64_t mod6_1;
    uint64_t mod6_5;

    float time_ms;
};

// V10: Anomaly leaderboard entry
struct AnomalyEntry {
    uint64_t segment_id;
    double density;
    uint64_t prime_count;
    uint64_t total_tested;
    double filter_rate;
};

// ============================================================================
// 32-bit kernel
// ============================================================================

__global__ void angular_segment_kernel_32(
    uint32_t target_segment,
    uint64_t num_segments,
    uint32_t segment_exp,
    uint32_t min_shell,
    uint32_t max_shell,
    uint32_t ref_shell,
    SegmentStats* stats
) {
    uint64_t local_primes = 0;
    uint64_t local_total = 0;
    uint64_t local_filtered = 0;
    uint64_t local_filt3 = 0, local_filt5 = 0, local_filt7 = 0, local_filt11 = 0, local_filt13 = 0;
    uint64_t local_passed_mr = 0;  // V10 NEW
    uint32_t local_shell_min = UINT32_MAX;
    uint32_t local_shell_max = 0;
    double local_shell_sum = 0.0;
    uint32_t local_pop_min = UINT32_MAX;
    uint32_t local_pop_max = 0;
    double local_pop_sum = 0.0;

    uint64_t local_min_prime = UINT64_MAX;
    uint64_t local_max_prime = 0;

    uint64_t local_first_theta_prime = 0;
    uint64_t local_first_theta_key = UINT64_MAX;
    uint64_t local_last_theta_prime = 0;
    uint64_t local_last_theta_key = 0;

    uint64_t local_twins = 0;
    uint64_t local_m1 = 0, local_m5 = 0;

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
        uint64_t shell_size = 1ULL << (shell - 1);

        for (uint64_t j = idx; j < shell_size; j += stride) {
            uint32_t odd = (1U << shell) + 2 * (uint32_t)j + 1;

            uint32_t frac = odd ^ (1U << shell);
            uint32_t theta_key = frac << (ref_shell - shell);

            uint32_t theta_bits = ref_shell;
            uint32_t seg;
            if (theta_bits >= segment_exp) {
                seg = theta_key >> (theta_bits - segment_exp);
            } else {
                seg = theta_key << (segment_exp - theta_bits);
            }

            if (seg != target_segment) continue;

            local_total++;
            uint32_t s = shell;
            uint32_t p = __popc(odd);
            local_shell_sum += s;
            local_pop_sum += p;
            if (s < local_shell_min) local_shell_min = s;
            if (s > local_shell_max) local_shell_max = s;
            if (p < local_pop_min) local_pop_min = p;
            if (p > local_pop_max) local_pop_max = p;

            // V10: Fast composite pre-filter with mod 13
            bool prime = false;
            bool filtered = false;

            if (odd <= 13) {
                // Small primes: 3, 5, 7, 11, 13 - don't filter, go straight to MR
                local_passed_mr++;
                prime = is_prime_32(odd);
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
            } else if (is_div_by_13_v10_32(odd)) {  // V10 NEW
                filtered = true;
                local_filt13++;
            } else {
                local_passed_mr++;
                prime = is_prime_32(odd);
            }

            if (filtered) {
                local_filtered++;
            }

            if (prime) {
                local_primes++;

                if (odd < local_min_prime) {
                    local_min_prime = odd;
                }
                if (odd > local_max_prime) {
                    local_max_prime = odd;
                }

                if (theta_key < local_first_theta_key) {
                    local_first_theta_key = theta_key;
                    local_first_theta_prime = odd;
                }
                if (theta_key > local_last_theta_key) {
                    local_last_theta_key = theta_key;
                    local_last_theta_prime = odd;
                }

                uint32_t odd2 = odd + 2;
                if (odd2 > odd && is_prime_32(odd2)) local_twins++;

                if (odd > 3) {
                    uint32_t r = odd % 6;
                    if (r == 1) local_m1++;
                    else if (r == 5) local_m5++;
                }
            }
        }
    }

    // Warp reduction
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        local_primes += __shfl_down_sync(0xFFFFFFFF, local_primes, off);
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, off);
        local_filtered += __shfl_down_sync(0xFFFFFFFF, local_filtered, off);
        local_filt3 += __shfl_down_sync(0xFFFFFFFF, local_filt3, off);
        local_filt5 += __shfl_down_sync(0xFFFFFFFF, local_filt5, off);
        local_filt7 += __shfl_down_sync(0xFFFFFFFF, local_filt7, off);
        local_filt11 += __shfl_down_sync(0xFFFFFFFF, local_filt11, off);
        local_filt13 += __shfl_down_sync(0xFFFFFFFF, local_filt13, off);
        local_passed_mr += __shfl_down_sync(0xFFFFFFFF, local_passed_mr, off);
        local_shell_sum += __shfl_down_sync(0xFFFFFFFF, local_shell_sum, off);
        local_pop_sum += __shfl_down_sync(0xFFFFFFFF, local_pop_sum, off);
        local_twins += __shfl_down_sync(0xFFFFFFFF, local_twins, off);
        local_m1 += __shfl_down_sync(0xFFFFFFFF, local_m1, off);
        local_m5 += __shfl_down_sync(0xFFFFFFFF, local_m5, off);

        uint64_t other_min = __shfl_down_sync(0xFFFFFFFF, local_min_prime, off);
        if (other_min < local_min_prime) local_min_prime = other_min;

        uint64_t other_max = __shfl_down_sync(0xFFFFFFFF, local_max_prime, off);
        if (other_max > local_max_prime) local_max_prime = other_max;

        uint64_t other_first_key = __shfl_down_sync(0xFFFFFFFF, local_first_theta_key, off);
        uint64_t other_first_prime = __shfl_down_sync(0xFFFFFFFF, local_first_theta_prime, off);
        if (other_first_key < local_first_theta_key) {
            local_first_theta_key = other_first_key;
            local_first_theta_prime = other_first_prime;
        }

        uint64_t other_last_key = __shfl_down_sync(0xFFFFFFFF, local_last_theta_key, off);
        uint64_t other_last_prime = __shfl_down_sync(0xFFFFFFFF, local_last_theta_prime, off);
        if (other_last_key > local_last_theta_key) {
            local_last_theta_key = other_last_key;
            local_last_theta_prime = other_last_prime;
        }

        uint32_t other_shell_min = __shfl_down_sync(0xFFFFFFFF, local_shell_min, off);
        uint32_t other_shell_max = __shfl_down_sync(0xFFFFFFFF, local_shell_max, off);
        uint32_t other_pop_min = __shfl_down_sync(0xFFFFFFFF, local_pop_min, off);
        uint32_t other_pop_max = __shfl_down_sync(0xFFFFFFFF, local_pop_max, off);
        if (other_shell_min < local_shell_min) local_shell_min = other_shell_min;
        if (other_shell_max > local_shell_max) local_shell_max = other_shell_max;
        if (other_pop_min < local_pop_min) local_pop_min = other_pop_min;
        if (other_pop_max > local_pop_max) local_pop_max = other_pop_max;
    }

    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->filtered_count, local_filtered);
        atomicAdd((unsigned long long*)&stats->filtered_by_3, local_filt3);
        atomicAdd((unsigned long long*)&stats->filtered_by_5, local_filt5);
        atomicAdd((unsigned long long*)&stats->filtered_by_7, local_filt7);
        atomicAdd((unsigned long long*)&stats->filtered_by_11, local_filt11);
        atomicAdd((unsigned long long*)&stats->filtered_by_13, local_filt13);
        atomicAdd((unsigned long long*)&stats->passed_to_mr, local_passed_mr);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        atomicAdd((unsigned long long*)&stats->mod6_1, local_m1);
        atomicAdd((unsigned long long*)&stats->mod6_5, local_m5);

        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);

        uint64_t old = stats->min_prime;
        while (local_min_prime < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->min_prime, old, local_min_prime);
            if (prev == old) break;
            old = prev;
        }
        old = stats->max_prime;
        while (local_max_prime > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->max_prime, old, local_max_prime);
            if (prev == old) break;
            old = prev;
        }

        if (local_first_theta_key < UINT64_MAX) {
            uint64_t old_key = stats->first_theta_key;
            while (local_first_theta_key < old_key) {
                uint64_t prev = atomicCAS((unsigned long long*)&stats->first_theta_key, old_key, local_first_theta_key);
                if (prev == old_key) {
                    stats->first_theta_prime = local_first_theta_prime;
                    break;
                }
                old_key = prev;
            }
        }

        if (local_last_theta_key > 0) {
            uint64_t old_key = stats->last_theta_key;
            while (local_last_theta_key > old_key) {
                uint64_t prev = atomicCAS((unsigned long long*)&stats->last_theta_key, old_key, local_last_theta_key);
                if (prev == old_key) {
                    stats->last_theta_prime = local_last_theta_prime;
                    break;
                }
                old_key = prev;
            }
        }

        atomicAdd(&stats->shell_sum, local_shell_sum);
        atomicAdd(&stats->popcount_sum, local_pop_sum);
    }
}

// ============================================================================
// 64-bit kernel
// ============================================================================

__global__ void angular_segment_kernel_64(
    uint64_t target_segment,
    uint64_t num_segments,
    uint32_t segment_exp,
    uint32_t min_shell,
    uint32_t max_shell,
    uint32_t ref_shell,
    SegmentStats* stats
) {
    uint64_t local_primes = 0;
    uint64_t local_total = 0;
    uint64_t local_filtered = 0;
    uint64_t local_filt3 = 0, local_filt5 = 0, local_filt7 = 0, local_filt11 = 0, local_filt13 = 0;
    uint64_t local_passed_mr = 0;  // V10 NEW
    uint32_t local_shell_min = UINT32_MAX;
    uint32_t local_shell_max = 0;
    double local_shell_sum = 0.0;
    uint32_t local_pop_min = UINT32_MAX;
    uint32_t local_pop_max = 0;
    double local_pop_sum = 0.0;

    uint64_t local_min_prime = UINT64_MAX;
    uint64_t local_max_prime = 0;

    uint64_t local_first_theta_prime = 0;
    uint64_t local_first_theta_key = UINT64_MAX;
    uint64_t local_last_theta_prime = 0;
    uint64_t local_last_theta_key = 0;

    uint64_t local_twins = 0;
    uint64_t local_m1 = 0, local_m5 = 0;

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
        uint64_t shell_size = 1ULL << (shell - 1);

        for (uint64_t j = idx; j < shell_size; j += stride) {
            uint64_t odd = (1ULL << shell) + 2ULL * j + 1ULL;

            uint64_t frac = odd ^ (1ULL << shell);
            uint64_t theta_key;
            if (shell <= ref_shell) {
                theta_key = frac << (ref_shell - shell);
            } else {
                theta_key = frac >> (shell - ref_shell);
            }

            uint64_t seg;
            if (ref_shell >= segment_exp) {
                seg = theta_key >> (ref_shell - segment_exp);
            } else {
                seg = theta_key << (segment_exp - ref_shell);
            }

            if (seg != target_segment) continue;

            local_total++;
            uint32_t s = shell;
            uint32_t p = __popcll(odd);
            local_shell_sum += s;
            local_pop_sum += p;
            if (s < local_shell_min) local_shell_min = s;
            if (s > local_shell_max) local_shell_max = s;
            if (p < local_pop_min) local_pop_min = p;
            if (p > local_pop_max) local_pop_max = p;

            // V10: Fast composite pre-filter with mod 13
            bool prime = false;
            bool filtered = false;

            if (odd <= 13) {
                // Small primes: 3, 5, 7, 11, 13 - don't filter, go straight to MR
                local_passed_mr++;
                prime = is_prime_64(odd);
            } else if (is_div_by_3_v10(odd)) {
                filtered = true;
                local_filt3++;
            } else if (is_div_by_5_v10(odd)) {
                filtered = true;
                local_filt5++;
            } else if (is_div_by_7_v10(odd)) {
                filtered = true;
                local_filt7++;
            } else if (is_div_by_11_v10(odd)) {
                filtered = true;
                local_filt11++;
            } else if (is_div_by_13_v10(odd)) {  // V10 NEW
                filtered = true;
                local_filt13++;
            } else {
                local_passed_mr++;
                prime = is_prime_64(odd);
            }

            if (filtered) {
                local_filtered++;
            }

            if (prime) {
                local_primes++;

                if (odd < local_min_prime) {
                    local_min_prime = odd;
                }
                if (odd > local_max_prime) {
                    local_max_prime = odd;
                }

                if (theta_key < local_first_theta_key) {
                    local_first_theta_key = theta_key;
                    local_first_theta_prime = odd;
                }
                if (theta_key > local_last_theta_key) {
                    local_last_theta_key = theta_key;
                    local_last_theta_prime = odd;
                }

                uint64_t odd2 = odd + 2;
                if (odd2 > odd && is_prime_64(odd2)) local_twins++;

                if (odd > 3) {
                    uint32_t r = odd % 6;
                    if (r == 1) local_m1++;
                    else if (r == 5) local_m5++;
                }
            }
        }
    }

    // Warp reduction
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        local_primes += __shfl_down_sync(0xFFFFFFFF, local_primes, off);
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, off);
        local_filtered += __shfl_down_sync(0xFFFFFFFF, local_filtered, off);
        local_filt3 += __shfl_down_sync(0xFFFFFFFF, local_filt3, off);
        local_filt5 += __shfl_down_sync(0xFFFFFFFF, local_filt5, off);
        local_filt7 += __shfl_down_sync(0xFFFFFFFF, local_filt7, off);
        local_filt11 += __shfl_down_sync(0xFFFFFFFF, local_filt11, off);
        local_filt13 += __shfl_down_sync(0xFFFFFFFF, local_filt13, off);
        local_passed_mr += __shfl_down_sync(0xFFFFFFFF, local_passed_mr, off);
        local_shell_sum += __shfl_down_sync(0xFFFFFFFF, local_shell_sum, off);
        local_pop_sum += __shfl_down_sync(0xFFFFFFFF, local_pop_sum, off);
        local_twins += __shfl_down_sync(0xFFFFFFFF, local_twins, off);
        local_m1 += __shfl_down_sync(0xFFFFFFFF, local_m1, off);
        local_m5 += __shfl_down_sync(0xFFFFFFFF, local_m5, off);

        uint64_t other_min = __shfl_down_sync(0xFFFFFFFF, local_min_prime, off);
        if (other_min < local_min_prime) local_min_prime = other_min;

        uint64_t other_max = __shfl_down_sync(0xFFFFFFFF, local_max_prime, off);
        if (other_max > local_max_prime) local_max_prime = other_max;

        uint64_t other_first_key = __shfl_down_sync(0xFFFFFFFF, local_first_theta_key, off);
        uint64_t other_first_prime = __shfl_down_sync(0xFFFFFFFF, local_first_theta_prime, off);
        if (other_first_key < local_first_theta_key) {
            local_first_theta_key = other_first_key;
            local_first_theta_prime = other_first_prime;
        }

        uint64_t other_last_key = __shfl_down_sync(0xFFFFFFFF, local_last_theta_key, off);
        uint64_t other_last_prime = __shfl_down_sync(0xFFFFFFFF, local_last_theta_prime, off);
        if (other_last_key > local_last_theta_key) {
            local_last_theta_key = other_last_key;
            local_last_theta_prime = other_last_prime;
        }

        uint32_t other_shell_min = __shfl_down_sync(0xFFFFFFFF, local_shell_min, off);
        uint32_t other_shell_max = __shfl_down_sync(0xFFFFFFFF, local_shell_max, off);
        uint32_t other_pop_min = __shfl_down_sync(0xFFFFFFFF, local_pop_min, off);
        uint32_t other_pop_max = __shfl_down_sync(0xFFFFFFFF, local_pop_max, off);
        if (other_shell_min < local_shell_min) local_shell_min = other_shell_min;
        if (other_shell_max > local_shell_max) local_shell_max = other_shell_max;
        if (other_pop_min < local_pop_min) local_pop_min = other_pop_min;
        if (other_pop_max > local_pop_max) local_pop_max = other_pop_max;
    }

    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->filtered_count, local_filtered);
        atomicAdd((unsigned long long*)&stats->filtered_by_3, local_filt3);
        atomicAdd((unsigned long long*)&stats->filtered_by_5, local_filt5);
        atomicAdd((unsigned long long*)&stats->filtered_by_7, local_filt7);
        atomicAdd((unsigned long long*)&stats->filtered_by_11, local_filt11);
        atomicAdd((unsigned long long*)&stats->filtered_by_13, local_filt13);
        atomicAdd((unsigned long long*)&stats->passed_to_mr, local_passed_mr);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        atomicAdd((unsigned long long*)&stats->mod6_1, local_m1);
        atomicAdd((unsigned long long*)&stats->mod6_5, local_m5);

        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);

        uint64_t old = stats->min_prime;
        while (local_min_prime < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->min_prime, old, local_min_prime);
            if (prev == old) break;
            old = prev;
        }
        old = stats->max_prime;
        while (local_max_prime > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->max_prime, old, local_max_prime);
            if (prev == old) break;
            old = prev;
        }

        if (local_first_theta_key < UINT64_MAX) {
            uint64_t old_key = stats->first_theta_key;
            while (local_first_theta_key < old_key) {
                uint64_t prev = atomicCAS((unsigned long long*)&stats->first_theta_key, old_key, local_first_theta_key);
                if (prev == old_key) {
                    stats->first_theta_prime = local_first_theta_prime;
                    break;
                }
                old_key = prev;
            }
        }

        if (local_last_theta_key > 0) {
            uint64_t old_key = stats->last_theta_key;
            while (local_last_theta_key > old_key) {
                uint64_t prev = atomicCAS((unsigned long long*)&stats->last_theta_key, old_key, local_last_theta_key);
                if (prev == old_key) {
                    stats->last_theta_prime = local_last_theta_prime;
                    break;
                }
                old_key = prev;
            }
        }

        atomicAdd(&stats->shell_sum, local_shell_sum);
        atomicAdd(&stats->popcount_sum, local_pop_sum);
    }
}

// ============================================================================
// Host functions
// ============================================================================

void init_stats(SegmentStats* s, uint64_t seg, uint32_t minS, uint32_t maxS) {
    memset(s, 0, sizeof(SegmentStats));
    s->segment_id = seg;
    s->min_shell = minS;
    s->max_shell = maxS;
    s->seen_min_shell = UINT32_MAX;
    s->seen_max_shell = 0;
    s->min_popcount = UINT32_MAX;
    s->max_popcount = 0;
    s->min_prime = UINT64_MAX;
    s->max_prime = 0;
    s->first_theta_key = UINT64_MAX;
    s->first_theta_prime = 0;
    s->last_theta_key = 0;
    s->last_theta_prime = 0;
}

double perimeter_t_to_degrees(double t) {
    t = fmod(t, 1.0);
    if (t < 0) t += 1.0;
    double R = 1.0;
    double s = t * 8.0 * R;
    double x, y;
    if (s < 2.0 * R) { x = -R + s; y = R; }
    else if (s < 4.0 * R) { x = R; y = R - (s - 2.0 * R); }
    else if (s < 6.0 * R) { x = R - (s - 4.0 * R); y = -R; }
    else { x = -R; y = -R + (s - 6.0 * R); }
    return atan2(y, x) * 180.0 / M_PI;
}

void segment_angle_range(uint64_t segment, uint64_t num_segments, double* start_deg, double* end_deg) {
    double t_start = (double)segment / (double)num_segments;
    double t_end = (double)(segment + 1) / (double)num_segments;
    *start_deg = perimeter_t_to_degrees(t_start);
    *end_deg = perimeter_t_to_degrees(t_end);
}

uint64_t get_theta_key_host(uint64_t prime, uint32_t ref_shell) {
    if (prime < 4) return 0;
    uint32_t shell = 0;
    uint64_t temp = prime;
    while (temp > 1) { temp >>= 1; shell++; }
    if (shell < 2) return 0;
    uint64_t frac = prime ^ (1ULL << shell);
    if (shell <= ref_shell) {
        return frac << (ref_shell - shell);
    } else {
        return frac >> (shell - ref_shell);
    }
}

double theta_key_to_degrees(uint64_t theta_key, uint32_t ref_shell) {
    if (ref_shell < 2) return 0.0;
    uint64_t num_keys = 1ULL << ref_shell;
    double t = (double)theta_key / (double)num_keys;
    return perimeter_t_to_degrees(t);
}

double prime_to_degrees(uint64_t prime, uint32_t ref_shell) {
    if (prime < 4) return 135.0;
    uint64_t theta_key = get_theta_key_host(prime, ref_shell);
    return theta_key_to_degrees(theta_key, ref_shell);
}

void write_csv_header(FILE* fp, int with_angles) {
    fprintf(fp, "segment,shell_config,shell_seen,"
                "total,primes,density,filtered,filt3,filt5,filt7,filt11,filt13,filter_pct,passed_mr,mr_pct,avg_shell,pop_range,avg_pop,"
                "min_prime,max_prime,first_theta_prime,last_theta_prime,"
                "twins,mod6_1,mod6_5,time_ms");
    if (with_angles) {
        fprintf(fp, ",seg_start_deg,seg_end_deg,"
                    "min_prime_deg,max_prime_deg,first_theta_deg,last_theta_deg");
    }
    fprintf(fp, "\n");
}

void write_csv_row(FILE* fp, const SegmentStats* s, int with_angles, uint64_t num_segments, uint32_t ref_shell) {
    double density = s->total_tested ? (double)s->prime_count / s->total_tested : 0;
    double avg_shell = s->total_tested ? s->shell_sum / s->total_tested : 0;
    double avg_pop = s->total_tested ? s->popcount_sum / s->total_tested : 0;
    double filter_pct = s->total_tested ? 100.0 * (double)s->filtered_count / s->total_tested : 0;
    double mr_pct = s->total_tested ? 100.0 * (double)s->passed_to_mr / s->total_tested : 0;

    fprintf(fp, "%" PRIu64 ",%u-%u,%u-%u,"
                "%" PRIu64 ",%" PRIu64 ",%.12f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f,%" PRIu64 ",%.2f,%.2f,%u-%u,%.2f,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f",
            s->segment_id,
            s->min_shell, s->max_shell,
            s->seen_min_shell, s->seen_max_shell,
            s->total_tested, s->prime_count, density,
            s->filtered_count, s->filtered_by_3, s->filtered_by_5, s->filtered_by_7, s->filtered_by_11, s->filtered_by_13,
            filter_pct, s->passed_to_mr, mr_pct, avg_shell,
            s->min_popcount, s->max_popcount, avg_pop,
            s->min_prime, s->max_prime,
            s->first_theta_prime, s->last_theta_prime,
            s->twin_count, s->mod6_1, s->mod6_5, s->time_ms);

    if (with_angles) {
        double seg_start, seg_end;
        segment_angle_range(s->segment_id, num_segments, &seg_start, &seg_end);

        double min_prime_deg = s->min_prime && s->min_prime != UINT64_MAX ?
            prime_to_degrees(s->min_prime, ref_shell) : 0.0;
        double max_prime_deg = s->max_prime ?
            prime_to_degrees(s->max_prime, ref_shell) : 0.0;
        double first_theta_deg = s->first_theta_prime ?
            prime_to_degrees(s->first_theta_prime, ref_shell) : 0.0;
        double last_theta_deg = s->last_theta_prime ?
            prime_to_degrees(s->last_theta_prime, ref_shell) : 0.0;

        fprintf(fp, ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                seg_start, seg_end,
                min_prime_deg, max_prime_deg,
                first_theta_deg, last_theta_deg);
    }
    fprintf(fp, "\n");
}

// V10: Anomaly leaderboard functions
void update_anomaly_leaderboard(AnomalyEntry* top_dense, AnomalyEntry* top_sparse,
                                 int* top_dense_count, int* top_sparse_count,
                                 const SegmentStats* s) {
    if (s->total_tested == 0) return;

    double density = (double)s->prime_count / s->total_tested;
    double filter_rate = 100.0 * (double)s->filtered_count / s->total_tested;

    AnomalyEntry entry;
    entry.segment_id = s->segment_id;
    entry.density = density;
    entry.prime_count = s->prime_count;
    entry.total_tested = s->total_tested;
    entry.filter_rate = filter_rate;

    // Insert into top_dense (highest density first)
    int insert_pos = *top_dense_count;
    for (int i = 0; i < *top_dense_count; i++) {
        if (density > top_dense[i].density) {
            insert_pos = i;
            break;
        }
    }
    if (insert_pos < ANOMALY_LEADERBOARD_SIZE) {
        for (int i = (*top_dense_count < ANOMALY_LEADERBOARD_SIZE ? *top_dense_count : ANOMALY_LEADERBOARD_SIZE - 1); i > insert_pos; i--) {
            top_dense[i] = top_dense[i-1];
        }
        top_dense[insert_pos] = entry;
        if (*top_dense_count < ANOMALY_LEADERBOARD_SIZE) (*top_dense_count)++;
    }

    // Insert into top_sparse (lowest density first)
    insert_pos = *top_sparse_count;
    for (int i = 0; i < *top_sparse_count; i++) {
        if (density < top_sparse[i].density) {
            insert_pos = i;
            break;
        }
    }
    if (insert_pos < ANOMALY_LEADERBOARD_SIZE) {
        for (int i = (*top_sparse_count < ANOMALY_LEADERBOARD_SIZE ? *top_sparse_count : ANOMALY_LEADERBOARD_SIZE - 1); i > insert_pos; i--) {
            top_sparse[i] = top_sparse[i-1];
        }
        top_sparse[insert_pos] = entry;
        if (*top_sparse_count < ANOMALY_LEADERBOARD_SIZE) (*top_sparse_count)++;
    }
}

void write_anomaly_leaderboard(const char* filename, AnomalyEntry* top_dense, AnomalyEntry* top_sparse,
                                int top_dense_count, int top_sparse_count) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "# Anomaly Leaderboard - V10\n");
    fprintf(fp, "# Top %d densest and sparsest segments\n\n", ANOMALY_LEADERBOARD_SIZE);

    fprintf(fp, "type,rank,segment,density,primes,total,filter_rate\n");

    for (int i = 0; i < top_dense_count; i++) {
        fprintf(fp, "dense,%d,%" PRIu64 ",%.12f,%" PRIu64 ",%" PRIu64 ",%.2f\n",
                i + 1, top_dense[i].segment_id, top_dense[i].density,
                top_dense[i].prime_count, top_dense[i].total_tested, top_dense[i].filter_rate);
    }

    for (int i = 0; i < top_sparse_count; i++) {
        fprintf(fp, "sparse,%d,%" PRIu64 ",%.12f,%" PRIu64 ",%" PRIu64 ",%.2f\n",
                i + 1, top_sparse[i].segment_id, top_sparse[i].density,
                top_sparse[i].prime_count, top_sparse[i].total_tested, top_sparse[i].filter_rate);
    }

    fclose(fp);
}

void save_checkpoint(const char* checkpoint_file, uint64_t current_segment, uint64_t total_primes, uint64_t total_tested) {
    FILE* cp = fopen(checkpoint_file, "w");
    if (cp) {
        fprintf(cp, "%" PRIu64 "\n%" PRIu64 "\n%" PRIu64 "\n", current_segment, total_primes, total_tested);
        fclose(cp);
    }
}

int load_checkpoint(const char* checkpoint_file, uint64_t* start_segment, uint64_t* total_primes, uint64_t* total_tested) {
    FILE* cp = fopen(checkpoint_file, "r");
    if (!cp) return 0;
    if (fscanf(cp, "%" SCNu64 "\n%" SCNu64 "\n%" SCNu64, start_segment, total_primes, total_tested) != 3) {
        fclose(cp);
        return 0;
    }
    fclose(cp);
    return 1;
}

void print_usage(const char* prog) {
    printf("theta_order_complete_v10 - Extended Filters + Anomaly Tracking\n");
    printf("===============================================================\n\n");
    printf("V10 changes over V9:\n");
    printf("  - FEATURE: Added mod 13 filter (12-bit chunks, 6 iterations)\n");
    printf("  - FEATURE: Track passed_to_mr count (filter efficiency metric)\n");
    printf("  - FEATURE: Anomaly leaderboard CSV output\n\n");
    printf("Usage: %s <mode> [options]\n\n", prog);

    printf("MODES:\n");
    printf("  scan32    Full 32-bit theta scan (shells 2-31)\n");
    printf("  scan64    Full 64-bit theta scan (shells 2-62)\n");
    printf("  zoom64    Zoom into segment range\n");
    printf("  batch     Batch mode with checkpoints\n\n");

    printf("OPTIONS:\n");
    printf("  --min-shell S       Minimum shell (default: 2)\n");
    printf("  --max-shell S       Maximum shell\n");
    printf("  --segment-exp E     Number of segments = 2^E (default: 10)\n");
    printf("  --start-seg N       Starting segment\n");
    printf("  --end-seg N         Ending segment\n");
    printf("  --output FILE       Output CSV file\n");
    printf("  --anomaly FILE      Anomaly leaderboard CSV file (default: anomaly_v10.csv)\n");
    printf("  --checkpoint FILE   Checkpoint file\n");
    printf("  --angles            Add angle columns to CSV\n");
    printf("  --debug             Show debug info\n");
    printf("  -q                  Quiet mode\n");
    printf("  -v                  Verbose mode\n\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    if (!strcmp(mode, "-h") || !strcmp(mode, "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    int use_64bit = 0;
    if (!strcmp(mode, "scan64") || !strcmp(mode, "zoom64")) {
        use_64bit = 1;
    }

    uint32_t min_shell = 2;
    uint32_t max_shell = use_64bit ? 62 : 31;
    uint32_t segment_exp = 10;
    uint64_t num_segments = 0;
    uint64_t start_segment = 0;
    uint64_t end_segment = 0;
    const char* outfile = use_64bit ? "theta_angular_64_v10.csv" : "theta_angular_32_v10.csv";
    const char* anomaly_file = "anomaly_v10.csv";
    const char* checkpoint_file = NULL;
    int verbose = 1;
    int with_angles = 0;
    int debug_mode = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--min-shell") && i+1 < argc) min_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-shell") && i+1 < argc) max_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--segment-exp") && i+1 < argc) segment_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--segments") && i+1 < argc) {
            num_segments = strtoull(argv[++i], NULL, 0);
            segment_exp = 0;
            uint64_t tmp = num_segments;
            while (tmp > 1) { tmp >>= 1; segment_exp++; }
        }
        else if (!strcmp(argv[i], "--start-seg") && i+1 < argc) start_segment = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--end-seg") && i+1 < argc) end_segment = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "--anomaly") && i+1 < argc) anomaly_file = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint") && i+1 < argc) checkpoint_file = argv[++i];
        else if (!strcmp(argv[i], "--angles")) with_angles = 1;
        else if (!strcmp(argv[i], "-q")) verbose = 0;
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--debug")) debug_mode = 1;
    }

    if (num_segments == 0) {
        num_segments = 1ULL << segment_exp;
    }

    if (use_64bit) {
        if (max_shell > 62) max_shell = 62;
    } else {
        if (max_shell > 31) max_shell = 31;
    }
    if (min_shell < 2) min_shell = 2;
    if (min_shell > max_shell) {
        fprintf(stderr, "Error: min_shell > max_shell\n");
        return 1;
    }

    if (end_segment == 0 || end_segment > num_segments) {
        end_segment = num_segments;
    }

    uint32_t ref_shell = max_shell;

    uint64_t grand_total_primes = 0;
    uint64_t grand_total_tested = 0;
    uint64_t grand_total_filtered = 0;
    uint64_t grand_total_passed_mr = 0;
    uint64_t grand_filt3 = 0, grand_filt5 = 0, grand_filt7 = 0, grand_filt11 = 0, grand_filt13 = 0;

    // V10: Anomaly leaderboards
    AnomalyEntry top_dense[ANOMALY_LEADERBOARD_SIZE];
    AnomalyEntry top_sparse[ANOMALY_LEADERBOARD_SIZE];
    int top_dense_count = 0;
    int top_sparse_count = 0;

    if (checkpoint_file && !strcmp(mode, "batch")) {
        uint64_t cp_seg, cp_primes, cp_tested;
        if (load_checkpoint(checkpoint_file, &cp_seg, &cp_primes, &cp_tested)) {
            start_segment = cp_seg;
            grand_total_primes = cp_primes;
            grand_total_tested = cp_tested;
            if (verbose) {
                printf("Resuming from checkpoint: segment %" PRIu64 "\n", start_segment);
            }
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (verbose) {
        printf("=== V10: %s Mode ===\n", mode);
        printf("Precision: %d-bit\n", use_64bit ? 64 : 32);
        printf("Shells: %u to %u (ref_shell=%u)\n", min_shell, max_shell, ref_shell);
        printf("Segments: %" PRIu64 " (2^%u)\n", num_segments, segment_exp);
        printf("Segment range: %" PRIu64 " to %" PRIu64 "\n", start_segment, end_segment - 1);
        printf("Output: %s\n", outfile);
        printf("Anomaly: %s\n", anomaly_file);
        printf("Filters: mod3 + mod5 + mod7(9-bit) + mod11(10-bit) + mod13(12-bit)\n");

        uint64_t total_odds = 0;
        for (uint32_t s = min_shell; s <= max_shell; s++) {
            total_odds += 1ULL << (s - 1);
        }
        printf("Total odd integers: ~%" PRIu64 " (%.2e)\n", total_odds, (double)total_odds);
        printf("\n");
    }

    cudaSetDevice(0);

    SegmentStats* d_stats;
    cudaMalloc(&d_stats, sizeof(SegmentStats));
    SegmentStats h_stats;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    FILE* fp = fopen(outfile, start_segment > 0 ? "a" : "w");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", outfile);
        return 1;
    }
    if (start_segment == 0) write_csv_header(fp, with_angles);

    clock_t wall_start = clock();
    uint64_t checkpoint_interval = 64;

    for (uint64_t seg = start_segment; seg < end_segment && !shutdown_requested; seg++) {
        init_stats(&h_stats, seg, min_shell, max_shell);
        cudaMemcpy(d_stats, &h_stats, sizeof(SegmentStats), cudaMemcpyHostToDevice);

        cudaEventRecord(t0);

        uint64_t work_estimate = 1ULL << (max_shell - 1);
        int blocks = (int)((work_estimate + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        if (blocks > 65535) blocks = 65535;
        if (blocks < 256) blocks = 256;

        if (debug_mode && seg == start_segment) {
            printf("\n=== DEBUG: V10 Kernel launch ===\n");
            printf("  Blocks: %d, Threads: %d\n", blocks, THREADS_PER_BLOCK);
            printf("  Filters: mod3 + mod5 + mod7(9-bit) + mod11(10-bit) + mod13(12-bit)\n\n");
        }

        if (use_64bit) {
            angular_segment_kernel_64<<<blocks, THREADS_PER_BLOCK>>>(
                seg, num_segments, segment_exp, min_shell, max_shell, ref_shell, d_stats);
        } else {
            angular_segment_kernel_32<<<blocks, THREADS_PER_BLOCK>>>(
                (uint32_t)seg, num_segments, segment_exp, min_shell, max_shell, ref_shell, d_stats);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
            break;
        }
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(&h_stats, d_stats, sizeof(SegmentStats), cudaMemcpyDeviceToHost);
        h_stats.time_ms = ms;

        if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
        if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
        if (h_stats.min_prime == UINT64_MAX) h_stats.min_prime = 0;
        if (h_stats.first_theta_key == UINT64_MAX) h_stats.first_theta_prime = 0;

        write_csv_row(fp, &h_stats, with_angles, num_segments, ref_shell);
        fflush(fp);

        // V10: Update anomaly leaderboard
        update_anomaly_leaderboard(top_dense, top_sparse, &top_dense_count, &top_sparse_count, &h_stats);

        grand_total_primes += h_stats.prime_count;
        grand_total_tested += h_stats.total_tested;
        grand_total_filtered += h_stats.filtered_count;
        grand_total_passed_mr += h_stats.passed_to_mr;
        grand_filt3 += h_stats.filtered_by_3;
        grand_filt5 += h_stats.filtered_by_5;
        grand_filt7 += h_stats.filtered_by_7;
        grand_filt11 += h_stats.filtered_by_11;
        grand_filt13 += h_stats.filtered_by_13;

        if (checkpoint_file && (seg % checkpoint_interval == 0 || seg == end_segment - 1)) {
            save_checkpoint(checkpoint_file, seg + 1, grand_total_primes, grand_total_tested);
            // Also update anomaly file at checkpoints
            write_anomaly_leaderboard(anomaly_file, top_dense, top_sparse, top_dense_count, top_sparse_count);
        }

        if (verbose >= 2 || (verbose == 1 && (seg % 64 == 0 || seg == end_segment - 1))) {
            double density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
            double filter_pct = h_stats.total_tested ? 100.0 * (double)h_stats.filtered_count / h_stats.total_tested : 0;
            printf("Seg %4" PRIu64 "/%" PRIu64 ": primes=%8" PRIu64 " d=%.6f filt=%.1f%% (3:%" PRIu64 " 5:%" PRIu64 " 7:%" PRIu64 " 11:%" PRIu64 " 13:%" PRIu64 ") %.0fms\n",
                   seg, num_segments,
                   h_stats.prime_count, density, filter_pct,
                   h_stats.filtered_by_3, h_stats.filtered_by_5, h_stats.filtered_by_7, h_stats.filtered_by_11, h_stats.filtered_by_13, ms);
        }
    }

    double wall_time = (double)(clock() - wall_start) / CLOCKS_PER_SEC;

    // V10: Write final anomaly leaderboard
    write_anomaly_leaderboard(anomaly_file, top_dense, top_sparse, top_dense_count, top_sparse_count);

    if (verbose) {
        printf("\n=== SUMMARY ===\n");
        printf("Total tested: %" PRIu64 "\n", grand_total_tested);
        printf("Total primes: %" PRIu64 "\n", grand_total_primes);
        printf("Fast-filtered: %" PRIu64 " (%.1f%%)\n",
               grand_total_filtered,
               grand_total_tested ? 100.0 * (double)grand_total_filtered / grand_total_tested : 0);
        printf("  - mod 3:  %" PRIu64 " (%.1f%%)\n", grand_filt3,
               grand_total_tested ? 100.0 * (double)grand_filt3 / grand_total_tested : 0);
        printf("  - mod 5:  %" PRIu64 " (%.1f%%)\n", grand_filt5,
               grand_total_tested ? 100.0 * (double)grand_filt5 / grand_total_tested : 0);
        printf("  - mod 7:  %" PRIu64 " (%.1f%%)\n", grand_filt7,
               grand_total_tested ? 100.0 * (double)grand_filt7 / grand_total_tested : 0);
        printf("  - mod 11: %" PRIu64 " (%.1f%%)\n", grand_filt11,
               grand_total_tested ? 100.0 * (double)grand_filt11 / grand_total_tested : 0);
        printf("  - mod 13: %" PRIu64 " (%.1f%%)\n", grand_filt13,
               grand_total_tested ? 100.0 * (double)grand_filt13 / grand_total_tested : 0);
        printf("Passed to MR: %" PRIu64 " (%.1f%%)\n", grand_total_passed_mr,
               grand_total_tested ? 100.0 * (double)grand_total_passed_mr / grand_total_tested : 0);
        printf("Wall time: %.2f seconds\n", wall_time);
        printf("Output: %s\n", outfile);
        printf("Anomaly leaderboard: %s\n", anomaly_file);
    }

    fclose(fp);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_stats);

    return shutdown_requested ? 1 : 0;
}
