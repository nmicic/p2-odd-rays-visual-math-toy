#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Nenad Micic <nenad@micic.be>
#
# theta_order_batch_v10.sh - Batch runner for V10
#
# V10 features:
#   - mod 13 filter (12-bit chunks)
#   - Filter-pass-rate tracking
#   - Anomaly leaderboard output
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/theta_order_complete_v10"

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Compiling..."
    cd "$SCRIPT_DIR"
    nvcc -O3 -arch=sm_86 theta_order_complete_v10.cu -o theta_order_complete_v10
    if [ $? -ne 0 ]; then
        echo "Compilation failed!"
        exit 1
    fi
    echo "Compilation successful."
fi

# Default parameters
MODE="${1:-scan32}"
SEGMENT_EXP="${2:-8}"
MAX_SHELL="${3:-28}"
OUTPUT_DIR="${SCRIPT_DIR}/theta_results_v10"

mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    scan32)
        echo "=== V10: 32-bit scan ==="
        echo "Segments: 2^$SEGMENT_EXP = $((1 << SEGMENT_EXP))"
        echo "Shells: 2 to $MAX_SHELL"
        echo "Output: $OUTPUT_DIR/theta_angular_32_v10.csv"
        echo "Anomaly: $OUTPUT_DIR/anomaly_32_v10.csv"
        echo ""
        $BINARY scan32 \
            --segment-exp $SEGMENT_EXP \
            --max-shell $MAX_SHELL \
            --output "$OUTPUT_DIR/theta_angular_32_v10.csv" \
            --anomaly "$OUTPUT_DIR/anomaly_32_v10.csv" \
            --angles
        ;;

    scan64)
        echo "=== V10: 64-bit scan ==="
        echo "Segments: 2^$SEGMENT_EXP = $((1 << SEGMENT_EXP))"
        echo "Shells: 2 to $MAX_SHELL"
        echo "Output: $OUTPUT_DIR/theta_angular_64_v10.csv"
        echo "Anomaly: $OUTPUT_DIR/anomaly_64_v10.csv"
        echo ""
        $BINARY scan64 \
            --segment-exp $SEGMENT_EXP \
            --max-shell $MAX_SHELL \
            --output "$OUTPUT_DIR/theta_angular_64_v10.csv" \
            --anomaly "$OUTPUT_DIR/anomaly_64_v10.csv" \
            --angles
        ;;

    zoom64)
        START_SEG="${4:-0}"
        END_SEG="${5:-16}"
        echo "=== V10: 64-bit zoom ==="
        echo "Segments: $START_SEG to $END_SEG"
        echo "Shells: 2 to $MAX_SHELL"
        echo ""
        $BINARY zoom64 \
            --segment-exp $SEGMENT_EXP \
            --max-shell $MAX_SHELL \
            --start-seg $START_SEG \
            --end-seg $END_SEG \
            --output "$OUTPUT_DIR/theta_zoom_64_v10.csv" \
            --anomaly "$OUTPUT_DIR/anomaly_zoom_v10.csv" \
            --angles -v
        ;;

    batch)
        echo "=== V10: Batch mode with checkpoints ==="
        echo "Segments: 2^$SEGMENT_EXP = $((1 << SEGMENT_EXP))"
        echo "Shells: 2 to $MAX_SHELL"
        echo "Checkpoint: $OUTPUT_DIR/v10.ckpt"
        echo ""
        $BINARY batch \
            --segment-exp $SEGMENT_EXP \
            --max-shell $MAX_SHELL \
            --output "$OUTPUT_DIR/theta_batch_v10.csv" \
            --anomaly "$OUTPUT_DIR/anomaly_batch_v10.csv" \
            --checkpoint "$OUTPUT_DIR/v10.ckpt" \
            --angles
        ;;

    deep)
        # Deep scan: high shell range, many segments
        DEEP_SHELL="${4:-40}"
        echo "=== V10: Deep 64-bit scan ==="
        echo "Segments: 2^$SEGMENT_EXP = $((1 << SEGMENT_EXP))"
        echo "Shells: 2 to $DEEP_SHELL"
        echo "This will take a long time..."
        echo ""
        $BINARY scan64 \
            --segment-exp $SEGMENT_EXP \
            --max-shell $DEEP_SHELL \
            --output "$OUTPUT_DIR/theta_deep_64_v10.csv" \
            --anomaly "$OUTPUT_DIR/anomaly_deep_v10.csv" \
            --checkpoint "$OUTPUT_DIR/v10_deep.ckpt" \
            --angles
        ;;

    compare)
        # Compare V9 vs V10
        echo "=== Comparing V9 vs V10 ==="
        echo "Running V9..."
        "${SCRIPT_DIR}/theta_order_complete_v9" scan32 \
            --segment-exp 8 --max-shell 28 \
            --output "$OUTPUT_DIR/compare_v9.csv" -q

        echo "Running V10..."
        $BINARY scan32 \
            --segment-exp 8 --max-shell 28 \
            --output "$OUTPUT_DIR/compare_v10.csv" -q

        echo ""
        echo "Comparing prime counts..."
        V9_PRIMES=$(awk -F, 'NR>1 {sum+=$5} END {print sum}' "$OUTPUT_DIR/compare_v9.csv")
        V10_PRIMES=$(awk -F, 'NR>1 {sum+=$5} END {print sum}' "$OUTPUT_DIR/compare_v10.csv")
        echo "V9 primes:  $V9_PRIMES"
        echo "V10 primes: $V10_PRIMES"
        if [ "$V9_PRIMES" == "$V10_PRIMES" ]; then
            echo "✓ Prime counts match!"
        else
            echo "✗ Prime counts DIFFER!"
        fi

        echo ""
        echo "Comparing filter rates..."
        V9_FILT=$(awk -F, 'NR>1 {sum+=$7} END {print sum}' "$OUTPUT_DIR/compare_v9.csv")
        V10_FILT=$(awk -F, 'NR>1 {sum+=$7} END {print sum}' "$OUTPUT_DIR/compare_v10.csv")
        echo "V9 filtered:  $V9_FILT"
        echo "V10 filtered: $V10_FILT"
        echo "V10 filters $((V10_FILT - V9_FILT)) more composites (mod 13 contribution)"
        ;;

    help|--help|-h)
        echo "Usage: $0 <mode> [segment_exp] [max_shell] [extra...]"
        echo ""
        echo "Modes:"
        echo "  scan32           32-bit full scan"
        echo "  scan64           64-bit full scan"
        echo "  zoom64 S E       64-bit zoom (segments S to E)"
        echo "  batch            Batch mode with checkpoints"
        echo "  deep [shell]     Deep 64-bit scan to specified shell"
        echo "  compare          Compare V9 vs V10 results"
        echo ""
        echo "Examples:"
        echo "  $0 scan32 8 28           # 32-bit, 256 segments, shells 2-28"
        echo "  $0 scan64 10 40          # 64-bit, 1024 segments, shells 2-40"
        echo "  $0 zoom64 8 40 0 16      # 64-bit zoom, segments 0-16"
        echo "  $0 deep 10 50            # Deep scan to shell 50"
        echo "  $0 compare               # Compare V9 vs V10"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Use '$0 help' for usage"
        exit 1
        ;;
esac
