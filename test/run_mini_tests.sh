#!/bin/bash
# impopk v0.1 — Mini integration test suite
# Run from project root: bash test/run_mini_tests.sh
set -e

BIN=./target/release
PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=========================================="
echo "  impopk v0.1 — Mini Test Suite"
echo "=========================================="
echo ""

# --- Test 1: All 9 binaries ---
echo "=== 1. Binary verification ==="
for bin in ibs ibs-from-paf ibs-from-tpa tpa-spatial-index tpa-validate ibd ibd-validate ancestry jacquard; do
    $BIN/$bin --help >/dev/null 2>&1 && ok "$bin --help" || fail "$bin --help"
done
echo ""

# --- Test 2: Ancestry from pre-computed IBS ---
echo "=== 2. Ancestry inference ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --estimate-params --min-posterior 0.5 \
    -o test/out_ancestry.tsv 2>/dev/null

if [ -s test/out_ancestry.tsv ]; then
    SEGS=$(tail -n+2 test/out_ancestry.tsv | wc -l)
    ok "ancestry produced $SEGS segment(s)"
else
    fail "ancestry produced no output"
fi
echo ""

# --- Test 3: IBD from pre-computed IBS ---
echo "=== 3. IBD detection ==="
$BIN/ibd \
    --similarity-file test/ibs_paf_5Mb_EUR.tsv \
    --output test/out_ibd.tsv \
    --identity-floor 0.9 --min-lod 3.0 \
    --baum-welch-iters 10 2>/dev/null

if [ -f test/out_ibd.tsv ]; then
    SEGS=$(tail -n+2 test/out_ibd.tsv | wc -l)
    ok "ibd produced $SEGS segment(s)"
else
    fail "ibd produced no output"
fi
echo ""

# --- Test 4: tpa-validate self-check ---
echo "=== 4. tpa-validate ==="
RESULT=$($BIN/tpa-validate \
    --reference test/ibs_paf_1Mb_EUR.tsv \
    --test test/ibs_paf_1Mb_EUR.tsv 2>/dev/null | grep "VERDICT")

if echo "$RESULT" | grep -q "PASS"; then
    ok "tpa-validate self-check"
else
    fail "tpa-validate: $RESULT"
fi
echo ""

# --- Test 5: Cargo tests ---
echo "=== 5. Workspace tests ==="
TEST_OUT=$(cargo test --workspace 2>&1)
TOTAL=$(echo "$TEST_OUT" | grep -oP '\d+ passed' | awk '{sum+=$1} END {print sum}')
FAILED=$(echo "$TEST_OUT" | grep -oP '\d+ failed' | awk '{sum+=$1} END {print sum+0}')
if [ "$FAILED" = "0" ]; then
    ok "$TOTAL unit tests passed"
else
    fail "$FAILED test failures"
fi
echo ""

# --- Test 6: Clippy ---
echo "=== 6. Clippy ==="
if cargo clippy --workspace -- -D warnings 2>&1 | grep -q "Finished"; then
    ok "clippy clean"
else
    fail "clippy warnings"
fi
echo ""

# --- Summary ---
echo "=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="

# Clean outputs
rm -f test/out_ancestry.tsv test/out_ibd.tsv

exit $FAIL
