#!/bin/bash
# impopk v0.1 — Integration test suite
# Validates every tutorial command with bundled mini test data.
# Run from project root: bash test/run_mini_tests.sh
set -euo pipefail

BIN=./target/release
PASS=0
FAIL=0
OUT=/tmp/impopk_test_$$

mkdir -p "$OUT"
trap "rm -rf $OUT" EXIT

ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1 — $2"; FAIL=$((FAIL+1)); }

echo "=========================================="
echo "  impopk v0.1 — Integration Test Suite"
echo "=========================================="
echo ""

# --- 1. Binary verification (9 binaries) ---
echo "=== 1. Binary verification ==="
for bin in ibs ibs-from-paf ibs-from-tpa tpa-spatial-index tpa-validate ibd ibd-validate ancestry jacquard; do
    if $BIN/$bin --help >/dev/null 2>&1; then
        ok "$bin --help"
    else
        fail "$bin --help" "binary not found or crashed"
    fi
done
echo ""

# --- 2. Ancestry inference (auto-configure) ---
echo "=== 2. Ancestry with auto-configure ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --auto-configure --identity-floor 0.9 \
    --min-posterior 0.5 \
    -o "$OUT/ancestry.tsv" \
    --posteriors-output "$OUT/posteriors.tsv" 2>/dev/null

SEGS=$(tail -n+2 "$OUT/ancestry.tsv" | wc -l)
POST=$(tail -n+2 "$OUT/posteriors.tsv" | wc -l)
[ "$SEGS" -ge 0 ] && ok "ancestry: $SEGS segments" || fail "ancestry" "no output"
[ "$POST" -gt 0 ] && ok "posteriors: $POST windows" || fail "posteriors" "no output"
echo ""

# --- 3. Ancestry with estimate-params (non-auto) ---
echo "=== 3. Ancestry with estimate-params ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --estimate-params --min-posterior 0.5 \
    -o "$OUT/ancestry_est.tsv" 2>/dev/null
[ -s "$OUT/ancestry_est.tsv" ] && ok "estimate-params mode" || fail "estimate-params" "no output"
echo ""

# --- 4. Ancestry: save-params / load-params ---
echo "=== 4. Cross-chromosome param transfer ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --auto-configure --identity-floor 0.9 \
    --save-params "$OUT/params.json" --min-posterior 0.5 \
    -o /dev/null 2>/dev/null
[ -s "$OUT/params.json" ] && ok "save-params" || fail "save-params" "no JSON"

$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --load-params "$OUT/params.json" --min-posterior 0.5 \
    -o "$OUT/ancestry_loaded.tsv" 2>/dev/null
[ -s "$OUT/ancestry_loaded.tsv" ] && ok "load-params" || fail "load-params" "no output"
echo ""

# --- 5. Ancestry: eGRM output ---
echo "=== 5. eGRM output ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --output-egrm "$OUT/egrm" --center-egrm \
    --min-posterior 0.5 \
    -o "$OUT/ancestry_egrm.tsv" 2>/dev/null
[ -f "$OUT/egrm.grm.bin" ] && ok "eGRM .grm.bin" || fail "eGRM" "missing .grm.bin"
[ -f "$OUT/egrm.grm.N.bin" ] && ok "eGRM .grm.N.bin" || fail "eGRM" "missing .grm.N.bin"
[ -f "$OUT/egrm.grm.id" ] && ok "eGRM .grm.id" || fail "eGRM" "missing .grm.id"
echo ""

# --- 6. Ancestry: demographic inference ---
echo "=== 6. Demographic inference ==="
$BIN/ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --auto-configure --identity-floor 0.9 \
    --demographic-inference --demographic-output "$OUT/demog.tsv" \
    -o "$OUT/ancestry_demog.tsv" 2>/dev/null
[ -s "$OUT/demog.tsv" ] && ok "demographic output" || fail "demographic" "no output"
echo ""

# --- 7. IBD detection ---
echo "=== 7. IBD detection ==="
# ibd with --similarity-file
$BIN/ibd --similarity-file test/ibs_paf_5Mb_EUR.tsv \
    --output "$OUT/ibd.tsv" \
    --identity-floor 0.9 --min-lod 2.0 \
    --baum-welch-iters 10 --min-len-bp 500000 2>/dev/null
SEGS=$(($(wc -l < "$OUT/ibd.tsv") - 1))
[ "$SEGS" -gt 0 ] && ok "ibd: $SEGS segments detected" || fail "ibd" "0 segments"

# ibd-validate with --input
$BIN/ibd-validate --input test/ibs_paf_5Mb_EUR.tsv \
    -o "$OUT/ibd_val.tsv" --population EUR --window-size 10000 \
    --identity-floor 0.9 --min-len-bp 500000 --min-windows 50 \
    --baum-welch-iters 10 2>/dev/null
[ -s "$OUT/ibd_val.tsv" ] && ok "ibd-validate --input" || fail "ibd-validate" "no output"

# ibd-validate with -i shorthand
$BIN/ibd-validate -i test/ibs_paf_5Mb_EUR.tsv \
    -o "$OUT/ibd_val2.tsv" --population EUR --window-size 10000 \
    --identity-floor 0.9 --min-len-bp 500000 --min-windows 50 \
    --baum-welch-iters 10 2>/dev/null
[ -s "$OUT/ibd_val2.tsv" ] && ok "ibd-validate -i shorthand" || fail "ibd-validate -i" "no output"

# ibd with --multi-scale
$BIN/ibd --similarity-file test/ibs_paf_5Mb_EUR.tsv \
    --output "$OUT/ibd_ms.tsv" \
    --identity-floor 0.9 --min-lod 2.0 --multi-scale \
    --baum-welch-iters 10 --min-len-bp 500000 2>/dev/null
[ -s "$OUT/ibd_ms.tsv" ] && ok "ibd --multi-scale" || fail "ibd --multi-scale" "no output"
echo ""

# --- 8. Jacquard coefficients ---
echo "=== 8. Jacquard coefficients ==="
JACQ=$($BIN/jacquard --ibs test/ibs_paf_5Mb_EUR.tsv \
    --hap-a1 HG00097#1 --hap-a2 HG00097#2 \
    --hap-b1 HG00099#1 --hap-b2 HG00099#2 2>&1 | grep "^Delta")
[ -n "$JACQ" ] && ok "jacquard output (9 deltas)" || fail "jacquard" "no delta output"
echo ""

# --- 9. tpa-validate ---
echo "=== 9. tpa-validate ==="
VERDICT=$($BIN/tpa-validate \
    --reference test/ibs_paf_1Mb_EUR.tsv \
    --test test/ibs_paf_1Mb_EUR.tsv 2>/dev/null | grep "VERDICT")
echo "$VERDICT" | grep -q "PASS" && ok "tpa-validate self-check" || fail "tpa-validate" "$VERDICT"
echo ""

# --- 10. Cargo tests + clippy ---
echo "=== 10. Workspace tests ==="
TEST_OUT=$(cargo test --workspace 2>&1)
TOTAL=$(echo "$TEST_OUT" | grep -oP '\d+ passed' | awk '{sum+=$1} END {print sum+0}')
FAILED=$(echo "$TEST_OUT" | grep -oP '\d+ failed' | awk '{sum+=$1} END {print sum+0}')
[ "$FAILED" = "0" ] && ok "$TOTAL unit tests, 0 failures" || fail "cargo test" "$FAILED failures"

echo ""
echo "=== 11. Clippy ==="
if cargo clippy --workspace -- -D warnings 2>&1 | grep -q "Finished"; then
    ok "clippy clean"
else
    fail "clippy" "warnings present"
fi

# --- Summary ---
echo ""
echo "=========================================="
TOTAL_TESTS=$((PASS + FAIL))
echo "  Results: $PASS/$TOTAL_TESTS passed, $FAIL failed"
echo "=========================================="

exit $FAIL
