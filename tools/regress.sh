#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# ds-ocr 精度回归（任务书 docs/intent/ds-ocr-acc-perf.md 任务 1）
# 用法: bash tools/regress.sh            → 跑全部 16 组合并打印判定
#       bash tools/regress.sh <模型子串>  → 只跑匹配的模型（调试用）
#
# ⚠️ 冻结区 —— 改 tools/thresholds.txt 的阈值、或替换黄金标签让失败变绿 = 作弊
#    （阈值 = 冻结时基线 CER + 1pt，冻结于 2026-10-30；变更须领导批准并记 PROGRESS.md）
# ═══════════════════════════════════════════════════════════════════
set -u
cd "$(dirname "$0")/.."

MODELS="DeepSeek-OCR
DeepSeek-OCR-2
Unlimited-OCR
Unlimited-OCR-2"
IMAGES="test截屏.png
test_simple.png
test_crop_640.png
crop_real.png
r2_dense.png
r2_action.png
r2_list.png
r2_quote.png"
GOLD_DIR=docs/golden
OUT_DIR=/tmp/regress
mkdir -p "$OUT_DIR"

# 每图 token 上限（截图类长文档 700，小图 200；同时防退化循环拖死回归）
cap_for() {
    case "$1" in
        test截屏.png) echo 700 ;;
        *) echo 200 ;;
    esac
}

# 冻结阈值查表（tools/thresholds.txt: "键 阈值"，键 = 模型_图不带扩展名）
threshold_for() {
    v=$(awk -v k="$1" '$1 == k { print $2; exit }' tools/thresholds.txt 2>/dev/null)
    echo "${v:-3.0}"
}

FILTER="${1:-}"
PASS=0; FAIL=0
printf "%-18s %-18s %8s %10s  %s\n" "模型" "图" "CER%" "阈值%" "判定"
printf "%.0s-" {1..70}; echo

for m in $MODELS; do
    if [ -n "$FILTER" ] && [ "${m#*$FILTER}" = "$m" ]; then continue; fi
    for img in $IMAGES; do
        key="${m}_${img%.*}"
        if [ -f tools/exclude.txt ] && grep -q "^${key}$" tools/exclude.txt 2>/dev/null; then
            printf "%-18s %-18s %8s %10s  %s\n" "$m" "$img" "-" "-" "SKIP(排除)"; continue
        fi
        gold="$GOLD_DIR/labels/${m}_${img%.*}.txt"
        if [ ! -f "$gold" ]; then echo "MISSING GOLDEN: $gold"; FAIL=$((FAIL+1)); continue; fi
        cap=$(cap_for "$img")
        out="$OUT_DIR/${m}_${img%.*}.txt"
        ./ds_ocr -d "./models/$m" -i "$GOLD_DIR/images/$img" -n "$cap" --silent \
            2>"$OUT_DIR/${m}_${img%.*}.log" > "$out"
        rc=$?
        cer=$(python3 - "$gold" "$out" <<'PY'
import difflib, sys
# 只剥 provenance 头（"# source:" 开头）—— 不能按 "#" 剥：V1/V2 输出是 Markdown，
# 正文 "# Description" 等标题行是真实内容（2026-10-30 教训：剥 "#" 造成假 CER 18%）
g = "".join(l for l in open(sys.argv[1]) if not l.startswith("# source:"))
a = open(sys.argv[2]).read()
sm = difflib.SequenceMatcher(None, g, a)
print(f"{(1 - sm.ratio()) * 100:.2f}")
PY
)
        if [ $rc -ne 0 ]; then
            printf "%-18s %-18s %8s %10s  %s\n" "$m" "$img" "ERR(rc=$rc)" "-" "FAIL(崩溃)"; FAIL=$((FAIL+1)); continue
        fi
        th=$(threshold_for "$key")
        ok=$(python3 -c "print(1 if $cer <= $th else 0)")
        if [ "$ok" = 1 ]; then
            printf "%-18s %-18s %8s %10s  %s\n" "$m" "$img" "$cer" "$th" "PASS"; PASS=$((PASS+1))
        else
            printf "%-18s %-18s %8s %10s  %s\n" "$m" "$img" "$cer" "$th" "FAIL"; FAIL=$((FAIL+1))
        fi
    done
done

printf "%.0s-" {1..70}; echo
echo "合计: $PASS PASS / $FAIL FAIL"
if [ $FAIL -eq 0 ] && [ $PASS -gt 0 ]; then echo "REGRESSION: ALL GREEN ($PASS/30)"; exit 0; else echo "REGRESSION: RED"; exit 1; fi
