#!/usr/bin/env bash
# ============================================================
# Layer 2 · 降采样重跑 —— RE1-OB × {k=1,2,4} × {Granger, PC, PCMCI}
#
# 用法（在 WSL 里）:
#   bash run_layer2_ob.sh                 # 三档 × 三方法，全跑（约 2.5 h）
#   bash run_layer2_ob.sh 2               # 只跑 k=2（约 50 min）
#   bash run_layer2_ob.sh 2 granger       # 只跑 k=2 的 Granger（约 8 min）
#   bash run_layer2_ob.sh "1 2" "pc"      # 跑 k=1,2 的 PC
#
# 说明:
#   - 窗口参数按 k 自动换算：k=1→20, k=2→10, k=4→5（保证每段物理时长都是 10 分钟）
#   - 每个方法单独存日志，屏幕只显示尾部，避免刷屏
#   - 任一方法失败不会中断后续步骤，结束后统一提示
#
# 可用环境变量覆盖（用于跑全量等其他数据目录）:
#   DSROOT=~/RCAEval/data_ds_full OUT=/mnt/d/北交威/大创/结果/降采样实验_全量 \
#     bash run_layer2_ob.sh "1 2 4" "granger pc"
# ============================================================
set -u

PY=$HOME/RCAEval/env/bin/python
REPO=/mnt/d/北交威/大创
DSROOT=${DSROOT:-$HOME/RCAEval/data_ds}
OUT=${OUT:-$REPO/结果/降采样实验}
GR=$REPO/结果/Granger/run_granger.py
FIX=$REPO/pc_pcmci_fixed/run_rca_full_ob.py

KS=${1:-"1 2 4"}
METHODS=${2:-"granger pc pcmci"}

echo "============================================================"
echo "Layer 2 降采样重跑 (RE1-OB)"
echo "  档位:   $KS"
echo "  方法:   $METHODS"
echo "  输出:   $OUT"
echo "============================================================"

# ---------------- 前置检查 ----------------
[ -x "$PY" ]  || { echo "❌ 找不到可执行的 python: $PY"; exit 1; }
[ -f "$GR" ]  || { echo "❌ 找不到 $GR"; exit 1; }
[ -f "$FIX" ] || { echo "❌ 找不到 $FIX"; exit 1; }

for k in $KS; do
  d=$DSROOT/ds$k/RE1-OB
  n=$(find "$d" -name data.csv 2>/dev/null | wc -l)
  printf "  ds%-2s %-45s %s 个案例\n" "$k" "$d" "$n"
  [ "$n" -gt 0 ] || { echo "❌ ds$k 没有数据。先跑: make_downsampled.py --k $k --per-fault 1"; exit 1; }
done

FAIL=""
T0=$(date +%s)

# ---------------- 主循环 ----------------
for k in $KS; do
  WIN=$((20 / k))
  DS=$DSROOT/ds$k/RE1-OB
  OUTK=$OUT/ds$k
  mkdir -p "$OUTK"

  echo ""
  echo "############################################################"
  echo "# ds$k   (窗口参数 --length / --window = $WIN)"
  echo "############################################################"

  for m in $METHODS; do
    case "$m" in
      granger)
        echo "--- [ds$k] Granger ---"
        $PY "$GR" --data-root "$DS" --length "$WIN" \
            --output "$OUTK/RE1-OB-granger" 2>&1 \
            | tee "$OUTK/RE1-OB-granger.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/granger"
        ;;
      pc)
        echo "--- [ds$k] PC ---"
        $PY "$FIX" --algo pc --data-root "$DS" --window "$WIN" \
            --output "$OUTK/RE1-OB-pc" 2>&1 \
            | tee "$OUTK/RE1-OB-pc.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pc"
        ;;
      pcmci)
        echo "--- [ds$k] PCMCI ---"
        $PY "$FIX" --algo pcmci --data-root "$DS" --window "$WIN" \
            --output "$OUTK/RE1-OB-pcmci" 2>&1 \
            | tee "$OUTK/RE1-OB-pcmci.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pcmci"
        ;;
      *)
        echo "⚠️  未知方法: $m（可选 granger / pc / pcmci）"
        ;;
    esac
  done

  echo "✅ ds$k 处理完毕"
done

T1=$(date +%s)
echo ""
echo "============================================================"
echo "全部结束，耗时 $(( (T1 - T0) / 60 )) 分钟"
echo "产出目录: $OUT"
if [ -n "$FAIL" ]; then
  echo "⚠️ 以下步骤返回非 0（请检查对应日志，不一定是真失败）:"
  for f in $FAIL; do echo "    $f"; done
else
  echo "✅ 所有步骤返回码为 0"
fi
echo "============================================================"
