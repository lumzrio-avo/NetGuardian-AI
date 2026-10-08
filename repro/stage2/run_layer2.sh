#!/usr/bin/env bash
# ============================================================
# Layer 2 · 降采样重跑（通用版：支持 RE1-OB / RE1-SS / RE1-TT）
#
# 用法:
#   bash run_layer2.sh "1 2 4" "granger pc" RE1-OB
#   bash run_layer2.sh "1 2 4" "pc" RE1-SS
#
# 环境变量:
#   DSROOT  降采样数据根（默认 $HOME/RCAEval/data_ds）
#   OUT     输出根（默认 $REPO/结果/降采样实验）
#
# 方法 → 脚本映射（三方法在各数据集上的实现不统一，属历史原因）:
#   granger  · 任意      → run_granger.py
#   pc       · RE1-OB    → run_rca_full_ob.py --algo pc
#   pc       · RE1-SS    → run_pc_enhanced_ss.py
#   pc       · RE1-TT    → run_pc_tt.py
#   pcmci    · RE1-OB    → run_rca_full_ob.py --algo pcmci
#   pcmci    · RE1-SS/TT → run_pcmci_add_latency.py（需 --old-output，本脚本未支持）
#
# 注意: 窗口参数按 k 自动换算（k=1→20, k=2→10, k=4→5），
#       保证每段物理时长恒为 10 分钟（三个 runner 的窗口都是按「点数」取的）。
# ============================================================
set -u

PY=$HOME/RCAEval/env/bin/python
REPO=/mnt/d/北交威/大创
DSROOT=${DSROOT:-$HOME/RCAEval/data_ds}
OUT=${OUT:-$REPO/结果/降采样实验}

KS=${1:-"1 2 4"}
METHODS=${2:-"granger pc"}
DSNAME=${3:-RE1-OB}

GR=$REPO/结果/Granger/run_granger.py
FIX=$REPO/pc_pcmci_fixed

echo "============================================================"
echo "Layer 2 降采样重跑"
echo "  数据集: $DSNAME"
echo "  档位:   $KS"
echo "  方法:   $METHODS"
echo "  数据根: $DSROOT"
echo "  输出:   $OUT"
echo "============================================================"

[ -x "$PY" ] || { echo "❌ 找不到 python: $PY"; exit 1; }
[ -f "$GR" ] || { echo "❌ 找不到 $GR"; exit 1; }
[ -d "$FIX" ] || { echo "❌ 找不到 $FIX"; exit 1; }

for k in $KS; do
  d=$DSROOT/ds$k/$DSNAME
  n=$(find "$d" -name data.csv 2>/dev/null | wc -l)
  printf "  ds%-2s %-45s %s 个案例\n" "$k" "$d" "$n"
  [ "$n" -gt 0 ] || { echo "❌ ds$k/$DSNAME 没有数据"; exit 1; }
done

FAIL=""
T0=$(date +%s)

for k in $KS; do
  WIN=$((20 / k))
  DS=$DSROOT/ds$k/$DSNAME
  OUTK=$OUT/ds$k
  mkdir -p "$OUTK"

  echo ""
  echo "############################################################"
  echo "# ds$k / $DSNAME   (窗口参数 = $WIN)"
  echo "############################################################"

  for m in $METHODS; do
    case "$m:$DSNAME" in
      granger:*)
        TGT=$OUTK/$DSNAME-granger
        echo "--- [ds$k] Granger ---"
        $PY "$GR" --data-root "$DS" --length "$WIN" --output "$TGT" 2>&1 \
          | tee "$OUTK/$DSNAME-granger.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/granger"
        ;;
      pc:RE1-OB)
        TGT=$OUTK/$DSNAME-pc
        echo "--- [ds$k] PC (OB 实现) ---"
        $PY "$FIX/run_rca_full_ob.py" --algo pc --data-root "$DS" --window "$WIN" \
          --output "$TGT" 2>&1 | tee "$OUTK/$DSNAME-pc.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pc"
        ;;
      pc:RE1-SS)
        TGT=$OUTK/$DSNAME-pc
        echo "--- [ds$k] PC (SS 实现) ---"
        $PY "$FIX/run_pc_enhanced_ss.py" --data-root "$DS" --window "$WIN" \
          --output "$TGT" 2>&1 | tee "$OUTK/$DSNAME-pc.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pc"
        ;;
      pc:RE1-TT)
        TGT=$OUTK/$DSNAME-pc
        echo "--- [ds$k] PC (TT 实现) ---"
        $PY "$FIX/run_pc_tt.py" --data-root "$DS" --window "$WIN" \
          --output "$TGT" 2>&1 | tee "$OUTK/$DSNAME-pc.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pc"
        ;;
      pcmci:RE1-OB)
        TGT=$OUTK/$DSNAME-pcmci
        echo "--- [ds$k] PCMCI (OB 实现) ---"
        $PY "$FIX/run_rca_full_ob.py" --algo pcmci --data-root "$DS" --window "$WIN" \
          --output "$TGT" 2>&1 | tee "$OUTK/$DSNAME-pcmci.log" | tail -4
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL ds$k/pcmci"
        ;;
      pcmci:*)
        echo "--- [ds$k] PCMCI ($DSNAME) ---"
        echo "  ⚠️ 该数据集需走 run_pcmci_add_latency.py 并指定 --old-output（原基线），"
        echo "     本脚本暂未支持；请单独调用。"
        ;;
      *)
        echo "⚠️ 未知组合: $m / $DSNAME"
        ;;
    esac
  done
  echo "✅ ds$k 处理完毕"
done

T1=$(date +%s)
echo ""
echo "============================================================"
echo "结束，耗时 $(( (T1 - T0) / 60 )) 分钟"
echo "产出: $OUT"
if [ -n "$FAIL" ]; then
  echo "⚠️ 非 0 返回的步骤: $FAIL"
else
  echo "✅ 全部返回 0"
fi
echo "============================================================"
