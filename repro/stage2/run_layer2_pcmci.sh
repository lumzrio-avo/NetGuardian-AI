#!/usr/bin/env bash
# ============================================================
# Layer 2 · PCMCI 降采样重跑（RE1-SS / RE1-TT）
#
# 为什么单独一个脚本：SS/TT 的 PCMCI 走 run_pcmci_add_latency.py，
# 需要 --old-output（原基线提供 {case}_node_names.json），参数形态与 OB 不同。
#
# 用法（在 WSL 终端里）:
#   bash run_layer2_pcmci.sh              # SS + TT，各 3 档
#   bash run_layer2_pcmci.sh "RE1-SS"     # 只跑 SS
#
# 环境变量: DSROOT / OUT / BASE / WORKERS 均可覆盖
# ============================================================
set -u

PY=$HOME/RCAEval/env/bin/python
REPO=/mnt/d/北交威/大创
DSROOT=${DSROOT:-$HOME/RCAEval/data_ds_full}
OUT=${OUT:-$REPO/结果/降采样实验_全量}
BASE=${BASE:-$REPO/结果/pc_pcmci_v2}
RUNNER=$REPO/pc_pcmci_fixed/run_pcmci_add_latency.py

DSS=${1:-"RE1-SS RE1-TT"}
KS="1 2 4"
WORKERS=${WORKERS:-6}

echo "============================================================"
echo "Layer 2 · PCMCI 降采样重跑"
echo "  数据集: $DSS"
echo "  档位:   $KS"
echo "  并行度: $WORKERS"
echo "  输出:   $OUT"
echo "============================================================"

[ -x "$PY" ] || { echo "❌ 找不到可执行的 python: $PY"; exit 1; }
[ -f "$RUNNER" ] || { echo "❌ 找不到 $RUNNER"; exit 1; }

FAIL=""
T0=$(date +%s)

for ds in $DSS; do
  # 基线目录探测：常规是 $BASE/{ds}-pcmci/{ds}；
  # ⚠️ 历史遗留：RE1-TT 的基线被写进了名为 RE1-SS 的子目录（内容确实是 TT 的），故做兼容
  OLD=""
  for cand in "$BASE/${ds}-pcmci/$ds" "$BASE/${ds}-pcmci/RE1-SS" "$BASE/${ds}-pcmci/RE1-TT"; do
    if [ -d "$cand" ]; then OLD="$cand"; break; fi
  done
  if [ -z "$OLD" ]; then
    echo "❌ 找不到 $ds 的基线目录（已尝试 $BASE/${ds}-pcmci/{$ds, RE1-SS, RE1-TT}）"
    FAIL="$FAIL $ds(无基线)"
    continue
  fi
  echo "  基线目录: $OLD"

  for k in $KS; do
    WIN=$((20 / k))
    DATA=$DSROOT/ds$k/$ds
    TGT=$OUT/ds$k/${ds}-pcmci

    n=$(find "$DATA" -name data.csv 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "❌ ds$k/$ds 无数据，跳过"
      FAIL="$FAIL $ds/ds$k(无数据)"
      continue
    fi

    mkdir -p "$TGT"
    echo ""
    echo "############################################################"
    echo "# $ds / ds$k   (window=$WIN, 案例 $n, 基线 $OLD)"
    echo "############################################################"
    $PY "$RUNNER" \
        --datasets "$ds" \
        --data-root "$DATA" \
        --old-output "$OLD" \
        --output "$TGT" \
        --window "$WIN" \
        --workers "$WORKERS" 2>&1 \
        | tee "$OUT/ds$k/$ds-pcmci.log" | tail -6
    [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL="$FAIL $ds/ds$k"
    echo "✅ ds$k/$ds 完毕"
  done
done

T1=$(date +%s)
echo ""
echo "============================================================"
echo "结束，耗时 $(( (T1 - T0) / 60 )) 分钟"
echo "产出: $OUT"
if [ -n "$FAIL" ]; then
  echo "⚠️ 有问题的步骤: $FAIL"
else
  echo "✅ 全部返回 0"
fi
echo "============================================================"
