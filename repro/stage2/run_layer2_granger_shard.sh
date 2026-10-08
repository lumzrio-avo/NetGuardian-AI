#!/usr/bin/env bash
# ============================================================
# Layer 2 · Granger 分片并行重跑
#   用途：RE1-SS / RE1-TT 这类单例很慢的数据集（SS 单例 ~239 s，TT 单例 ~811 s）
#         单进程跑满 125 例 × 3 档要 25 h 以上，必须分片。
#
# 用法:
#   bash run_layer2_granger_shard.sh 1 RE1-SS 6      # k=1，6 路并行
#   bash run_layer2_granger_shard.sh 4 RE1-SS 6
#
# 环境变量:
#   DSROOT  降采样数据根（默认 $HOME/RCAEval/data_ds_full）
#   OUT     输出根（默认 $REPO/结果/降采样实验_全量）
#
# 产出: $OUT/ds<k>/<数据集>-granger-shard<i>/   （i = 1..N，各自含独立的 summary/eval）
#       ⚠️ 分片结果需用 结果/merge_granger_shards.py 合并后才是完整评估
# ============================================================
set -u

PY=$HOME/RCAEval/env/bin/python
REPO=/mnt/d/北交威/大创
DSROOT=${DSROOT:-$HOME/RCAEval/data_ds_full}
OUT=${OUT:-$REPO/结果/降采样实验_全量}
GR=$REPO/结果/Granger/run_granger.py

K=${1:?用法: bash run_layer2_granger_shard.sh <k> <数据集> [分片数]}
DSNAME=${2:-RE1-SS}
N=${3:-6}

WIN=$((20 / K))
DS=$DSROOT/ds$K/$DSNAME

[ -x "$PY" ] || { echo "❌ 找不到 python: $PY"; exit 1; }
n=$(find "$DS" -name data.csv 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "❌ $DS 没有数据"; exit 1; }

echo "============================================================"
echo "Granger 分片并行: k=$K  $DSNAME  $N 路  窗口=$WIN"
echo "  数据: $DS  ($n 例)"
echo "  预计: 每分片约 $((n / N)) 例"
echo "============================================================"

T0=$(date +%s)
for i in $(seq 1 "$N"); do
  TGT=$OUT/ds$K/$DSNAME-granger-shard$i
  mkdir -p "$TGT"
  printf "  [%s] 分片 %d/%d → %s\n" "$(date +%H:%M:%S)" "$i" "$N" "$TGT"
  $PY "$GR" --data-root "$DS" --length "$WIN" --output "$TGT" --shard "$i/$N" \
      > "$TGT.log" 2>&1 &
done

wait
T1=$(date +%s)

echo ""
echo "============================================================"
echo "全部分片结束，耗时 $(( (T1 - T0) / 60 )) 分钟"
echo "下一步: ~/RCAEval/env/bin/python $REPO/结果/merge_granger_shards.py --k $K --dataset $DSNAME"
echo "============================================================"
