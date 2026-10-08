#!/usr/bin/env bash
# ============================================================
# Layer 2 分片重跑 · 进度查看
#
# 用法（WSL 中）:
#   bash check_layer2_progress.sh                  # 默认看 RE1-TT / ds4 / 4 路
#   bash check_layer2_progress.sh 4 RE1-TT 4       # k=4, 数据集, 分片数
#   watch -n 60 "bash check_layer2_progress.sh"    # 每 60 秒自动刷新
# ============================================================
set -u

B=${B:-/mnt/d/北交威/大创/结果/降采样实验_全量}
DSROOT=${DSROOT:-$HOME/RCAEval/data_ds_full}
K=${1:-4}
DS=${2:-RE1-TT}
N=${3:-4}

# ⚠️ 总数要从「数据目录」查，不是输出目录（输出目录里没有原始 data.csv）
TOTAL=$(find "$DSROOT/ds$K/$DS" -name data.csv 2>/dev/null | wc -l)
PER=$(( TOTAL / N ))

echo "=== $DS / ds$K   进度（应处理 $TOTAL 例）==="
echo ""

done_all=0
for i in $(seq 1 "$N"); do
  d="$B/ds$K/$DS-granger-shard$i"
  if [ -d "$d" ]; then
    # 只数案例级 csv：{output}/{dataset}/{svc}_{fault}/{case}.csv
    n=$(find "$d/$DS" -mindepth 2 -maxdepth 2 -name "*.csv" 2>/dev/null | wc -l)
  else
    n=0
  fi
  done_all=$((done_all + n))
  printf "  shard%-2s  %3d / %d 例\n" "$i" "$n" "$PER"
done

echo "  ----"
printf "  合计      %3d 例  (%.0f%%)\n" "$done_all" "$(awk -v a="$done_all" -v b="$TOTAL" 'BEGIN{ if(b>0) print a*100.0/b; else print 0 }')"

# --- 速度：最近 5 分钟完成了多少例 ---
REC=$(find "$B/ds$K" -name "*.csv" -newermt "-5 minutes" 2>/dev/null | wc -l)
if [ "$REC" -gt 0 ]; then
  LEFT=$(( TOTAL - done_all ))
  ETA=$(awk -v l="$LEFT" -v r="$REC" 'BEGIN{ if(r>0) printf "%.1f", l/r*5/60; else print "?" }')
  printf "  近 5 分钟完成 %d 例  →  按此速度还需约 %s 小时\n" "$REC" "$ETA"
else
  echo "  近 5 分钟无新完成（可能都在处理大图，TT 单例可达数十分钟）"
fi

echo ""
echo "=== 最近一条分片日志 ==="
for i in $(seq 1 "$N"); do
  f="$B/ds$K/$DS-granger-shard$i.log"
  [ -f "$f" ] && { echo "  [shard$i] $(tail -1 "$f" | cut -c1-100)"; }
done

echo ""
echo "=== 内存 ==="
free -h | head -2
