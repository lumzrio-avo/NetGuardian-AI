#!/usr/bin/env bash
# ============================================================
# TT 邻接矩阵补跑：N 路分片并行（默认 6 路）
#
# 背景：TT 单案例 ~3.24h（1169 指标 → preprocess 后约 761~800 节点，
#       granger() 双循环 ~60 万个有序对）。串行 125 例 ≈ 17 天；
#       案例之间完全独立，故可分片并行。
#
# 用法（在 WSL 里执行）:
#   bash "/mnt/d/北交威/大创/结果/Granger/run_tt_shards.sh"       # 默认 6 路
#   bash "/mnt/d/北交威/大创/结果/Granger/run_tt_shards.sh" 4     # 改成 4 路
#
# 查看进度:
#   tail -f "/mnt/d/北交威/大创/结果/Granger/output_granger_TT_shards/shard1.log"
#   pgrep -fc run_granger.py        # 还有几个分片在跑（0 = 全部结束）
#   find "/mnt/d/北交威/大创/结果/Granger/output_granger_TT_shards" \
#        -name "*_adj.npy" | wc -l   # 已产出的邻接矩阵数（目标 125）
#
# 全部跑完后合并 + 评估:
#   /home/lumzrio/RCAEval/env/bin/python \
#       "/mnt/d/北交威/大创/结果/Granger/merge_tt_shards.py"
# ============================================================
set -u

N="${1:-6}"
GR_DIR="/mnt/d/北交威/大创/结果/Granger"
OUT_BASE="$GR_DIR/output_granger_TT_shards"
PY="/home/lumzrio/RCAEval/env/bin/python"

if [ ! -x "$PY" ]; then
    echo "[ERROR] 找不到解释器: $PY"
    exit 1
fi
if [ ! -d /home/lumzrio/RCAEval/data/RE1/RE1-TT ]; then
    echo "[ERROR] 找不到 TT 数据: /home/lumzrio/RCAEval/data/RE1/RE1-TT"
    exit 1
fi

# DATASET_MAP 里是相对路径，cwd 必须是 ~/RCAEval
cd /home/lumzrio/RCAEval || exit 1

# 关键：限制 BLAS/numpy 内部线程，否则 6 个进程会互相抢核，反而更慢
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_BASE"

echo "============================================================"
echo " TT 分片补跑"
echo "   并行度:     $N"
echo "   输出根目录: $OUT_BASE"
echo "   解释器:     $PY"
echo "============================================================"

for i in $(seq 1 "$N"); do
    shard_dir="$OUT_BASE/shard$i"
    log="$OUT_BASE/shard$i.log"
    mkdir -p "$shard_dir"

    setsid nohup nice -n 5 "$PY" "$GR_DIR/run_granger.py" \
        --dataset re1-tt \
        --output  "$shard_dir" \
        --shard   "$i/$N" \
        --resume \
        > "$log" 2>&1 &

    printf '  shard %s/%s  已启动   log=%s\n' "$i" "$N" "$log"
done

echo
echo "启动完毕（进程已脱离终端，关掉窗口也不会停）。"
echo "  tail -f $OUT_BASE/shard1.log"
echo "  pgrep -fc run_granger.py"
