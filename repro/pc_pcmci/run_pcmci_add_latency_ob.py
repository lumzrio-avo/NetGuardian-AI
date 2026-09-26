# -*- coding: utf-8 -*-
"""
run_pcmci_add_latency_ob.py — RE1-OB 专用 driver（基于 run_pcmci_add_latency.py）

【背景】
  你指定的 run_pcmci_add_latency.py 默认数据集是 RE1-TT / RE1-SS，且只跑 delay/loss。
  本工作区 data/ 目录下只有 RE1-OB（25 服务×故障 × 5 案例 = 125 案例），需要全跑。

【做法】
  不修改原脚本，monkey-patch 复用其全部核心算法（pcmci / PageRank / 评估 / 可视化），
  仅调整配置适配 RE1-OB：
    * DATASETS 加入 RE1-OB（旧基线 = output/RE1-OB）
    * FAULT_TYPES 改为全部 5 种（cpu / delay / disk / loss / mem）
    * LATENCY_SUFFIX / LATENCY_NEW_NAME 适配 RE1-OB 的列名（{svc}_latency 已直接可用）
    * 数据根 = data/RE1-OB（旧版 output/RE1-OB 提供原 50mem 特征 + 旧排名做基线对比）
    * 落盘到 output/pcmci_latency_fix/RE1-OB/（新建子目录，不动 TT/SS/原 RE1-OB 结果）
    * 默认断点续跑（RE1-OB 在 latency_fix 里从无记录 ⇒ 等价于全跑）

用法:
    python run_pcmci_add_latency_ob.py                # 全量 125 案例
    python run_pcmci_add_latency_ob.py --workers 8    # 指定并行度
    python run_pcmci_add_latency_ob.py --test         # 冒烟测试（每数据集 1 案例）

Author: 大创团队
Date:   2026-09-14
"""
import os

# 并行时避免 BLAS 线程超订（必须在 import numpy 之前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# === 复用你指定版本的源代码 ===
import run_pcmci_add_latency as base

# === 适配 RE1-OB ===
# 1) 列名：RE1-OB 的 latency 列已经是 {svc}_latency，无须重命名（等价 no-op）
base.LATENCY_SUFFIX = "_latency"
base.LATENCY_NEW_NAME = "_latency"

# 2) 数据集 + 旧基线（沿用你指定脚本里的同名字段）
base.DATASETS["RE1-OB"] = {
    "data_root": "D:/WorkBuddy_PCMCI_OB/data/RE1-OB",
    "old_output": "D:/WorkBuddy_PCMCI_OB/output/RE1-OB",
}

# 3) 故障类型：全跑 5 种（cpu / delay / disk / loss / mem），覆盖 RE1-OB 全量 125 案例
base.FAULT_TYPES = ["cpu", "delay", "disk", "loss", "mem"]

# 4) 故障类型 → 指标名映射（对齐 RCAEval main.py）
#    cpu/disk/mem 对应的"指标"按你原口径保留 (cpu→cpu, disk→disk, mem→mem)
base.FAULT_TO_METRIC = {
    "cpu":  "cpu",
    "disk": "disk",
    "mem":  "mem",
    "delay": "latency",
    "loss":  "latency",
}


def main():
    # 把 --datasets RE1-OB 注入 argv（保留用户其他参数）
    extra = ["--datasets", "RE1-OB"]
    sys.argv = [sys.argv[0]] + extra + [a for a in sys.argv[1:]]
    # 调用原脚本的 main()，所有 monkey-patch 都已生效
    base.main()


if __name__ == "__main__":
    main()