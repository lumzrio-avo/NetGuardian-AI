# -*- coding: utf-8 -*-
"""
run_granger.py — Granger因果基线复现脚本

复现 RCAEval 官方 granger_pagerank 的完整流程：
  1. 数据加载  （严格对齐 main.py process() 逻辑）
  2. 预处理    （严格对齐 io/time_series.py preprocess()）
  3. Granger因果检验（graph_construction/granger.py, maxlag=3, p<0.05）
  4. PageRank排序（graph_heads/page_rank.py）
  5. 结果保存  （每个案例一个 CSV，汇总一个 CSV）

用法:
    python run_granger.py                          # 自动扫描 data/ 下所有数据集
    python run_granger.py --dataset re1-ob         # 只跑 RE1-OB
    python run_granger.py --dataset re1-ob --length 20   # 指定窗口长度（分钟）
    python run_granger.py --dataset re1-ob --test  # 只跑前 2 个案例（冒烟测试）
    python run_granger.py --dataset re1-ob --output my_results  # 指定输出目录

依赖:
    pip install pandas numpy statsmodels scikit-network tqdm
    （RCAEval 代码库需在 PYTHONPATH 可访问）

Author: 大创团队
Date:   2026-07-22
"""
import networkx as nx
import argparse
import glob
import os
import sys
import time
from datetime import datetime
from os.path import basename, dirname, join

import numpy as np
import pandas as pd
from tqdm import tqdm

# 静默 statsmodels 等库的 FutureWarning（不静默的话刷屏很乱）
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 确保 RCAEval 可导入
# ============================================================
try:
    from RCAEval.io.time_series import (
        convert_mem_mb, drop_constant, drop_time, preprocess,
    )
    from RCAEval.graph_construction.granger import granger
#    from RCAEval.graph_heads.page_rank import page_rank
    from RCAEval.benchmark.evaluation import Evaluator
    from RCAEval.classes.graph import Node
except ImportError:
    # 自动检测 RCAEval 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rcaeval_paths = [
        join(script_dir, "RCAEval-main", "RCAEval-main"),
        join(script_dir, "RCAEval"),
        os.path.expanduser("~/RCAEval"),
    ]
    for p in rcaeval_paths:
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break
    from RCAEval.io.time_series import (
        convert_mem_mb, drop_constant, drop_time, preprocess,
    )
    from RCAEval.graph_construction.granger import granger
    from RCAEval.graph_heads.page_rank import page_rank
    from RCAEval.benchmark.evaluation import Evaluator
    from RCAEval.classes.graph import Node


# ============================================================
# 数据集路径映射（对齐 main.py DATASET_MAP）
# ============================================================
DATASET_MAP = {
    "re1-ob": "data/RE1/RE1-OB",
    "re1-ss": "data/RE1/RE1-SS",
    "re1-tt": "data/RE1/RE1-TT",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
}

DEFAULT_SLI_MAP = {
    "re1-ob": "frontend_latency",
    "re1-ss": "front-end_cpu",
    "re1-tt": "ts-ui-dashboard_latency",
    "re2-ob": "frontend_latency",
    "re2-ss": "frontend_latency",
    "re2-tt": "ts-ui-dashboard_latency",
    "online-boutique": "frontend_latency",
    "sock-shop-1": "front-end_cpu",
    "sock-shop-2": "front-end_cpu",
    "train-ticket": "ts-ui-dashboard_latency",
}

# 故障类型 → 指标名映射（对齐 RCAEval main.py）
# 数据目录中的 fault_type（如 "delay"）与数据列后缀（如 "latency"）不一致时需要映射
FAULT_TO_METRIC = {
    "delay":  "latency",   # delay 故障影响的是延迟指标
    "loss":   "latency",   # loss 故障也反映在延迟指标上
    "disk":   "diskio",    # disk 故障看 diskio（但 RE1-OB 数据无此列）
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Granger因果基线复现 (RCAEval granger_pagerank)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASET_MAP.keys()),
        help="数据集名称 (不指定则扫描所有 data/ 目录)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="数据根目录 (替代 --dataset 的自动路径映射)",
    )
    parser.add_argument(
        "--length", type=int, default=20,
        help="窗口长度（分钟），默认 20 min (即 normal/anomaly 各 600 点 @ 1Hz)",
    )
    parser.add_argument(
        "--output", type=str, default="output_granger",
        help="输出目录",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="冒烟测试模式（只跑前 2 个案例）",
    )
    parser.add_argument(
        "--rca-path", type=str, default=None,
        help="RCAEval 代码库路径 (默认自动检测)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续跑：跳过输出目录中已存在的案例",
    )
    return parser.parse_args()


# ============================================================
# 核心函数：复制 official main.py process() + granger_pagerank()
# ============================================================

def load_and_preprocess(
    data_path: str,
    window_length_min: int = 20,
    verbose: bool = False,
) -> tuple:
    """
    严格对齐 main.py process() 的数据加载与预处理逻辑。
    自动识别 OB (Online Boutique) 和 SS (Sock Shop) 两种列名格式。

    OB 格式:  _latency-50 / _latency-90 / frontend_latency
    SS 格式:  lat_50 / lat_90 / lat_99 / front-end_cpu

    步骤:
    1. pd.read_csv(data_path)
    2. 识别格式 → 丢弃低分位延迟列
    3. inf → NaN → ffill() → fillna(0)
    4. 读取 inject_time.txt
    5. 按 inject_time 切分 normal/anomaly
    6. 重命名保留的延迟列 → _latency
    7. 确定 SLI
    """
    data_dir = dirname(data_path)
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)

    # === Step 1: 读取 CSV ===
    df = pd.read_csv(data_path)

    if verbose:
        print(f"  [LOAD] {data_path}")

    # === Step 2: 识别格式 → 丢弃低分位延迟列 ===
    # OB: 丢弃 _latency-50; SS: 丢弃 lat_50, lat_99
    has_ob_latency = any(c.endswith("_latency-50") for c in df.columns)
    has_ss_latency = any(c.endswith("lat_50") and not c.startswith("_") for c in df.columns)

    if has_ob_latency:
        df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
        latency_suffix_old = "_latency-90"
        latency_suffix_new = "_latency"
    elif has_ss_latency:
        for col in list(df.columns):
            if col.endswith("lat_50") or col.endswith("lat_99"):
                df = df.drop(columns=[col])
        latency_suffix_old = "lat_90"
        latency_suffix_new = "_latency"
    else:
        latency_suffix_old = None

    # === Step 3: 无穷值 + 缺失值处理 ===
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    df = df.fillna(0)

    # === Step 4: 读取故障注入时间 ===
    inject_path = join(data_dir, "inject_time.txt")
    with open(inject_path, "r") as f:
        inject_time = int(f.readlines()[0].strip())

    if verbose:
        inject_dt = datetime.fromtimestamp(inject_time)
        print(f"  [INJECT] time={inject_time} ({inject_dt})")

    # === Step 5: 按 inject_time 切分 (对齐 main.py) ===
    points = window_length_min * 60 // 2  # 20 min → 600 points each
    normal_df = df[df["time"] < inject_time].tail(points)
    anomal_df = df[df["time"] >= inject_time].head(points)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    if verbose:
        print(f"  [SPLIT] normal={normal_df.shape}, anomaly={anomal_df.shape}, "
              f"combined={data.shape}")

    # === Step 6: 重命名延迟列 ===
    if latency_suffix_old:
        data = data.rename(
            columns={
                c: c.replace(latency_suffix_old, latency_suffix_new)
                for c in data.columns
                if c.endswith(latency_suffix_old)
            }
        )

    # === Step 7: 确定 SLI ===
    # 优先用当前服务 + _latency
    sli = f"{service}_latency"
    if sli not in data.columns:
        # SS 回退: front-end_cpu; OB 回退: frontend_latency
        if "front-end_cpu" in data.columns:
            sli = "front-end_cpu"
        else:
            sli = "frontend_latency"

    return data, inject_time, service, metric, sli


def run_granger_pagerank(
    data: pd.DataFrame,
    inject_time: int,
    dataset: str = None,
    sli: str = "frontend_latency",
    verbose: bool = False,
) -> dict:
    """
    严格对齐 granger_pagerank.py 的因果发现 + PageRank 排序。

    Args:
        data:   已经过 load_and_preprocess 的 DataFrame
        inject_time: 故障注入时间戳（传递但 granger_pagerank 内部未显式使用）
        dataset: 数据集名称（用于 preprocess 兼容不同格式）
        sli:    SLI 指标名
        verbose: 是否打印日志

    Returns:
        {
            "adj":          np.ndarray,         # 邻接矩阵 (n x n)
            "node_names":   List[str],          # 列名
            "ranks":        List[str],          # 按 PageRank 降序排列的节点名
            "ranks_scored": List[Tuple[str, float]],  # 含 PageRank 分数的排名
            "n_nodes":      int,                # 节点数
            "n_edges":      int,                # 边数
        }
    """
    n_cols_before = data.shape[1]

    # === Step 1: io/time_series.preprocess ===
    # 对齐 granger_pagerank: preprocess(data, dataset, dk_select_useful=False)
    data = preprocess(data=data, dataset=dataset, dk_select_useful=False)
    


    node_names = data.columns.to_list()
    n_cols_after = len(node_names)

    if verbose:
        print(f"  [PREPROCESS] {n_cols_before} → {n_cols_after} columns "
              f"(dropped {n_cols_before - n_cols_after})")

    # === Step 2: Granger 因果检验 ===
    # 对齐: adj = granger(data)  →  maxlag=3, p<0.05, test=None (4种平均)
    t0 = time.time()
    adj = granger(data)

    n_edges = int(adj.sum())
    elapsed = time.time() - t0

    if verbose:
        print(f"  [GRANGER] {n_edges} edges found in {elapsed:.1f}s "
              f"(density={n_edges / (len(node_names) * (len(node_names) - 1)):.3f})")

    # === Step 3: PageRank 排序 ===
    if adj.sum().sum() == 0:
        # 无边 → 按列名原顺序（对齐官方 fallback）
        if verbose:
            print(f"  [PAGERANK] adj all-zero, falling back to original order")
        ranks = node_names
        ranks_scored = [(n, 0.0) for n in node_names]
    else:
        # 用 networkx 替代官方 scikit-network PageRank
        # adj[i,j]=1 表示 j Granger-causes i（即 j→i）
        n = len(node_names)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    G.add_edge(j, i)  # j causes i → edge j→i

        pr_scores = nx.pagerank(G, alpha=0.85)
        # pr_scores 的 key 是整数 0..n-1，用 node_names[i] 映射
        scored_ranks = [(node_names[i], pr_scores.get(i, 0.0)) for i in range(n)]
        scored_ranks_sorted = sorted(scored_ranks, key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in scored_ranks_sorted]
        ranks_scored = scored_ranks_sorted

        if verbose and len(ranks) > 0:
            print(f"  [PAGERANK] top-3 scores: {scored_ranks_sorted[:3]}")

    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
        "ranks_scored": ranks_scored,
        "n_nodes": len(node_names),
        "n_edges": n_edges,
    }


# ============================================================
# 输出：CSV 保存
# ============================================================

def save_case_result(
    result: dict,
    service: str,
    fault_type: str,
    case_id: int,
    dataset: str,
    output_dir: str,
):
    """
    保存单个案例的排名结果为 CSV。

    输出文件: output_dir/{dataset}/{service}_{fault_type}/{case_id}.csv
    """
    case_dir = join(output_dir, dataset, f"{service}_{fault_type}")
    os.makedirs(case_dir, exist_ok=True)

    csv_path = join(case_dir, f"{case_id}.csv")

    df = pd.DataFrame({
        "rank": range(1, len(result["ranks"]) + 1),
        "node": result["ranks"],
        "pagerank_score": [score for _, score in result["ranks_scored"]]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def save_summary(all_results: list, output_dir: str):
    """
    保存所有案例的汇总统计 CSV。

    输出文件: output_dir/summary.csv
    """
    rows = []
    for r in all_results:
        rows.append({
            "dataset":    r["dataset"],
            "service":    r["service"],
            "fault_type": r["fault_type"],
            "case_id":    r["case_id"],
            "n_nodes":    r["n_nodes"],
            "n_edges":    r["n_edges"],
            "edge_density": round(
                r["n_edges"] / max(r["n_nodes"] * (r["n_nodes"] - 1), 1), 4
            ),
            "top1":       r["ranks"][0] if r["ranks"] else "",
            "top3":       "|".join(r["ranks"][:3]) if len(r["ranks"]) >= 3 else "",
            "ground_truth_service": r["service"],
            "ground_truth_metric": FAULT_TO_METRIC.get(r["fault_type"], r["fault_type"]),
            "rank_of_truth": r.get("rank_of_truth", ""),
            "rank_of_truth_metric": r.get("rank_of_truth_metric", ""),
        })

    df = pd.DataFrame(rows)
    csv_path = join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================
# 评估计算
# ============================================================

def compute_and_print_eval(all_results: list, output_dir: str = None):
    """
    严格对齐 main.py 的评估逻辑：
    - service-level (答案=service+"unknown"): AC@1/3/5, Avg@5
    - fine-grained (答案=service+fault):     AC@1/3/5, Avg@5
    - 按故障类型分组统计

    输出：控制台打印 + 可选保存 eval.csv
    """
    if not all_results:
        print("[EVAL] 无结果，跳过评估")
        return

    # 初始化评估器（对齐 main.py 结构）
    s_evaluator_all = Evaluator()  # service-level, 所有类型
    f_evaluator_all = Evaluator()  # fine-grained, 所有类型

    type_evaluators = {}  # fault_type → {"service": Evaluator, "metric": Evaluator}
    for ft in sorted(set(r["fault_type"] for r in all_results)):
        type_evaluators[ft] = {
            "service": Evaluator(),
            "metric": Evaluator(),
        }

    # 逐案例添加
    for r in all_results:
        ranks = r["ranks"]
        service = r["service"]
        fault_type = r["fault_type"]

        # —— Service-level：只比较服务名 ——
        s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
        # 去重（对齐 main.py）
        s_ranks_dedup = [s_ranks[0]] + [
            s_ranks[i] for i in range(1, len(s_ranks))
            if s_ranks[i] not in s_ranks[:i]
        ]
        s_answer = Node(service, "unknown")

        s_evaluator_all.add_case(ranks=s_ranks_dedup, answer=s_answer)
        type_evaluators[fault_type]["service"].add_case(ranks=s_ranks_dedup, answer=s_answer)

        # —— Fine-grained：比较服务+指标 ——
        # 对齐 RCAEval main.py：用映射后的metric做答案
        f_ranks = [Node(x.split("_")[0], x.split("_")[1] if "_" in x else "unknown") for x in ranks]
        mapped_metric = FAULT_TO_METRIC.get(fault_type, fault_type)
        f_answer = Node(service, mapped_metric)

        f_evaluator_all.add_case(ranks=f_ranks, answer=f_answer)
        type_evaluators[fault_type]["metric"].add_case(ranks=f_ranks, answer=f_answer)

    # —— 打印评估报告 ——
    print(f"\n{'=' * 60}")
    print(f"评估结果 (n={s_evaluator_all.num} cases)")
    print(f"{'=' * 60}")

    # 总览
    print(f"\n{'指标':<12} {'Svc-AC@1':>8} {'Svc-AC@3':>8} {'Svc-Avg@5':>10}"
          f" {'Mtr-AC@1':>8} {'Mtr-Avg@5':>10}")
    print("-" * 60)
    print(f"{'Overall':<12} {s_evaluator_all.accuracy(1):>8.4f} "
          f"{s_evaluator_all.accuracy(3):>8.4f} {s_evaluator_all.average(5):>10.4f}"
          f" {f_evaluator_all.accuracy(1):>8.4f} {f_evaluator_all.average(5):>10.4f}")

    # 按故障类型
    eval_rows = []
    fault_order = ["cpu", "mem", "delay", "loss", "disk", "socket"]
    for ft in fault_order:
        if ft not in type_evaluators:
            continue
        ev = type_evaluators[ft]
        if ev["service"].num == 0:
            continue
        s_avg5 = ev["service"].average(5)
        f_avg5 = ev["metric"].average(5)
        ft_display = ft.upper()
        print(f"{ft_display:<12} {ev['service'].accuracy(1):>8.4f} "
              f"{ev['service'].accuracy(3):>8.4f} {s_avg5:>10.4f}"
              f" {ev['metric'].accuracy(1):>8.4f} {f_avg5:>10.4f}")

        eval_rows.append({
            "fault_type": ft,
            "n_cases": ev["service"].num,
            "svc_ac1": ev["service"].accuracy(1),
            "svc_ac3": ev["service"].accuracy(3),
            "svc_ac5": ev["service"].accuracy(5),
            "svc_avg5": s_avg5,
            "mtr_ac1": ev["metric"].accuracy(1),
            "mtr_avg5": f_avg5,
        })

    print("-" * 60)

    # 保存评估结果 CSV
    if output_dir and eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        # 追加总体行
        overall_row = {
            "fault_type": "OVERALL",
            "n_cases": s_evaluator_all.num,
            "svc_ac1": s_evaluator_all.accuracy(1),
            "svc_ac3": s_evaluator_all.accuracy(3),
            "svc_ac5": s_evaluator_all.accuracy(5),
            "svc_avg5": s_evaluator_all.average(5),
            "mtr_ac1": f_evaluator_all.accuracy(1),
            "mtr_avg5": f_evaluator_all.average(5),
        }
        eval_df = pd.concat([eval_df, pd.DataFrame([overall_row])], ignore_index=True)
        eval_path = join(output_dir, "eval.csv")
        eval_df.to_csv(eval_path, index=False)
        print(f"\n评估结果已保存: {eval_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()

    # === 确定数据根目录 ===
    if args.data_root:
        data_roots = [args.data_root]
    elif args.dataset:
        data_roots = [DATASET_MAP[args.dataset.lower()]]
    else:
        # 自动扫描所有已知数据集
        data_roots = [
            p for p in DATASET_MAP.values()
            if os.path.isdir(p)
        ]
        if not data_roots:
            print("[ERROR] 未发现数据集。请用 --data-root 指定路径。")
            print("  示例: python run_granger.py --data-root data/RE1/RE1-OB")
            sys.exit(1)

    # === 创建输出目录 ===
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Granger因果基线复现 (granger_pagerank)")
    print(f"数据根目录: {data_roots}")
    print(f"窗口长度:   {args.length} min (各 {args.length * 60 // 2} 点)")
    print(f"输出目录:   {args.output}")
    print(f"测试模式:   {'是' if args.test else '否'}")
    print("=" * 60)

    # === 收集所有 data.csv 路径 ===
    all_data_paths = []
    for root in data_roots:
        dataset_name = os.path.basename(root)
        # data/{dataset}/{service}_{metric}/{case_id}/data.csv
        pattern = join(root, "**", "data.csv")
        paths = sorted(glob.glob(pattern, recursive=True))

        if not paths:
            print(f"[WARN] {root}: 未发现 data.csv 文件，跳过")
            continue

        for p in paths:
            all_data_paths.append((p, dataset_name))

    if args.test:
        all_data_paths = all_data_paths[:2]
        print(f"[TEST] 仅处理前 {len(all_data_paths)} 个案例")

    print(f"\n共发现 {len(all_data_paths)} 个案例\n")

    # === Resume: 加载已完成的案例 ===
    all_results = []
    existing_cases = set()  # (dataset, service, fault_type, case_id)
    if args.resume:
        # 1. 优先从 summary.csv 加载（最快）
        summary_path = join(args.output, "summary.csv")
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            for _, row in existing_df.iterrows():
                existing_cases.add((
                    row["dataset"], row["service"], row["fault_type"], int(row["case_id"])
                ))
                # summary.csv 中无 ranks 字段，构造占位记录（ranks=[] 不影响后续处理）
                all_results.append({
                    "dataset":     row["dataset"],
                    "service":     row["service"],
                    "fault_type":  row["fault_type"],
                    "case_id":     int(row["case_id"]),
                    "n_nodes":     row.get("n_nodes", ""),
                    "n_edges":     row.get("n_edges", ""),
                    "ranks":       [],
                    "rank_of_truth":       row.get("rank_of_truth", ""),
                    "rank_of_truth_metric": row.get("rank_of_truth_metric", ""),
                })
            print(f"[RESUME] 从 summary.csv 加载了 {len(existing_cases)} 个已完成案例")

        # 2. 同时扫描案例 CSV 目录（summary.csv 不存在时的兜底）
        # 只记录到 existing_cases 跳过列表，不污染 all_results（避免影响 eval）
        csv_only_count = 0
        for ds_name in set(ds_name for _, ds_name in all_data_paths):
            ds_dir = join(args.output, ds_name)
            if not os.path.isdir(ds_dir):
                continue
            for svc_fault in os.listdir(ds_dir):
                svc_dir = join(ds_dir, svc_fault)
                if not os.path.isdir(svc_dir):
                    continue
                if "_" not in svc_fault:
                    continue
                service, fault_type = svc_fault.split("_", 1)
                for fn in os.listdir(svc_dir):
                    if fn.endswith(".csv"):
                        try:
                            case_id = int(fn[:-4])
                        except ValueError:
                            continue
                        key = (ds_name, service, fault_type, case_id)
                        if key not in existing_cases:
                            existing_cases.add(key)
                            csv_only_count += 1

        if csv_only_count > 0:
            print(f"[RESUME] 额外从案例 CSV 目录发现 {csv_only_count} 个已完成案例")

        if existing_cases:
            print(f"[RESUME] 共识别 {len(existing_cases)} 个已完成案例（将被跳过）")

    # === 逐案例处理 ===
    start_time = time.time()
    skipped = 0

    try:
        for data_path, ds_name in tqdm(all_data_paths, desc="Processing"):
            # 解析路径信息
            service, metric = basename(dirname(dirname(data_path))).split("_", 1)
            case_id = int(basename(dirname(data_path)))

            # 断点续跑：跳过已完成的
            if args.resume and (ds_name, service, metric, case_id) in existing_cases:
                skipped += 1
                continue

            try:

                # Step 1: 加载 + 预处理
                data, inject_time, svc, fault, sli = load_and_preprocess(
                    data_path,
                    window_length_min=args.length,
                    verbose=False,
                )

                # Step 2: Granger + PageRank
                result = run_granger_pagerank(
                    data,
                    inject_time=inject_time,
                    dataset=ds_name,
                    sli=sli,
                    verbose=False,
                )

                # 计算答案在排名中的位置
                # Service-level: rank of {service}_xxx
                try:
                    rank_of_truth = next(
                        i + 1 for i, n in enumerate(result["ranks"])
                        if n.split("_")[0] == service
                    )
                except StopIteration:
                    rank_of_truth = "not_found"

                # Metric-level: rank of {service}_{mapped_metric}
                mapped_metric = FAULT_TO_METRIC.get(metric, metric)
                truth_metric = f"{service}_{mapped_metric}"
                try:
                    rank_of_truth_metric = result["ranks"].index(truth_metric) + 1
                except ValueError:
                    rank_of_truth_metric = "not_found"

                # Step 3: 保存案例结果
                case_output = save_case_result(
                    result, service, metric, case_id, ds_name, args.output,
                )

                # 收集汇总信息
                all_results.append({
                    "dataset":    ds_name,
                    "service":    service,
                    "fault_type": metric,
                    "case_id":    case_id,
                    "n_nodes":    result["n_nodes"],
                    "n_edges":    result["n_edges"],
                    "ranks":      result["ranks"],
                    "rank_of_truth": rank_of_truth,
                    "rank_of_truth_metric": rank_of_truth_metric,
                })

            except Exception as e:
                print(f"\n[ERROR] {data_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Ctrl+C 中断! 已保存 {len(all_results)} 个案例结果")
        print(f"下次运行: python run_granger.py --dataset re1-ss --output {args.output} --resume")

    finally:
        # 无论正常结束还是 Ctrl+C 中断，都保存当前进度
        if all_results:
            summary_path = save_summary(all_results, args.output)

            elapsed = time.time() - start_time
            new_in_this_run = len(all_results) - len(existing_cases)
            print(f"\n{'=' * 60}")
            print(f"完成! 成功 {len(all_results)} 案例, 跳过 {skipped} (resume), "
                  f"本次新增 {new_in_this_run}, 数据共 {len(all_data_paths)} 案例")
            print(f"总耗时: {elapsed:.1f}s (平均 {elapsed/max(new_in_this_run,1):.1f}s/案例)")
            print(f"结果目录: {args.output}/")
            print(f"汇总文件: {summary_path}")

            # 运行评估
            compute_and_print_eval(all_results, output_dir=args.output)

            print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
