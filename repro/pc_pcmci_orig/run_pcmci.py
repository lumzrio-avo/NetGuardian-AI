# -*- coding: utf-8 -*-
"""
run_pcmci.py — PCMCI 因果推断基线复现脚本

功能：
  1. 扫描 data/RE1-SS/ 下所有案例（data.csv + inject_time.txt）
  2. 对每个案例执行预处理（对齐 granger.py 的 load_and_preprocess 逻辑，无 RCAEval 依赖）
  3. 运行 PCMCI 算法（调用 pcmci_module.py 的 pcmci 函数）
  4. 保存因果邻接矩阵 + PageRank 排序结果
  5. 计算 Top-K Accuracy 评估指标
  6. 汇总所有案例的评估结果

用法:
    python run_pcmci.py                          # 跑全部案例
    python run_pcmci.py --test                   # 只跑前 2 个案例（冒烟测试）
    python run_pcmci.py --tau_max 5              # 指定最大滞后阶数
    python run_pcmci.py --alpha 0.01             # 指定显著性水平

Author: 大创团队
Date:   2026-07-27
"""
import os
import sys
import glob
import time
import argparse
import warnings
from os.path import basename, dirname, join

# Windows 控制台 UTF-8 编码（避免 emoji 输出报错）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 导入 PCMCI 算法
try:
    from pcmci_module import pcmci
except ImportError:
    print("错误：pcmci_module.py 中没有找到 pcmci 函数！")
    sys.exit(1)

# ============================================================
# 配置区
# ============================================================
DATA_ROOT = "D:/WorkBuddy_PCMCI_SS/data/RE1-SS"
OUTPUT_DIR = "D:/WorkBuddy_PCMCI_SS/output_ss"
TAU_MAX = 3
ALPHA = 0.05
WINDOW_LENGTH_MIN = 20  # 窗口长度（分钟）
MAX_FEATURES = 50       # 特征选择：保留方差最大的 N 个变量（SS 数据列数多，需筛选）
DROP_NODE_METRICS = True  # 是否丢弃节点级指标（192-* 开头的列）

# 故障类型 → 指标名映射（对齐 RCAEval main.py）
# SS 数据包含 cpu/mem/delay/loss/disk 五种故障类型
FAULT_TO_METRIC = {
    "delay": "latency",
    "loss":  "latency",
    "disk":  "diskio",
    "cpu":   "cpu",
    "mem":   "memory",
}


# ============================================================
# 预处理函数（对齐 granger.py 的 load_and_preprocess，无 RCAEval 依赖）
# ============================================================
def load_and_preprocess(data_path, window_length_min=20, verbose=False):
    """
    数据加载与预处理，严格对齐 granger.py 的 load_and_preprocess 逻辑。

    步骤:
    1. pd.read_csv(data_path)
    2. 识别 OB 格式 → 丢弃 _latency-50 列
    3. inf → NaN → ffill() → fillna(0)
    4. 读取 inject_time.txt
    5. 按 inject_time 切分 normal/anomaly（各 window_length_min*60//2 点）
    6. 重命名 _latency-90 → _latency
    7. 确定 SLI
    """
    data_dir = dirname(data_path)
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)

    # === Step 1: 读取 CSV ===
    df = pd.read_csv(data_path)

    if verbose:
        print(f"  [LOAD] {data_path}, shape={df.shape}")

    # === Step 2: 识别数据格式 → 丢弃低分位延迟列 ===
    # OB 格式:  _latency-50 / _latency-90
    # SS 格式:  istio-latency-50 / istio-latency-90 / istio-latency-95 / istio-latency-99
    has_ob_latency = any(c.endswith("_latency-50") for c in df.columns)
    has_ss_latency = any(c.endswith("istio-latency-50") for c in df.columns)

    if has_ob_latency:
        # OB: 丢弃 _latency-50，保留 _latency-90 → 重命名为 _latency
        df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
        latency_suffix_old = "_latency-90"
        latency_suffix_new = "_latency"
    elif has_ss_latency:
        # SS: 丢弃 istio-latency-50/95/99，保留 istio-latency-90 → 重命名为 latency
        df = df.loc[:, ~df.columns.str.endswith("istio-latency-50")]
        df = df.loc[:, ~df.columns.str.endswith("istio-latency-95")]
        df = df.loc[:, ~df.columns.str.endswith("istio-latency-99")]
        latency_suffix_old = "istio-latency-90"
        latency_suffix_new = "latency"
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
        from datetime import datetime
        inject_dt = datetime.fromtimestamp(inject_time)
        print(f"  [INJECT] time={inject_time} ({inject_dt})")

    # === Step 5: 按 inject_time 切分 ===
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
    # OB: {service}_latency; SS: front-end_cpu (SS 默认 SLI)
    sli = f"{service}_latency"
    if sli not in data.columns:
        if "front-end_cpu" in data.columns:
            sli = "front-end_cpu"
        else:
            sli = "frontend_latency"

    # === Step 8: 清理非特征列（对齐 RCAEval preprocess） ===
    # 丢弃 time, time.1 等非特征列
    drop_cols = [c for c in data.columns if c.lower() in ("time", "time.1") or c.startswith("time")]
    data = data.drop(columns=drop_cols, errors="ignore")

    # 丢弃常量列（std == 0）
    const_cols = [c for c in data.columns if data[c].std() == 0]
    if const_cols:
        data = data.drop(columns=const_cols)
        if verbose:
            print(f"  [DROP_CONST] dropped {len(const_cols)} constant columns")

    return data, inject_time, service, metric, sli


# ============================================================
# 特征选择（SS 数据列数多，PCMCI 需控制变量数以保证可计算性）
# ============================================================
def select_features(data, max_features=50, drop_node_metrics=True,
                    preserve_cols=None, preserve_patterns=None, verbose=False):
    """
    特征选择：
    1. 可选：丢弃节点级指标（列名以 IP 地址开头，如 192-168-*）
    2. 强制保留指定列 / 匹配指定模式的列（默认保留所有 latency 列）
    3. 剩余列按方差降序保留 top-N，使总列数不超过 max_features

    Args:
        data: 预处理后的 DataFrame
        max_features: 保留的最大特征数
        drop_node_metrics: 是否丢弃节点级指标
        preserve_cols: 必须保留的列名列表（如 SLI、真实根因指标）
        preserve_patterns: 必须保留的列名模式列表（默认保留含 "_latency" 的列）
        verbose: 是否打印日志

    Returns:
        筛选后的 DataFrame
    """
    n_before = data.shape[1]

    # Step 1: 丢弃节点级指标
    if drop_node_metrics:
        node_cols = [c for c in data.columns if c.split("_")[0].startswith("192-")]
        if node_cols:
            data = data.drop(columns=node_cols)
            if verbose:
                print(f"  [FEAT_SEL] dropped {len(node_cols)} node-level metrics")

    # Step 2: 确定必须保留的列
    if preserve_patterns is None:
        # SS 数据中 delay/loss 故障的真实根因是 latency，必须保留
        preserve_patterns = ["_latency"]
    if preserve_cols is None:
        preserve_cols = []

    mandatory_cols = set(preserve_cols)
    for pat in preserve_patterns:
        mandatory_cols.update([c for c in data.columns if pat in c])
    mandatory_cols = [c for c in data.columns if c in mandatory_cols]  # 保持原顺序

    # Step 3: 方差 Top-N 筛选（排除已强制保留的列）
    remaining_cols = [c for c in data.columns if c not in mandatory_cols]
    n_slots = max(max_features - len(mandatory_cols), 0)

    if n_slots > 0 and len(remaining_cols) > n_slots:
        variances = data[remaining_cols].var().sort_values(ascending=False)
        selected_cols = variances.head(n_slots).index.tolist()
    else:
        selected_cols = remaining_cols

    final_cols = mandatory_cols + [c for c in selected_cols if c not in mandatory_cols]
    data = data[final_cols]

    if verbose:
        print(f"  [FEAT_SEL] mandatory={len(mandatory_cols)}, selected={len(selected_cols)}, "
              f"total={data.shape[1]} ({n_before} → {data.shape[1]})")

    return data


# ============================================================
# PageRank 排序
# ============================================================
def run_pagerank(adj_matrix, node_names):
    """
    对 PCMCI 邻接矩阵运行 PageRank，返回节点排名。

    Args:
        adj_matrix: 因果邻接矩阵 (N, N)
        node_names: 节点名称列表

    Returns:
        ranks: 按分数降序排列的节点名列表
        scores: 对应的 PageRank 分数
    """
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)

    if adj_float.sum() == 0:
        # 无边 → 按原顺序
        return list(node_names), [0.0] * n

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj_float[i, j] != 0:
                G.add_edge(j, i)  # j causes i → edge j→i

    pr_scores = nx.pagerank(G, alpha=0.85)
    scored = [(node_names[i], pr_scores.get(i, 0.0)) for i in range(n)]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    ranks = [x[0] for x in scored_sorted]
    scores = [x[1] for x in scored_sorted]
    return ranks, scores


# ============================================================
# 评估指标
# ============================================================
def compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5)):
    """
    计算 Top-K Accuracy（基于邻接矩阵非零边密度）。
    """
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)
    metrics = {}

    for k in top_k_list:
        correct = 0
        total = 0
        for i in range(n):
            top_k_indices = np.argsort(-adj_float[i])[:k]
            for j in top_k_indices:
                if adj_float[i, j] != 0:
                    correct += 1
                    break
            total += 1
        metrics[f"Top-{k} Acc"] = correct / total if total > 0 else 0

    return metrics


def compute_rank_accuracy(ranks, service, fault_type, top_k_list=(1, 3, 5)):
    """
    计算根因排名的 Top-K Accuracy。
    - Service-level: 排名中是否包含 {service} 开头的节点
    - Metric-level: 排名中是否包含 {service}_{mapped_metric} 节点
    """
    mapped_metric = FAULT_TO_METRIC.get(fault_type, fault_type)
    truth_service = service
    truth_metric = f"{service}_{mapped_metric}"

    metrics = {}

    # Service-level
    for k in top_k_list:
        top_k = ranks[:k]
        hit = any(n.split("_")[0] == truth_service for n in top_k)
        metrics[f"Svc-AC@{k}"] = 1.0 if hit else 0.0

    # Metric-level
    for k in top_k_list:
        top_k = ranks[:k]
        hit = truth_metric in top_k
        metrics[f"Mtr-AC@{k}"] = 1.0 if hit else 0.0

    # 排名位置
    try:
        svc_rank = next(i + 1 for i, n in enumerate(ranks) if n.split("_")[0] == truth_service)
    except StopIteration:
        svc_rank = len(ranks) + 1
    try:
        mtr_rank = ranks.index(truth_metric) + 1
    except ValueError:
        mtr_rank = len(ranks) + 1

    metrics["Svc_Rank"] = svc_rank
    metrics["Mtr_Rank"] = mtr_rank

    return metrics


# ============================================================
# 可视化
# ============================================================
def visualize_causal_graph(adj_matrix, node_names, output_path, title="PCMCI Causal Graph"):
    """可视化因果图"""
    try:
        n = adj_matrix.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        adj_float = np.array(adj_matrix, dtype=float)
        for i in range(n):
            for j in range(n):
                if adj_float[i, j] != 0:
                    G.add_edge(j, i)

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(n, 1)))
        nx.draw(G, pos, with_labels=True, labels={i: node_names[i] for i in range(n)},
                node_size=600, font_size=7, arrows=True, edge_color='blue',
                alpha=0.7, node_color='lightblue')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return True
    except Exception as e:
        print(f"  [WARN] 可视化失败: {e}")
        return False


# ============================================================
# 主流程
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PCMCI 因果推断基线复现")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="数据根目录")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--tau_max", type=int, default=TAU_MAX,
                        help="最大滞后阶数 (默认 3)")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="显著性水平 (默认 0.05)")
    parser.add_argument("--test", action="store_true",
                        help="冒烟测试模式（只跑前 2 个案例）")
    parser.add_argument("--window", type=int, default=WINDOW_LENGTH_MIN,
                        help="窗口长度（分钟），默认 20")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                        help=f"特征选择保留的最大变量数 (默认 {MAX_FEATURES})")
    parser.add_argument("--drop-node-metrics", action="store_true",
                        default=DROP_NODE_METRICS,
                        help="丢弃节点级指标（192-* 开头的列）")
    parser.add_argument("--keep-node-metrics", action="store_true",
                        default=False,
                        help="保留节点级指标（覆盖 --drop-node-metrics）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过已有输出文件的案例，只处理未完成的")
    parser.add_argument("--preserve-patterns", nargs="+",
                        default=["_latency", "latency"],
                        help="强制保留的列名模式（默认保留所有 latency 列）")
    return parser.parse_args()


# ============================================================
# 从已有输出文件重建完整汇总（断点续跑后使用）
# ============================================================
def rebuild_summary_from_outputs(output_dir, top_k_list=(1, 3, 5)):
    """
    扫描 output_dir/RE1-SS/ 下所有已完成的案例输出，
    重建完整的汇总结果（含评估指标）。

    Returns:
        all_results: list[dict]，每个案例一条记录
    """
    import json as _json

    re1_dir = join(output_dir, "RE1-SS")
    if not os.path.exists(re1_dir):
        return []

    all_results = []

    for svc_fault_dir in sorted(os.listdir(re1_dir)):
        dir_path = join(re1_dir, svc_fault_dir)
        if not os.path.isdir(dir_path):
            continue

        try:
            service, fault_type = svc_fault_dir.split("_", 1)
        except ValueError:
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith("_adj_matrix.npy"):
                continue

            case_id = int(fname.replace("_adj_matrix.npy", ""))

            adj_path = join(dir_path, fname)
            nodes_path = join(dir_path, f"{case_id}_node_names.json")
            ranks_path = join(dir_path, f"{case_id}_ranks.csv")

            if not (os.path.exists(nodes_path) and os.path.exists(ranks_path)):
                continue

            try:
                adj_matrix = np.load(adj_path)
                with open(nodes_path, "r") as f:
                    node_names = _json.load(f)
                rank_df = pd.read_csv(ranks_path)
                ranks = rank_df["node"].tolist()
                scores = rank_df["pagerank_score"].tolist()

                n_nodes = len(node_names)
                n_edges = int(np.array(adj_matrix).sum())

                topk_metrics = compute_top_k_accuracy(adj_matrix, top_k_list=top_k_list)
                rank_metrics = compute_rank_accuracy(ranks, service, fault_type, top_k_list=top_k_list)

                result = {
                    "dataset": "RE1-SS",
                    "service": service,
                    "fault_type": fault_type,
                    "case_id": case_id,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "edge_density": round(n_edges / max(n_nodes * (n_nodes - 1), 1), 4),
                    "elapsed_sec": 0,
                    "top1_node": ranks[0] if ranks else "",
                    "top3_nodes": "|".join(ranks[:3]),
                    "ground_truth_service": service,
                    "ground_truth_metric": FAULT_TO_METRIC.get(fault_type, fault_type),
                    **topk_metrics,
                    **rank_metrics,
                }
                all_results.append(result)

            except Exception as e:
                print(f"  [WARN] 重建 {svc_fault_dir}/{case_id} 失败: {e}")
                continue

    return all_results


def main():
    args = parse_args()

    # 处理 --keep-node-metrics 覆盖
    if args.keep_node_metrics:
        args.drop_node_metrics = False

    # 处理 preserve_patterns 默认去重
    args.preserve_patterns = list(dict.fromkeys(args.preserve_patterns))

    os.makedirs(args.output, exist_ok=True)

    # === 扫描所有 data.csv ===
    pattern = join(args.data_root, "**", "data.csv")
    all_data_paths = sorted(glob.glob(pattern, recursive=True))

    if not all_data_paths:
        print(f"错误：在 {args.data_root} 下未找到 data.csv 文件！")
        sys.exit(1)

    if args.test:
        all_data_paths = all_data_paths[:2]
        print(f"[TEST] 仅处理前 {len(all_data_paths)} 个案例")

    print("=" * 70)
    print("PCMCI 因果推断基线复现")
    print(f"  数据根目录: {args.data_root}")
    print(f"  案例数量:   {len(all_data_paths)}")
    print(f"  tau_max:    {args.tau_max}")
    print(f"  alpha:      {args.alpha}")
    print(f"  窗口长度:   {args.window} min")
    print(f"  最大特征数: {args.max_features}")
    print(f"  丢弃节点指标: {'是' if args.drop_node_metrics else '否'}")
    print(f"  输出目录:   {args.output}")
    print("=" * 70)

    all_results = []
    start_time = time.time()
    progress_path = join(args.output, "progress.json")

    n_skipped = 0
    n_total = len(all_data_paths)
    for idx, data_path in enumerate(all_data_paths, start=1):
        # 解析路径信息: data/RE1-SS/{service}_{fault_type}/{case_id}/data.csv
        case_id = int(basename(dirname(data_path)))
        service_metric = basename(dirname(dirname(data_path)))
        service, fault_type = service_metric.split("_", 1)

        # === 断点续跑：跳过已有输出的案例 ===
        if args.resume:
            case_dir = join(args.output, "RE1-SS", f"{service}_{fault_type}")
            adj_path = join(case_dir, f"{case_id}_adj_matrix.npy")
            nodes_path = join(case_dir, f"{case_id}_node_names.json")
            ranks_path = join(case_dir, f"{case_id}_ranks.csv")
            if os.path.exists(adj_path) and os.path.exists(nodes_path) and os.path.exists(ranks_path):
                n_skipped += 1
                # 实时更新进度
                prog = {
                    "current": idx,
                    "total": n_total,
                    "skipped": n_skipped,
                    "completed": idx - n_skipped,
                    "case": f"{service}_{fault_type}/{case_id}",
                    "status": "skipped",
                    "elapsed_total_sec": round(time.time() - start_time, 2),
                }
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                continue

        # 打印进度
        print(f"\n[{idx}/{n_total}] 开始处理: {service}_{fault_type}/{case_id}", flush=True)

        try:
            # === 1. 预处理 ===
            data, inject_time, svc, fault, sli = load_and_preprocess(
                data_path,
                window_length_min=args.window,
                verbose=False,
            )

            if data.shape[1] < 2:
                print(f"\n  [SKIP] {service}_{fault_type}/{case_id}: 有效列数不足 ({data.shape[1]})")
                continue

            # === 1.5 特征选择（SS 数据列数多，需筛选以保证 PCMCI 可计算） ===
            # 强制保留 SLI 与所有 latency 指标，避免 delay/loss 故障丢失关键根因列
            data = select_features(
                data,
                max_features=args.max_features,
                drop_node_metrics=args.drop_node_metrics,
                preserve_cols=[sli] if sli in data.columns else None,
                preserve_patterns=args.preserve_patterns,
                verbose=False,
            )

            # === 2. 运行 PCMCI ===
            t0 = time.time()
            adj_matrix = pcmci(data, tau_max=args.tau_max, alpha=args.alpha)
            elapsed = time.time() - t0

            node_names = data.columns.to_list()
            n_nodes = len(node_names)
            n_edges = int(np.array(adj_matrix).sum())

            # === 3. 保存邻接矩阵 ===
            case_dir = join(args.output, "RE1-SS", f"{service}_{fault_type}")
            os.makedirs(case_dir, exist_ok=True)

            adj_path = join(case_dir, f"{case_id}_adj_matrix.npy")
            np.save(adj_path, np.array(adj_matrix))

            # 保存节点名
            nodes_path = join(case_dir, f"{case_id}_node_names.json")
            import json
            with open(nodes_path, "w") as f:
                json.dump(node_names, f, ensure_ascii=False)

            # === 4. PageRank 排序 ===
            ranks, scores = run_pagerank(adj_matrix, node_names)

            # 保存排名
            rank_df = pd.DataFrame({
                "rank": range(1, len(ranks) + 1),
                "node": ranks,
                "pagerank_score": scores,
            })
            rank_df.to_csv(join(case_dir, f"{case_id}_ranks.csv"), index=False)

            # === 5. 评估指标 ===
            # Top-K Accuracy（邻接矩阵层面）
            topk_metrics = compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5))

            # 根因排名准确率
            rank_metrics = compute_rank_accuracy(ranks, service, fault_type, top_k_list=(1, 3, 5))

            # 汇总
            result = {
                "dataset": "RE1-SS",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "edge_density": round(n_edges / max(n_nodes * (n_nodes - 1), 1), 4),
                "elapsed_sec": round(elapsed, 2),
                "top1_node": ranks[0] if ranks else "",
                "top3_nodes": "|".join(ranks[:3]),
                "ground_truth_service": service,
                "ground_truth_metric": FAULT_TO_METRIC.get(fault_type, fault_type),
                **topk_metrics,
                **rank_metrics,
            }
            all_results.append(result)

            # === 6. 可视化（仅前几个案例） ===
            if len(all_results) <= 3:
                graph_path = join(case_dir, f"{case_id}_causal_graph.png")
                visualize_causal_graph(
                    adj_matrix, node_names, graph_path,
                    title=f"PCMCI Causal Graph - {service}_{fault_type}/#{case_id}"
                )

            # 实时写入进度
            prog = {
                "current": idx,
                "total": n_total,
                "skipped": n_skipped,
                "completed": len(all_results),
                "case": f"{service}_{fault_type}/{case_id}",
                "status": "success",
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "elapsed_case_sec": elapsed,
                "elapsed_total_sec": round(time.time() - start_time, 2),
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(prog, f, ensure_ascii=False, indent=2)

            print(f"  ✓ 完成: {service}_{fault_type}/{case_id} | 节点={n_nodes}, 边={n_edges}, 耗时={elapsed:.1f}s", flush=True)

        except Exception as e:
            print(f"\n  [ERROR] {service}_{fault_type}/{case_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "dataset": "RE1-SS",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "n_nodes": 0,
                "n_edges": 0,
                "edge_density": 0,
                "elapsed_sec": 0,
                "top1_node": "",
                "top3_nodes": "",
                "ground_truth_service": service,
                "ground_truth_metric": FAULT_TO_METRIC.get(fault_type, fault_type),
                "Top-1 Acc": 0, "Top-3 Acc": 0, "Top-5 Acc": 0,
                "Svc-AC@1": 0, "Svc-AC@3": 0, "Svc-AC@5": 0,
                "Mtr-AC@1": 0, "Mtr-AC@3": 0, "Mtr-AC@5": 0,
                "Svc_Rank": 0, "Mtr_Rank": 0,
                "error": str(e),
            })
            # 失败也写入进度
            prog = {
                "current": idx,
                "total": n_total,
                "skipped": n_skipped,
                "completed": len(all_results),
                "case": f"{service}_{fault_type}/{case_id}",
                "status": "error",
                "error": str(e),
                "elapsed_total_sec": round(time.time() - start_time, 2),
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(prog, f, ensure_ascii=False, indent=2)
            continue

    # === 汇总 ===
    elapsed_total = time.time() - start_time

    # 断点续跑模式下，从所有输出文件重建完整汇总
    if args.resume:
        print(f"\n[RESUME] 跳过了 {n_skipped} 个已完成案例，新处理了 {len(all_results)} 个案例")
        all_results = rebuild_summary_from_outputs(args.output)

    if not all_results:
        print("\n错误：没有成功处理任何案例！")
        sys.exit(1)

    summary_df = pd.DataFrame(all_results)
    summary_path = join(args.output, "pcmci_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # === 计算总体评估 ===
    n_success = len([r for r in all_results if "error" not in r])
    n_fail = len(all_results) - n_success

    print(f"\n{'=' * 70}")
    print(f"PCMCI 复现完成！")
    if args.resume:
        print(f"  (断点续跑模式：跳过 {n_skipped}，新处理 {len(all_results) - n_skipped if n_skipped else len(all_results)}，总计 {len(all_results)})")
    print(f"  总案例数:   {len(all_results)}")
    print(f"  成功:       {n_success}")
    print(f"  失败:       {n_fail}")
    print(f"  本次耗时:   {elapsed_total:.1f}s")
    print(f"  结果目录:   {args.output}/")

    # 总体评估指标
    if n_success > 0:
        success_df = summary_df[summary_df.get("error", pd.Series(dtype=str)).isna()] if "error" in summary_df.columns else summary_df
        # 更安全的过滤方式
        success_df = summary_df[~summary_df.index.isin(
            [i for i, r in enumerate(all_results) if "error" in r]
        )] if "error" in summary_df.columns else summary_df

        print(f"\n{'=' * 70}")
        print(f"评估指标 (n={len(success_df)} 成功案例)")
        print(f"{'=' * 70}")
        print(f"\n{'指标':<20} {'值':>10}")
        print("-" * 35)

        eval_metrics = {
            "Svc-AC@1":  success_df["Svc-AC@1"].mean(),
            "Svc-AC@3":  success_df["Svc-AC@3"].mean(),
            "Svc-AC@5":  success_df["Svc-AC@5"].mean(),
            "Mtr-AC@1":  success_df["Mtr-AC@1"].mean(),
            "Mtr-AC@3":  success_df["Mtr-AC@3"].mean(),
            "Mtr-AC@5":  success_df["Mtr-AC@5"].mean(),
            "Avg Svc_Rank": success_df["Svc_Rank"].mean(),
            "Avg Mtr_Rank": success_df["Mtr_Rank"].mean(),
            "Top-1 Acc": success_df["Top-1 Acc"].mean(),
            "Top-3 Acc": success_df["Top-3 Acc"].mean(),
            "Top-5 Acc": success_df["Top-5 Acc"].mean(),
            "Avg n_nodes": success_df["n_nodes"].mean(),
            "Avg n_edges": success_df["n_edges"].mean(),
        }

        eval_rows = []
        for k, v in eval_metrics.items():
            print(f"  {k:<20} {v:>10.4f}")
            eval_rows.append({"metric": k, "value": round(v, 4)})

        # 按故障类型分组
        print(f"\n{'=' * 70}")
        print(f"按故障类型分组评估")
        print(f"{'=' * 70}")
        print(f"\n{'故障类型':<12} {'案例数':>6} {'Svc-AC@1':>10} {'Svc-AC@3':>10} {'Svc-AC@5':>10} {'Mtr-AC@1':>10}")
        print("-" * 60)

        fault_eval_rows = []
        for ft in sorted(success_df["fault_type"].unique()):
            ft_df = success_df[success_df["fault_type"] == ft]
            row = {
                "fault_type": ft,
                "n_cases": len(ft_df),
                "svc_ac1": ft_df["Svc-AC@1"].mean(),
                "svc_ac3": ft_df["Svc-AC@3"].mean(),
                "svc_ac5": ft_df["Svc-AC@5"].mean(),
                "mtr_ac1": ft_df["Mtr-AC@1"].mean(),
                "mtr_ac3": ft_df["Mtr-AC@3"].mean(),
                "mtr_ac5": ft_df["Mtr-AC@5"].mean(),
            }
            fault_eval_rows.append(row)
            print(f"  {ft:<12} {len(ft_df):>6} {row['svc_ac1']:>10.4f} {row['svc_ac3']:>10.4f} "
                  f"{row['svc_ac5']:>10.4f} {row['mtr_ac1']:>10.4f}")

        # 保存评估结果
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(join(args.output, "pcmci_metrics.csv"), index=False)

        fault_eval_df = pd.DataFrame(fault_eval_rows)
        fault_eval_df.to_csv(join(args.output, "pcmci_eval_by_fault.csv"), index=False)

        print(f"\n💾 评估指标已保存至: {join(args.output, 'pcmci_metrics.csv')}")
        print(f"💾 故障类型评估已保存至: {join(args.output, 'pcmci_eval_by_fault.csv')}")

    print(f"💾 汇总结果已保存至: {summary_path}")

    # === PCMCI 假设验证说明 ===
    print(f"\n{'=' * 70}")
    print("PCMCI 假设验证:")
    print("  - 因果平稳性: 微服务系统故障期间是否满足？→ 需在论文中讨论")
    print("  - 无隐藏变量: 当前数据是否满足条件？→ 需结合领域知识判断")
    print("  → 建议: 在报告中说明此假设的合理性，并指出潜在偏差")
    print(f"{'=' * 70}")
    print(f"\n✅ 全部流程完成！结果保存在: {args.output}")


if __name__ == "__main__":
    main()
