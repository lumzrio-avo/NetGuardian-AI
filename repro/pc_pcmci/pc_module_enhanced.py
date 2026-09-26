# -*- coding: utf-8 -*-
"""
pc_module_enhanced.py — Enhanced PC 算法基线模块

针对 RE1-SS 高维小样本问题的改进:
  1. 特征预筛选 (Feature Pre-selection): 按 anomaly score 排序 + 去高相关冗余,
     将变量数从 ~231 降至 ~40 (接近 OB 数据维度)
  2. 全窗口数据: 使用 normal+anomaly 全部样本 (vs 仅 normal 窗口),
     增加有效样本量 (288 → 721)
  3. 放宽 alpha: 0.05 → 0.1, 降低 Fisher-z 独立性检验门槛

Author: 大创团队
Date:   2026-08-24
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import networkx as nx

# Windows 控制台 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from causallearn.search.ConstraintBased.PC import pc as causallearn_pc

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common_features import select_features as _unified_select  # 统一变量筛选


# ============================================================
# 特征预筛选
# ============================================================
# ============================================================
# 辅助函数
# ============================================================
def _pc_graph_to_adj_matrix(cg_graph: np.ndarray, n: int) -> np.ndarray:
    """
    将 causallearn PC 图结构转换为二值邻接矩阵。

    causallearn 编码规则:
      graph[i, j] = -1, graph[j, i] = 1  → i → j  (有向边, i 是因)
      graph[i, j] = -1, graph[j, i] = -1  → i — j  (无向边)
      graph[i, j] = 0,  graph[j, i] = 0   → 无边

    返回: adj[i, j] = 1 表示 i 可能是 j 的原因
    """
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cg_graph[i, j] == -1:
                adj[i, j] = 1.0
    return adj


def _run_pagerank(adj_matrix: np.ndarray, node_names: list) -> tuple:
    """
    PageRank 排序 — adj[i, j] = 1 表示 i 可能是 j 的原因。
    PageRank 图中构建边: effect(j) → cause(i)，使原因节点获得更高分数。
    """
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)

    if adj_float.sum() == 0:
        return list(node_names), [0.0] * n

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj_float[i, j] != 0:
                G.add_edge(j, i)  # j (effect) → i (cause)

    pr_scores = nx.pagerank(G, alpha=0.85)
    scored = [(node_names[i], pr_scores.get(i, 0.0)) for i in range(n)]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    ranks = [x[0] for x in scored_sorted]
    scores = [x[1] for x in scored_sorted]
    return ranks, scores


# ============================================================
# 核心函数 (Enhanced)
# ============================================================
def run_pc_baseline_enhanced(
    data,
    metric_names: list,
    anomaly_scores,
    output_dir: str,
    t_fail: float,
    alpha: float = 0.1,
    indep_test: str = "fisherz",
    max_features: int = 40,
    corr_threshold: float = 0.95,
    use_full_window: bool = True,
) -> dict:
    """
    Enhanced PC 算法基线 — 特征预筛选 + 全窗口 + 放宽 alpha。

    参数:
        data: np.ndarray 或 pd.DataFrame, 形状 (T, N)
        metric_names: list[str], N 个指标名称
        anomaly_scores: np.ndarray, 形状 (N,), 每个指标的异常分数
        output_dir: str, 输出目录
        t_fail: float, 故障起始索引
        alpha: float, 显著性水平 (默认 0.1, 比原始 0.05 更宽松)
        indep_test: str, 独立性检验方法
        max_features: int, 预筛选后最大变量数 (默认 40)
        corr_threshold: float, 去冗余相关系数阈值 (默认 0.95)
        use_full_window: bool, 是否使用全窗口数据 (默认 True)

    返回:
        dict: adj_matrix, node_names, ranks, scores, n_nodes, n_edges, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    # === 日志配置 ===
    log_path = os.path.join(output_dir, "run_log_pc.txt")
    err_path = os.path.join(output_dir, "run_err_pc.txt")

    logger = logging.getLogger(f"pc_enhanced_{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    eh = logging.FileHandler(err_path, mode="a", encoding="utf-8")
    eh.setLevel(logging.WARNING)
    eh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(eh)

    # === 数据准备 ===
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy(dtype=float)
    else:
        data_np = np.array(data, dtype=float)

    n_total_rows, n_metrics = data_np.shape
    logger.info(f"原始数据形状: ({n_total_rows}, {n_metrics})")
    logger.info(f"t_fail (故障索引): {t_fail}")

    # === 处理常量列 ===
    col_std = data_np.std(axis=0)
    non_const_idx = np.where(col_std > 1e-10)[0]

    if len(non_const_idx) < n_metrics:
        const_names = [
            metric_names[i] for i in range(n_metrics) if i not in non_const_idx
        ]
        logger.warning(
            f"检测到 {n_metrics - len(non_const_idx)} 个常量列, 已剔除: {const_names[:10]}..."
        )

    data_valid = data_np[:, non_const_idx]
    valid_names = [metric_names[i] for i in non_const_idx]
    valid_anomaly = np.array(anomaly_scores, dtype=float)[non_const_idx]
    n_valid = len(valid_names)
    logger.info(f"剔除常量列后: {n_valid} 个变量")

    # === 处理 NaN / Inf ===
    if np.isnan(data_valid).any() or np.isinf(data_valid).any():
        logger.warning("数据中存在 NaN/Inf, 进行替换处理")
        data_valid = np.nan_to_num(data_valid, nan=0.0, posinf=0.0, neginf=0.0)

    # === ★ 统一变量筛选（2026-09-25 起） ===
    # 原实现 _preselect_features 是 RE1-SS 专用（按 anomaly score 排序 + 去相关），
    # 与 OB/TT/PCMCI 的四套规则互不相同，导致图规模不可比。
    # 现统一为 common_features.select_features：
    #   删非特征列 → 删节点级指标 → 删常量列 → 通用保留 *_latency → 方差 Top-N → 统一去冗余
    logger.info(f"统一变量筛选: max_features={max_features}, corr_threshold={corr_threshold}")
    _sel_df = _unified_select(
        pd.DataFrame(data_valid, columns=valid_names),
        max_features=max_features,
        drop_node_metrics=True,
        corr_threshold=corr_threshold,
        verbose=False,
    )
    selected_names = list(_sel_df.columns)
    data_selected = _sel_df.to_numpy(dtype=float)
    n_selected = len(selected_names)
    logger.info(
        f"统一筛选后: {n_selected} 个变量 (从 {n_valid} 降维, "
        f"保留率 {n_selected / max(n_valid, 1):.1%})"
    )
    logger.info(f"选中变量: {selected_names[:10]}...")

    # === ★ 数据窗口选择 (核心改进) ===
    if use_full_window:
        # 使用全窗口数据 (normal + anomaly)
        pc_data = data_selected
        logger.info(f"使用全窗口数据: shape={pc_data.shape} (normal+anomaly)")
    else:
        # 仅使用正常窗口 (原始行为)
        normal_end = int(t_fail * 0.8)
        if normal_end < 10:
            normal_end = min(10, n_total_rows)
        pc_data = data_selected[:normal_end]
        logger.info(f"使用正常窗口数据: shape={pc_data.shape} (前 {normal_end} 行)")

    # === 运行 PC 算法 ===
    logger.info(
        f"运行 PC 算法 (Enhanced): alpha={alpha}, indep_test={indep_test}, "
        f"样本数={pc_data.shape[0]}, 变量数={pc_data.shape[1]}"
    )

    try:
        cg = causallearn_pc(
            pc_data,
            alpha=alpha,
            indep_test=indep_test,
            show_progress=False,
        )
        raw_graph = cg.G.graph
        logger.info(f"PC 算法完成, 原始图形状: {raw_graph.shape}")
    except Exception as e:
        logger.error(f"PC 算法执行失败: {e}", exc_info=True)
        raw_graph = np.zeros((n_selected, n_selected), dtype=int)

    # === 提取邻接矩阵 ===
    adj_matrix = _pc_graph_to_adj_matrix(raw_graph, n_selected)
    n_edges = int(adj_matrix.sum())
    edge_density = n_edges / max(n_selected * (n_selected - 1), 1)
    logger.info(
        f"邻接矩阵: shape={adj_matrix.shape}, edges={n_edges}, "
        f"density={edge_density:.4f}"
    )

    # 保存邻接矩阵
    adj_df = pd.DataFrame(adj_matrix, index=selected_names, columns=selected_names)
    adj_path = os.path.join(output_dir, "pc_adj_matrix.csv")
    adj_df.to_csv(adj_path)
    logger.info(f"邻接矩阵已保存: {adj_path}")

    # === PageRank 排序 ===
    ranks, scores = _run_pagerank(adj_matrix, selected_names)
    logger.info(
        f"PageRank 排序完成, Top-5: {list(zip(ranks[:5], [round(s, 4) for s in scores[:5]]))}"
    )

    # 保存排名
    rank_df = pd.DataFrame(
        {
            "metric": ranks,
            "score": scores,
            "rank": list(range(1, len(ranks) + 1)),
        }
    )
    summary_path = os.path.join(output_dir, "pc_summary.csv")
    rank_df.to_csv(summary_path, index=False)
    logger.info(f"排名已保存: {summary_path}")

    # === 关闭 handler ===
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)

    return {
        "adj_matrix": adj_matrix,
        "node_names": selected_names,
        "ranks": ranks,
        "scores": scores,
        "n_nodes": n_selected,
        "n_edges": n_edges,
        "edge_density": round(edge_density, 4),
        "n_original": n_valid,
        "n_selected": n_selected,
    }
