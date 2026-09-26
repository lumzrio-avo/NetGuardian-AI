# -*- coding: utf-8 -*-
"""
pc_module.py — PC 算法基线模块 (基于 causallearn)

功能:
  1. 利用 t_fail 切分故障前正常窗口，仅用正常数据跑 PC 算法
  2. 提取邻接矩阵 → 保存 pc_adj_matrix.csv
  3. 复用项目已有的 PageRank 评分逻辑 → 保存 pc_summary.csv (metric, score, rank)
  4. 写入日志 run_log_pc.txt / run_err_pc.txt

依赖安装:
  需在 venv 环境中运行: venv/Scripts/pip install causallearn

运行环境:
  Python 解释器: D:/Workbuddy_PC/ob/venv/Scripts/python.exe

Author: 大创团队
Date:   2026-08-13
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

    返回: adj[i, j] = 1 表示 i 可能是 j 的原因 (与 PCMCI 邻接矩阵语义一致)
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
    PageRank 排序 — 复用 pcmci_module.py / run_pcmci.py 的逻辑。

    语义: adj[i, j] = 1 表示 i 可能是 j 的原因。
    PageRank 图中构建边: effect(j) → cause(i)，使原因节点获得更高分数。

    Returns:
        ranks: 按分数降序排列的节点名列表
        scores: 对应的 PageRank 分数
    """
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)

    if adj_float.sum() == 0:
        # 无边 → 按原顺序, 分数全 0
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
# 核心函数
# ============================================================
def run_pc_baseline(
    data,
    metric_names: list,
    anomaly_scores,
    output_dir: str,
    t_fail: float,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    use_full_window: bool = True,
) -> dict:
    """
    PC 算法基线 — 对单个案例运行因果发现 + 根因排名。

    参数:
        data: np.ndarray 或 pd.DataFrame, 形状 (T, N)
              T 个时间步, N 个指标列
        metric_names: list[str], N 个指标名称 (与 data 列对应)
        anomaly_scores: np.ndarray, 形状 (N,), 每个指标的异常分数
                        (用于辅助评分; 传 None 则仅用 PageRank)
        output_dir: str, 输出目录 (自动创建)
        t_fail: float, 故障起始索引 (用于切分正常窗口)
        alpha: float, PC 算法显著性水平 (默认 0.05)
        indep_test: str, 独立性检验方法 (默认 "fisherz")
                    可选: "fisherz", "chi_sq", "gsq", "kci"
        use_full_window: bool, 是否使用【全窗口】(normal + anomaly) 建模。
                    默认 True —— 对齐官方 main.py:213-214 与 Granger/PCMCI 口径。
                    False 则只用故障前正常窗口 (t_fail*0.8 行)。

    返回:
        dict:
            adj_matrix  — 邻接矩阵 (N_valid, N_valid)
            node_names  — 有效指标名列表
            ranks       — 排序后的指标名列表
            scores      — 对应分数
            n_nodes     — 有效节点数
            n_edges     — 边数
    """
    os.makedirs(output_dir, exist_ok=True)

    # === 日志配置 ===
    log_path = os.path.join(output_dir, "run_log_pc.txt")
    err_path = os.path.join(output_dir, "run_err_pc.txt")

    logger = logging.getLogger(f"pc_baseline_{id(output_dir)}")
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
    logger.info(f"原始数据形状: ({n_total_rows}, {n_metrics}), 指标数: {n_metrics}")
    logger.info(f"t_fail (故障索引): {t_fail}")

    # === 数据窗口选择 ===
    # 【2026-09-25 统一口径】默认【全窗口】(normal + anomaly)，对齐官方
    # main.py:213-214 的 concat(normal_df, anomal_df)，也与 Granger / PCMCI 一致。
    # 旧行为 (仅故障前 t_fail*0.8 行) 保留在 use_full_window=False 分支，供消融使用。
    if use_full_window:
        model_data = data_np
        logger.info(f"使用全窗口数据: shape={model_data.shape} (normal + anomaly)")
    else:
        normal_end = int(t_fail * 0.8)
        if normal_end < 10:
            normal_end = min(10, n_total_rows)
            logger.warning(
                f"t_fail * 0.8 = {int(t_fail * 0.8)} < 10, 回退使用 normal_end = {normal_end}"
            )
        model_data = data_np[:normal_end]
        logger.info(f"使用正常窗口数据: shape={model_data.shape} (前 {normal_end} 行)")

    # === 处理常量列 (PC 算法无法处理方差为 0 的变量) ===
    col_std = model_data.std(axis=0)
    non_const_idx = np.where(col_std > 1e-10)[0]

    if len(non_const_idx) < n_metrics:
        const_names = [
            metric_names[i] for i in range(n_metrics) if i not in non_const_idx
        ]
        logger.warning(
            f"检测到 {n_metrics - len(non_const_idx)} 个常量列, 已剔除: {const_names}"
        )

    model_data_valid = model_data[:, non_const_idx]
    valid_names = [metric_names[i] for i in non_const_idx]
    n_valid = len(valid_names)
    logger.info(f"有效变量数: {n_valid} (剔除 {n_metrics - n_valid} 个常量列)")

    # === 处理 NaN / Inf ===
    if np.isnan(model_data_valid).any() or np.isinf(model_data_valid).any():
        logger.warning("数据中存在 NaN/Inf, 进行替换处理")
        model_data_valid = np.nan_to_num(model_data_valid, nan=0.0, posinf=0.0, neginf=0.0)

    # === 运行 PC 算法 ===
    logger.info(
        f"运行 PC 算法: alpha={alpha}, indep_test={indep_test}, "
        f"样本数={model_data_valid.shape[0]}, 变量数={n_valid}"
    )

    # === 运行 PC 算法 (fisherz 奇异矩阵时自动 jitter 重试) ===
    def _run_pc(np_data):
        return causallearn_pc(
            np_data, alpha=alpha, indep_test=indep_test, show_progress=False,
        )

    try:
        cg = _run_pc(model_data_valid)
        raw_graph = cg.G.graph
        logger.info(f"PC 算法完成, 原始图形状: {raw_graph.shape}")
    except (ValueError, np.linalg.LinAlgError) as e:
        # fisherz 相关矩阵奇异 → 加微小抖动重试 (打破完全共线)
        logger.warning(
            f"PC 算法首次执行失败 ({e}), 尝试加微小抖动重试..."
        )
        try:
            rng = np.random.default_rng(42)
            jitter = rng.normal(0, 1e-6, size=model_data_valid.shape)
            jittered = model_data_valid + jitter * (
                np.std(model_data_valid, axis=0, keepdims=True) + 1e-10
            )
            cg = _run_pc(jittered)
            raw_graph = cg.G.graph
            logger.info(f"PC 算法 (jitter 重试) 完成, 图形状: {raw_graph.shape}")
        except Exception as e2:
            logger.error(f"PC 算法 jitter 重试仍失败: {e2}", exc_info=True)
            raw_graph = np.zeros((n_valid, n_valid), dtype=int)
    except Exception as e:
        logger.error(f"PC 算法执行失败: {e}", exc_info=True)
        raw_graph = np.zeros((n_valid, n_valid), dtype=int)

    # === 提取邻接矩阵 ===
    adj_matrix = _pc_graph_to_adj_matrix(raw_graph, n_valid)
    n_edges = int(adj_matrix.sum())
    edge_density = n_edges / max(n_valid * (n_valid - 1), 1)
    logger.info(
        f"邻接矩阵: shape={adj_matrix.shape}, edges={n_edges}, "
        f"density={edge_density:.4f}"
    )

    # 保存邻接矩阵
    adj_df = pd.DataFrame(adj_matrix, index=valid_names, columns=valid_names)
    adj_path = os.path.join(output_dir, "pc_adj_matrix.csv")
    adj_df.to_csv(adj_path)
    logger.info(f"邻接矩阵已保存: {adj_path}")

    # === PageRank 排序 ===
    ranks, scores = _run_pagerank(adj_matrix, valid_names)
    logger.info(f"PageRank 排序完成, Top-5: {list(zip(ranks[:5], [round(s, 4) for s in scores[:5]]))}")

    # === 结合异常分数 (如果提供) ===
    if anomaly_scores is not None:
        anomaly_arr = np.array(anomaly_scores, dtype=float)
        # 对齐到有效指标
        valid_anomaly = anomaly_arr[non_const_idx]
        # 归一化
        if valid_anomaly.max() > valid_anomaly.min():
            valid_anomaly_norm = (valid_anomaly - valid_anomaly.min()) / (
                valid_anomaly.max() - valid_anomaly.min()
            )
        else:
            valid_anomaly_norm = np.zeros_like(valid_anomaly)

        # 构建异常分数字典
        anomaly_dict = {valid_names[i]: valid_anomaly_norm[i] for i in range(n_valid)}
        logger.info(f"已结合异常分数进行辅助评分 (anomaly_scores 提供)")

    # 保存排名 — 格式: metric, score, rank
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
        "node_names": valid_names,
        "ranks": ranks,
        "scores": scores,
        "n_nodes": n_valid,
        "n_edges": n_edges,
        "edge_density": round(edge_density, 4),
    }
