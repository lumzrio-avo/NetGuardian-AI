# -*- coding: utf-8 -*-
"""
run_rca_full.py — PC/PCMCI 因果根因分析完整运行脚本

修复要点:
  1. 正确使用本地 data/RE1-OB 路径 (不再引用 D:/WorkBuddy_PCMCI_OB)
  2. 不使用 --test 时默认跑全部 125 个 case
  3. 输出到新目录 output/RE1-OB-{algo}-full/，不覆盖之前结果
  4. 添加 --resume 断点续跑支持
  5. 异常 case 不跳过，生成随机/默认排名保证 evaluator 不中断
  6. 完整日志记录 (run_log + run_err)，便于定位跳过/失败原因
  7. 评估逻辑与 granger.py 对齐 (service-level + metric-level AC@k)

用法:
    # PC 算法 — 全部 125 个 case
    venv/Scripts/python.exe run_rca_full.py --algo pc

    # PCMCI 算法 — 全部 125 个 case
    venv/Scripts/python.exe run_rca_full.py --algo pcmci

    # 冒烟测试 (前 3 个 case)
    venv/Scripts/python.exe run_rca_full.py --algo pc --test

    # 断点续跑
    venv/Scripts/python.exe run_rca_full.py --algo pc --resume

    # 指定输出目录
    venv/Scripts/python.exe run_rca_full.py --algo pc --output output/my_pc_results

Author: 大创团队
Date:   2026-08-19
"""

import os
import sys
import glob
import json
import time
import random
import argparse
import warnings
import traceback
from os.path import basename, dirname, join

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from common_eval import FAULT_TO_METRIC  # noqa: E402  (统一口径)
from common_features import select_features as _unified_select  # noqa: E402  (统一变量筛选)

DATA_ROOT = join(SCRIPT_DIR, "data", "RE1-OB")
WINDOW_LENGTH_MIN = 20
# 【2026-09-25 统一口径】各方法使用【各自的官方默认超参数】。
# RCAEval 论文原文: "we adapt their available implementations and use the default
# hyperparameter settings recommended in their respective papers."
#   PC     : alpha = 0.05  (graph_construction/pc.py: pc_fisherz 系列显式 0.05;
#                           e2e/pc_pagerank.py 走 causallearn pc() 默认, 亦为 0.05)
#   PCMCI  : alpha = 0.20  (graph_construction/pcmci.py: pcmci(data, tau_max=3, alpha=0.2))
ALPHA_PC = 0.05
ALPHA_PCMCI = 0.20
ALPHA = ALPHA_PC            # 兼容旧引用；实际取值见 resolve_alpha()
TAU_MAX = 3
# 【2026-09-25 统一口径】变量筛选用 common_features 的统一规则
MAX_FEATURES = 50        # 变量数上限（三系统、两方法同一值）
DROP_NODE_METRICS = True # 是否丢弃节点级指标（192-* 前缀）

# 故障类型 -> 指标名映射 (对齐 RCAEval main.py)
# 【2026-09-25 统一口径】指标映射表已统一到 common_eval.FAULT_TO_METRIC
# （原先每个 runner 各写一份，disk/mem 的写法互不相同）


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_preprocess(data_path, window_length_min=20, verbose=False):
    """
    数据加载与预处理，严格对齐 granger.py / run_pcmci.py 的逻辑。

    步骤:
      1. pd.read_csv(data_path)
      2. 识别 OB 格式 -> 丢弃 _latency-50 列
      3. inf -> NaN -> ffill() -> fillna(0)
      4. 读取 inject_time.txt
      5. 按 inject_time 切分 normal/anomaly (各 window_length_min*60//2 点)
      6. 重命名 _latency-90 -> _latency
      7. 确定 SLI
      8. 清理非特征列 + 常量列

    Returns:
        data: pd.DataFrame
        inject_time: int
        service: str
        metric: str (原始 fault_type)
        sli: str
    """
    data_dir = dirname(data_path)
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)

    # Step 1: 读取 CSV
    df = pd.read_csv(data_path)
    if verbose:
        print(f"  [LOAD] {data_path}, shape={df.shape}")

    # Step 2: 识别 OB 格式 -> 丢弃低分位延迟列
    has_ob_latency = any(c.endswith("_latency-50") for c in df.columns)
    if has_ob_latency:
        df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
        latency_suffix_old = "_latency-90"
        latency_suffix_new = "_latency"
    else:
        latency_suffix_old = None

    # Step 3: 无穷值 + 缺失值处理
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    df = df.fillna(0)

    # Step 4: 读取故障注入时间
    inject_path = join(data_dir, "inject_time.txt")
    with open(inject_path, "r") as f:
        inject_time = int(f.readlines()[0].strip())
    if verbose:
        from datetime import datetime
        inject_dt = datetime.fromtimestamp(inject_time)
        print(f"  [INJECT] time={inject_time} ({inject_dt})")

    # Step 5: 按 inject_time 切分
    points = window_length_min * 60 // 2  # 20 min -> 600 points each
    normal_df = df[df["time"] < inject_time].tail(points)
    anomal_df = df[df["time"] >= inject_time].head(points)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    if verbose:
        print(f"  [SPLIT] normal={normal_df.shape}, anomaly={anomal_df.shape}, "
              f"combined={data.shape}")

    # Step 6: 重命名延迟列
    if latency_suffix_old:
        data = data.rename(
            columns={
                c: c.replace(latency_suffix_old, latency_suffix_new)
                for c in data.columns
                if c.endswith(latency_suffix_old)
            }
        )

    # Step 7: 确定 SLI
    sli = f"{service}_latency"
    if sli not in data.columns:
        if "front-end_cpu" in data.columns:
            sli = "front-end_cpu"
        else:
            sli = "frontend_latency"

    # Step 8: 清理非特征列
    drop_cols = [c for c in data.columns if c.lower() in ("time", "time.1") or c.startswith("time")]
    data = data.drop(columns=drop_cols, errors="ignore")

    # 丢弃常量列 (std == 0)
    const_cols = [c for c in data.columns if data[c].std() == 0]
    if const_cols:
        data = data.drop(columns=const_cols)
        if verbose:
            print(f"  [DROP_CONST] dropped {len(const_cols)} constant columns")

    return data, inject_time, service, metric, sli


# ============================================================
# PageRank 排序 (复用 run_pcmci.py / pc_module.py 逻辑)
# ============================================================
def run_pagerank(adj_matrix, node_names):
    """
    PageRank 排序: adj[i,j]=1 表示 i 可能是 j 的原因。
    图中构建边: effect(j) -> cause(i)，使原因节点获得更高分数。
    """
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)

    if adj_float.sum() == 0:
        return list(node_names), [0.0] * n

    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj_float[i, j] != 0:
                G.add_edge(j, i)  # j (effect) -> i (cause)

    pr_scores = nx.pagerank(G, alpha=0.85)
    scored = [(node_names[i], pr_scores.get(i, 0.0)) for i in range(n)]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    ranks = [x[0] for x in scored_sorted]
    scores = [x[1] for x in scored_sorted]
    return ranks, scores


# ============================================================
# PC 算法 (调用 pc_module)
# ============================================================
def run_pc_algorithm(data, metric_names, anomaly_scores, output_dir, t_fail, alpha,
                     indep_test, use_full_window=True):
    """调用 pc_module.run_pc_baseline，返回 adj_matrix + ranks。"""
    from pc_module import run_pc_baseline
    result = run_pc_baseline(
        data=data,
        metric_names=metric_names,
        anomaly_scores=anomaly_scores,
        output_dir=output_dir,
        t_fail=t_fail,
        alpha=alpha,
        indep_test=indep_test,
        use_full_window=use_full_window,
    )
    return result


# ============================================================
# PCMCI 算法 (调用 pcmci_module)
# ============================================================
def run_pcmci_algorithm(data, tau_max, alpha):
    """调用 pcmci_module.pcmci，返回 adj_matrix。"""
    from pcmci_module import pcmci
    adj_matrix = pcmci(data, tau_max=tau_max, alpha=alpha)
    return np.array(adj_matrix)


# ============================================================
# 异常分数计算
# ============================================================
def compute_anomaly_scores(data, window_length_min=20):
    """计算每个指标的异常分数 (故障前后均值变化比)。"""
    points = window_length_min * 60 // 2
    n_rows = data.shape[0]
    split_idx = min(points, n_rows // 2)

    normal_data = data.iloc[:split_idx]
    anomal_data = data.iloc[split_idx:]

    normal_mean = normal_data.mean()
    anomal_mean = anomal_data.mean()

    scores = np.abs(anomal_mean - normal_mean) / (np.abs(normal_mean) + 1e-10)
    return scores.to_numpy()
# ============================================================
# 【2026-09-25 统一口径】评估函数改为共享实现 common_eval.py
#   服务级: 先 '-db' 归一、再对服务实体序列去重 (官方 main.py:380)
#   指标级: 节点名精确等于 f"{service}_{FAULT_TO_METRIC[fault]}" (官方 main.py:394)
#   此前本文件自带的实现是「不去重 + 不归一」+ 各自独立的指标映射表，
#   导致 PC / PCMCI 三系统互不一致、且与 Granger 不可比。原实现已移除。
# ============================================================
from common_eval import compute_rank_accuracy as _official_rank_acc


def compute_rank_accuracy(ranks, service, fault_type, top_k_list=(1, 3, 5)):
    """官方口径的 Svc-AC@k / Mtr-AC@k / Svc_Rank / Mtr_Rank（实现见 common_eval.py）。"""
    return _official_rank_acc(ranks, service, fault_type, top_k_list=top_k_list)


def compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5)):
    """计算 Top-K Accuracy (基于邻接矩阵非零边密度)。"""
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


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PC/PCMCI 因果根因分析 — 完整运行脚本 (125 cases)"
    )
    parser.add_argument(
        "--algo", type=str, default="pc", choices=["pc", "pcmci"],
        help="算法选择: pc 或 pcmci (默认 pc)",
    )
    parser.add_argument(
        "--data-root", type=str, default=DATA_ROOT,
        help=f"数据根目录 (默认 {DATA_ROOT})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录 (默认 output/RE1-OB-{algo}-full)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help=f"显著性水平 (默认 {ALPHA})",
    )
    parser.add_argument(
        "--tau-max", type=int, default=TAU_MAX,
        help=f"PCMCI 最大滞后阶数 (默认 {TAU_MAX})",
    )
    parser.add_argument(
        "--indep-test", type=str, default="fisherz",
        help="PC 算法独立性检验方法 (默认 fisherz)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="冒烟测试模式 (只跑前 3 个案例)",
    )
    parser.add_argument(
        "--normal-only", action="store_true",
        help="仅用故障前正常窗口建模（默认使用全窗口，与官方 main.py:213 一致）",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_LENGTH_MIN,
        help=f"窗口长度 (分钟), 默认 {WINDOW_LENGTH_MIN}",
    )
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                        help=f"变量筛选保留的上限 (默认 {MAX_FEATURES})")
    parser.add_argument("--drop-node-metrics", action="store_true",
                        default=DROP_NODE_METRICS,
                        help="丢弃节点级指标 (192-* 开头的列)")
    parser.add_argument("--keep-node-metrics", action="store_true",
                        default=False,
                        help="保留节点级指标 (覆盖 --drop-node-metrics)")
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续跑: 跳过已完成的案例",
    )
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    # === 【2026-09-25 统一口径】按算法的官方默认值解析 alpha ===
    # 不显式传 --alpha 时，PC 用 0.05、PCMCI 用 0.20（各自的官方默认值）。
    if args.alpha is None:
        args.alpha = ALPHA_PC if args.algo == "pc" else ALPHA_PCMCI

    if args.keep_node_metrics:
        args.drop_node_metrics = False

    # 确定输出目录 (不覆盖之前结果)
    if args.output is None:
        args.output = join(SCRIPT_DIR, "output", f"RE1-OB-{args.algo}-full")

    os.makedirs(args.output, exist_ok=True)

    # 顶层日志
    log_path = join(args.output, "run_log.txt")
    err_path = join(args.output, "run_err.txt")
    # 不清空旧日志 (append 模式，便于断点续跑时保留历史)
    # 但如果是全新运行且非 resume，则清空
    if not args.resume:
        open(log_path, "w").close()
        open(err_path, "w").close()

    def log_and_print(msg, level="INFO"):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")

    def log_error(msg):
        print(msg, file=sys.stderr)
        with open(err_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {msg}\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {msg}\n")

    # === 扫描所有 data.csv ===
    pattern = join(args.data_root, "**", "data.csv")
    all_data_paths = sorted(glob.glob(pattern, recursive=True))

    if not all_data_paths:
        log_error(f"错误: 在 {args.data_root} 下未找到 data.csv 文件!")
        sys.exit(1)

    if args.test:
        all_data_paths = all_data_paths[:3]
        log_and_print(f"[TEST] 仅处理前 {len(all_data_paths)} 个案例")

    # === Resume: 加载已完成的案例 ===
    existing_cases = set()
    if args.resume:
        summary_path = join(args.output, f"{args.algo}_summary.csv")
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            for _, row in existing_df.iterrows():
                existing_cases.add((
                    row.get("dataset", "RE1-OB"),
                    row.get("service", ""),
                    row.get("fault_type", ""),
                    int(row.get("case_id", 0)),
                ))
            log_and_print(f"[RESUME] 从 {args.algo}_summary.csv 加载了 {len(existing_cases)} 个已完成案例")

        # 同时扫描案例输出目录
        for svc_fault_dir in os.listdir(args.output) if os.path.isdir(args.output) else []:
            svc_fault_path = join(args.output, svc_fault_dir)
            if not os.path.isdir(svc_fault_path) or "_" not in svc_fault_dir:
                continue
            service, fault_type = svc_fault_dir.split("_", 1)
            for fn in os.listdir(svc_fault_path):
                case_dir = join(svc_fault_path, fn)
                if os.path.isdir(case_dir):
                    try:
                        case_id = int(fn)
                        key = ("RE1-OB", service, fault_type, case_id)
                        if key not in existing_cases:
                            existing_cases.add(key)
                    except ValueError:
                        continue

        if existing_cases:
            log_and_print(f"[RESUME] 共识别 {len(existing_cases)} 个已完成案例 (将被跳过)")

    log_and_print("=" * 70)
    log_and_print(f"{args.algo.upper()} 算法 — 因果发现 + 根因定位 (完整运行)")
    log_and_print(f"  数据根目录: {args.data_root}")
    log_and_print(f"  案例总数:   {len(all_data_paths)}")
    log_and_print(f"  算法:       {args.algo}")
    log_and_print(f"  alpha:      {args.alpha}")
    if args.algo == "pc":
        log_and_print(f"  indep_test: {args.indep_test}")
    else:
        log_and_print(f"  tau_max:    {args.tau_max}")
    log_and_print(f"  窗口长度:   {args.window} min")
    log_and_print(f"  输出目录:   {args.output}")
    log_and_print(f"  resume:     {args.resume}")
    log_and_print("=" * 70)

    all_results = []
    failed_cases = []          # 【2026-09-26】失败案例单独记录，不进入评估
    n_skip = 0
    n_success = 0
    n_fail = 0
    start_time = time.time()

    for idx, data_path in enumerate(all_data_paths):
        # 解析路径: data/RE1-OB/{service}_{fault_type}/{case_id}/data.csv
        case_id = int(basename(dirname(data_path)))
        service_metric = basename(dirname(dirname(data_path)))
        service, fault_type = service_metric.split("_", 1)

        case_label = f"{service}_{fault_type}/{case_id}"

        # Resume: 跳过已完成的
        if args.resume and ("RE1-OB", service, fault_type, case_id) in existing_cases:
            n_skip += 1
            continue

        log_and_print(f"\n[{idx + 1}/{len(all_data_paths)}] 处理 {case_label} ...")

        try:
            # === 1. 预处理 ===
            data, inject_time, svc, fault, sli = load_and_preprocess(
                data_path,
                window_length_min=args.window,
                verbose=False,
            )

            if data.shape[1] < 2:
                log_and_print(f"  [SKIP] {case_label}: 有效列数不足 ({data.shape[1]})")
                n_skip += 1
                continue

            # === 1.5 统一变量筛选（2026-09-25 起）===
            # 原实现对 RE1-OB 完全不筛选（只删常量列），与 SS/TT/PCMCI 的规则不同，
            # 导致三系统图规模不可比。现统一走 common_features.select_features。
            # RE1-OB 变量数（44~47）本就在 max_features 以内，主要影响是
            # 统一了"删节点级指标 / 通用保留 *_latency / 去冗余"这三步的行为。
            data = _unified_select(
                data,
                max_features=args.max_features,
                drop_node_metrics=args.drop_node_metrics,
                verbose=False,
            )

            metric_names = data.columns.tolist()
            n_metrics = len(metric_names)
            points = args.window * 60 // 2  # 正常窗口点数
            t_fail = float(points)  # 故障起始索引

            # === 2. 计算异常分数 ===
            anomaly_scores = compute_anomaly_scores(data, args.window)

            # === 3. 运行因果发现算法 ===
            case_output_dir = join(args.output, f"{service}_{fault_type}", str(case_id))

            t0 = time.time()

            if args.algo == "pc":
                result = run_pc_algorithm(
                    data=data,
                    metric_names=metric_names,
                    anomaly_scores=anomaly_scores,
                    output_dir=case_output_dir,
                    t_fail=t_fail,
                    alpha=args.alpha,
                    indep_test=args.indep_test,
                    use_full_window=not args.normal_only,
                )
                adj_matrix = result["adj_matrix"]
                ranks = result["ranks"]
                scores = result["scores"]
                n_nodes = result["n_nodes"]
                n_edges = result["n_edges"]
                edge_density = result["edge_density"]

            elif args.algo == "pcmci":
                adj_matrix = run_pcmci_algorithm(
                    data=data,
                    tau_max=args.tau_max,
                    alpha=args.alpha,
                )
                ranks, scores = run_pagerank(adj_matrix, metric_names)
                n_nodes = len(metric_names)
                n_edges = int(np.array(adj_matrix).sum())
                edge_density = round(n_edges / max(n_nodes * (n_nodes - 1), 1), 4)

                # 保存 PCMCI 案例结果
                os.makedirs(case_output_dir, exist_ok=True)
                np.save(join(case_output_dir, "pcmci_adj_matrix.npy"), np.array(adj_matrix))
                rank_df = pd.DataFrame({
                    "metric": ranks,
                    "score": scores,
                    "rank": list(range(1, len(ranks) + 1)),
                })
                rank_df.to_csv(join(case_output_dir, "pcmci_summary.csv"), index=False)

            elapsed = time.time() - t0

            log_and_print(
                f"  [OK] nodes={n_nodes}, edges={n_edges}, "
                f"density={edge_density:.4f}, elapsed={elapsed:.2f}s"
            )
            log_and_print(f"  [RANK] Top-5: {ranks[:5]}")

            # === 4. 评估指标 ===
            topk_metrics = compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5))
            rank_metrics = compute_rank_accuracy(ranks, service, fault_type, top_k_list=(1, 3, 5))

            # === 5. 汇总 ===
            row = {
                "dataset": "RE1-OB",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "edge_density": edge_density,
                "elapsed_sec": round(elapsed, 2),
                "top1_node": ranks[0] if ranks else "",
                "top3_nodes": "|".join(ranks[:3]),
                "ground_truth_service": service,
                "ground_truth_metric": FAULT_TO_METRIC.get(fault_type, fault_type),
                **topk_metrics,
                **rank_metrics,
            }
            all_results.append(row)
            n_success += 1

        except Exception as e:
            # 【2026-09-26 修正】原实现在此生成「随机打乱的默认排名」并计入汇总，
            # 使崩溃案例伪装成正常结果（n_edges=0 却仍带 Svc-AC 分数）。
            # 这曾导致 OB 全部案例崩溃（AttributeError: normal_only）
            # 被误读为「PC 在这份数据上输出空图」。
            # 现改为：失败案例不计入 all_results，单独记录并在汇总时显式报错。
            err_msg = f"  [ERROR] {case_label}: {type(e).__name__}: {e}"
            log_error(err_msg)
            traceback.print_exc()
            n_fail += 1
            failed_cases.append({
                "dataset": "RE1-OB",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

    # === 汇总 ===
    elapsed_total = time.time() - start_time

    if not all_results:
        log_error("错误: 没有成功处理任何案例!")
        sys.exit(1)

    summary_df = pd.DataFrame(all_results)
    summary_path = join(args.output, f"{args.algo}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    log_and_print(f"\n{'=' * 70}")
    log_and_print(f"{args.algo.upper()} 算法完成!")
    log_and_print(f"  总案例数:   {len(all_data_paths)}")
    log_and_print(f"  成功:       {n_success}")
    log_and_print(f"  失败:       {n_fail}")
    log_and_print(f"  跳过(resume): {n_skip}")
    log_and_print(f"  总耗时:     {elapsed_total:.1f}s (平均 {elapsed_total / max(n_success, 1):.1f}s/case)")
    log_and_print(f"  结果目录:   {args.output}/")

    # === 总体评估指标 ===
    # 【2026-09-26 修正】仅统计成功案例；失败案例单独列出，不再注入随机排名
    if n_fail:
        fail_path = join(args.output, "failed_cases.csv")
        pd.DataFrame(failed_cases).to_csv(fail_path, index=False)
        log_and_print(f"\n{'!' * 70}")
        log_and_print(f"!! 有 {n_fail} 个案例失败，已从评估中排除（结果不可直接使用）")
        log_and_print(f"!! 失败清单: {fail_path}")
        for _fc in failed_cases[:5]:
            log_and_print(f"!!   {_fc['service']}_{_fc['fault_type']}/{_fc['case_id']}: {_fc['error']}")
        log_and_print(f"{'!' * 70}")

    log_and_print(f"\n{'=' * 70}")
    log_and_print(f"评估指标 (n={len(all_results)} 个成功案例" +
                  (f"，另有 {n_fail} 个失败案例未计入)" if n_fail else ")"))
    log_and_print(f"{'=' * 70}")

    eval_metrics = {
        "Svc-AC@1":  summary_df["Svc-AC@1"].mean(),
        "Svc-AC@3":  summary_df["Svc-AC@3"].mean(),
        "Svc-AC@5":  summary_df["Svc-AC@5"].mean(),
        "Mtr-AC@1":  summary_df["Mtr-AC@1"].mean(),
        "Mtr-AC@3":  summary_df["Mtr-AC@3"].mean(),
        "Mtr-AC@5":  summary_df["Mtr-AC@5"].mean(),
        "Avg Svc_Rank": summary_df["Svc_Rank"].mean(),
        "Avg Mtr_Rank": summary_df["Mtr_Rank"].mean(),
        "Avg n_nodes": summary_df["n_nodes"].mean(),
        "Avg n_edges": summary_df["n_edges"].mean(),
    }

    eval_rows = []
    log_and_print(f"\n{'指标':<20} {'值':>10}")
    log_and_print("-" * 35)
    for k, v in eval_metrics.items():
        log_and_print(f"  {k:<20} {v:>10.4f}")
        eval_rows.append({"metric": k, "value": round(v, 4)})

    # === 按故障类型分组 ===
    log_and_print(f"\n{'=' * 70}")
    log_and_print(f"按故障类型分组评估")
    log_and_print(f"{'=' * 70}")
    log_and_print(
        f"\n{'故障类型':<12} {'案例数':>6} {'Svc-AC@1':>10} {'Svc-AC@3':>10} "
        f"{'Svc-AC@5':>10} {'Mtr-AC@1':>10} {'Mtr-AC@5':>10}"
    )
    log_and_print("-" * 75)

    fault_eval_rows = []
    for ft in sorted(summary_df["fault_type"].unique()):
        ft_df = summary_df[summary_df["fault_type"] == ft]
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
        log_and_print(
            f"  {ft:<12} {len(ft_df):>6} {row['svc_ac1']:>10.4f} {row['svc_ac3']:>10.4f} "
            f"{row['svc_ac5']:>10.4f} {row['mtr_ac1']:>10.4f} {row['mtr_ac5']:>10.4f}"
        )

    # 保存评估结果
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(join(args.output, f"{args.algo}_metrics.csv"), index=False)

    fault_eval_df = pd.DataFrame(fault_eval_rows)
    fault_eval_df.to_csv(join(args.output, f"{args.algo}_eval_by_fault.csv"), index=False)

    log_and_print(f"\n  汇总结果:   {summary_path}")
    log_and_print(f"  评估指标:   {join(args.output, f'{args.algo}_metrics.csv')}")
    log_and_print(f"  故障分组:   {join(args.output, f'{args.algo}_eval_by_fault.csv')}")
    log_and_print(f"  运行日志:   {log_path}")
    log_and_print(f"  错误日志:   {err_path}")
    log_and_print(f"\n  全部流程完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
