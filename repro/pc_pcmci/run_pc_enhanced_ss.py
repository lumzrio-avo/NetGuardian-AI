# -*- coding: utf-8 -*-
"""
run_pc_enhanced.py — Enhanced PC 算法基线入口脚本 (RE1-SS)

改进点:
  1. 特征预筛选: 231 → ~40 变量 (按 anomaly score + 去高相关)
  2. 全窗口数据: normal+anomaly 共 721 行 (vs 仅 normal 288 行)
  3. 放宽 alpha: 0.05 → 0.1

运行命令:
    D:/Workbuddy_PC/ob/venv/Scripts/python.exe src/run_pc_enhanced.py

Author: 大创团队
Date:   2026-08-24
"""

import os
import sys
import glob
import time
import argparse
import warnings
import traceback
from os.path import basename, dirname, join

# Windows 控制台 UTF-8
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
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from common_eval import FAULT_TO_METRIC  # noqa: E402  (统一口径)

# 导入 Enhanced PC 模块
try:
    from pc_module_enhanced import run_pc_baseline_enhanced
except ImportError as e:
    print(f"错误: 无法导入 pc_module_enhanced! 请确保 src/pc_module_enhanced.py 存在。")
    print(f"  详细信息: {e}")
    sys.exit(1)

# ============================================================
# 配置区
# ============================================================
DATA_ROOT = join(PROJECT_DIR, "data", "RE1-SS")
OUTPUT_DIR = join(PROJECT_DIR, "output", "RE1-SS-pc-enhanced")
WINDOW_LENGTH_MIN = 20
# 【2026-09-25 统一口径】改回官方 PC 默认 alpha = 0.05。
#   官方依据: RCAEval/graph_construction/pc.py 的 pc_fisherz / pc_fisherz_stable /
#             pc_gsq / pc_chisq 全部显式写 alpha=0.05;
#             e2e/pc_pagerank.py 直接调 causallearn pc()(其默认 alpha 亦为 0.05);
#             legacy/graph_eval.py 的 -a/--alpha 默认值也是 0.05。
#   原值 0.1 是 RE1-SS 专用的放宽，导致 PC 三系统口径不一致。
ALPHA = 0.05
MAX_FEATURES = 40        # ★ 预筛选最大变量数
CORR_THRESHOLD = 0.95   # ★ 去冗余相关系数阈值
# 【2026-09-25 统一口径】默认全窗口；实际取值为 `use_full_window = not args.normal_only`
# （原先这里的常量 USE_FULL_WINDOW 其实没被使用，已移除以免误导）

# 故障类型 -> 指标名映射
# 【2026-09-25 统一口径】指标映射表已统一到 common_eval.FAULT_TO_METRIC
# （原先每个 runner 各写一份，disk/mem 的写法互不相同）


# ============================================================
# 预处理函数 (复用 run_pc.py 的逻辑)
# ============================================================
def load_and_preprocess(data_path, window_length_min=20, verbose=False):
    """数据加载与预处理, 适配 Sock Shop (SS) 数据格式。"""
    data_dir = dirname(data_path)
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)

    # Step 1: 读取 CSV
    df = pd.read_csv(data_path)
    if verbose:
        print(f"  [LOAD] {data_path}, shape={df.shape}")

    # Step 2: 丢弃低分位延迟列, 保留 -90
    latency_50_suffix = "istio-latency-50"
    latency_90_suffix = "istio-latency-90"
    latency_95_suffix = "istio-latency-95"
    latency_99_suffix = "istio-latency-99"

    has_ss_latency = any(c.endswith(latency_50_suffix) for c in df.columns)
    if has_ss_latency:
        for suffix in [latency_50_suffix, latency_95_suffix, latency_99_suffix]:
            df = df.loc[:, ~df.columns.str.endswith(suffix)]
        rename_map = {}
        for c in df.columns:
            if c.endswith(latency_90_suffix):
                new_name = c.replace(latency_90_suffix, "latency")
                rename_map[c] = new_name
        if rename_map:
            df = df.rename(columns=rename_map)

    # Step 3: 丢弃低分位 bytes 列, 保留 -90
    bytes_50_suffix = "istio-bytes-50"
    bytes_90_suffix = "istio-bytes-90"
    bytes_95_suffix = "istio-bytes-95"
    bytes_99_suffix = "istio-bytes-99"

    has_ss_bytes = any(c.endswith(bytes_50_suffix) for c in df.columns)
    if has_ss_bytes:
        for suffix in [bytes_50_suffix, bytes_95_suffix, bytes_99_suffix]:
            df = df.loc[:, ~df.columns.str.endswith(suffix)]
        rename_map = {}
        for c in df.columns:
            if c.endswith(bytes_90_suffix):
                new_name = c.replace(bytes_90_suffix, "bytes")
                rename_map[c] = new_name
        if rename_map:
            df = df.rename(columns=rename_map)

    # Step 4: 无穷值 + 缺失值处理
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    df = df.fillna(0)

    # Step 5: 读取故障注入时间
    inject_path = join(data_dir, "inject_time.txt")
    with open(inject_path, "r") as f:
        inject_time = int(f.readlines()[0].strip())

    # Step 6: 按 inject_time 切分
    points = window_length_min * 60 // 2  # 20 min -> 600 points each
    normal_df = df[df["time"] < inject_time].tail(points)
    anomal_df = df[df["time"] >= inject_time].head(points)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

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

    # 丢弃常量列
    const_cols = [c for c in data.columns if data[c].std() == 0]
    if const_cols:
        data = data.drop(columns=const_cols)
        if verbose:
            print(f"  [DROP_CONST] dropped {len(const_cols)} constant columns")

    return data, inject_time, service, metric, sli
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
# 计算异常分数
# ============================================================
def compute_anomaly_scores(data, inject_time, window_length_min=20):
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
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced PC 算法基线 — RE1-SS")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"显著性水平 (默认 {ALPHA})")
    parser.add_argument("--indep-test", type=str, default="fisherz")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                        help=f"预筛选最大变量数 (默认 {MAX_FEATURES})")
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD,
                        help=f"去冗余相关系数阈值 (默认 {CORR_THRESHOLD})")
    parser.add_argument("--normal-only", action="store_true",
                        help="仅使用正常窗口数据 (默认使用全窗口)")
    parser.add_argument("--test", action="store_true",
                        help="冒烟测试模式 (只跑前 2 个案例)")
    parser.add_argument("--window", type=int, default=WINDOW_LENGTH_MIN)
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 顶层日志
    top_log_path = join(args.output, "run_log_pc.txt")
    top_err_path = join(args.output, "run_err_pc.txt")
    open(top_log_path, "w").close()
    open(top_err_path, "w").close()

    def log_and_print(msg, level="INFO"):
        print(msg, flush=True)
        with open(top_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")

    def log_error(msg):
        print(msg, flush=True)
        with open(top_err_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {msg}\n")
        with open(top_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {msg}\n")

    # === 扫描所有 data.csv ===
    pattern = join(args.data_root, "**", "data.csv")
    all_data_paths = sorted(glob.glob(pattern, recursive=True))

    if not all_data_paths:
        log_error(f"错误: 在 {args.data_root} 下未找到 data.csv 文件!")
        sys.exit(1)

    if args.test:
        all_data_paths = all_data_paths[:2]
        log_and_print(f"[TEST] 仅处理前 {len(all_data_paths)} 个案例")

    use_full_window = not args.normal_only

    log_and_print("=" * 70)
    log_and_print("Enhanced PC 算法基线 — RE1-SS 因果发现 + 根因定位")
    log_and_print(f"  数据根目录:    {args.data_root}")
    log_and_print(f"  案例数量:      {len(all_data_paths)}")
    log_and_print(f"  alpha:         {args.alpha}")
    log_and_print(f"  indep_test:    {args.indep_test}")
    log_and_print(f"  max_features:  {args.max_features}")
    log_and_print(f"  corr_threshold: {args.corr_threshold}")
    log_and_print(f"  use_full_window: {use_full_window}")
    log_and_print(f"  窗口长度:      {args.window} min")
    log_and_print(f"  输出目录:      {args.output}")
    log_and_print("=" * 70)

    all_results = []
    start_time = time.time()

    for idx, data_path in enumerate(all_data_paths):
        case_id = int(basename(dirname(data_path)))
        service_metric = basename(dirname(dirname(data_path)))
        service, fault_type = service_metric.split("_", 1)

        case_label = f"{service}_{fault_type}/{case_id}"
        pct = idx * 100 // len(all_data_paths)
        log_and_print(f"\n[{idx + 1}/{len(all_data_paths)}] ({pct}%) 处理 {case_label} ...")

        try:
            # === 1. 预处理 ===
            data, inject_time, svc, fault, sli = load_and_preprocess(
                data_path,
                window_length_min=args.window,
                verbose=False,
            )

            if data.shape[1] < 2:
                log_and_print(f"  [SKIP] {case_label}: 有效列数不足 ({data.shape[1]})")
                continue

            metric_names = data.columns.tolist()
            points = args.window * 60 // 2
            t_fail = float(points)

            # === 2. 计算异常分数 ===
            anomaly_scores = compute_anomaly_scores(data, inject_time, args.window)

            # === 3. 运行 Enhanced PC 算法 ===
            case_output_dir = join(args.output, f"{service}_{fault_type}", str(case_id))

            t0 = time.time()
            result = run_pc_baseline_enhanced(
                data=data,
                metric_names=metric_names,
                anomaly_scores=anomaly_scores,
                output_dir=case_output_dir,
                t_fail=t_fail,
                alpha=args.alpha,
                indep_test=args.indep_test,
                max_features=args.max_features,
                corr_threshold=args.corr_threshold,
                use_full_window=use_full_window,
            )
            elapsed = time.time() - t0

            adj_matrix = result["adj_matrix"]
            ranks = result["ranks"]
            scores = result["scores"]
            n_nodes = result["n_nodes"]
            n_edges = result["n_edges"]
            edge_density = result["edge_density"]
            n_original = result.get("n_original", n_nodes)
            n_selected = result.get("n_selected", n_nodes)

            log_and_print(
                f"  [OK] orig_vars={n_original} → selected={n_selected}, "
                f"edges={n_edges}, density={edge_density:.4f}, "
                f"elapsed={elapsed:.2f}s"
            )
            log_and_print(f"  [RANK] Top-5: {ranks[:5]}")

            # === 4. 评估指标 ===
            topk_metrics = compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5))
            rank_metrics = compute_rank_accuracy(ranks, service, fault_type, top_k_list=(1, 3, 5))

            # === 5. 汇总 ===
            row = {
                "dataset": "RE1-SS",
                "method": "PC-Enhanced",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "n_original_vars": n_original,
                "n_selected_vars": n_selected,
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

        except Exception as e:
            err_msg = f"  [ERROR] {case_label}: {e}"
            log_error(err_msg)
            traceback.print_exc()
            all_results.append({
                "dataset": "RE1-SS",
                "method": "PC-Enhanced",
                "service": service,
                "fault_type": fault_type,
                "case_id": case_id,
                "n_original_vars": 0,
                "n_selected_vars": 0,
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
            continue

    # === 汇总 ===
    elapsed_total = time.time() - start_time

    if not all_results:
        log_error("错误: 没有成功处理任何案例!")
        sys.exit(1)

    summary_df = pd.DataFrame(all_results)
    summary_path = join(args.output, "pc_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    n_success = len([r for r in all_results if "error" not in r])
    n_fail = len(all_results) - n_success

    log_and_print(f"\n{'=' * 70}")
    log_and_print(f"Enhanced PC 算法基线完成!")
    log_and_print(f"  总案例数:   {len(all_results)}")
    log_and_print(f"  成功:       {n_success}")
    log_and_print(f"  失败:       {n_fail}")
    log_and_print(f"  总耗时:     {elapsed_total:.1f}s (平均 {elapsed_total / max(n_success, 1):.1f}s/案例)")
    log_and_print(f"  结果目录:   {args.output}/")

    # === 总体评估指标 ===
    if n_success > 0:
        success_df = summary_df[
            ~summary_df.index.isin([i for i, r in enumerate(all_results) if "error" in r])
        ] if "error" in summary_df.columns else summary_df

        log_and_print(f"\n{'=' * 70}")
        log_and_print(f"评估指标 (n={len(success_df)} 成功案例)")
        log_and_print(f"{'=' * 70}")
        log_and_print(f"\n{'指标':<25} {'值':>10}")
        log_and_print("-" * 40)

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
            "Avg n_original_vars": success_df["n_original_vars"].mean(),
            "Avg n_selected_vars": success_df["n_selected_vars"].mean(),
            "Avg n_nodes": success_df["n_nodes"].mean(),
            "Avg n_edges": success_df["n_edges"].mean(),
            "Avg edge_density": success_df["edge_density"].mean(),
        }

        eval_rows = []
        for k, v in eval_metrics.items():
            log_and_print(f"  {k:<25} {v:>10.4f}")
            eval_rows.append({"metric": k, "value": round(v, 4)})

        # === 按故障类型分组 ===
        log_and_print(f"\n{'=' * 70}")
        log_and_print(f"按故障类型分组评估")
        log_and_print(f"{'=' * 70}")
        log_and_print(
            f"\n{'故障类型':<12} {'案例数':>6} {'边数(均)':>10} {'密度(均)':>10} "
            f"{'Svc-AC@1':>10} {'Svc-AC@3':>10} {'Svc-AC@5':>10} {'Mtr-AC@1':>10}"
        )
        log_and_print("-" * 80)

        fault_eval_rows = []
        for ft in sorted(success_df["fault_type"].unique()):
            ft_df = success_df[success_df["fault_type"] == ft]
            row = {
                "fault_type": ft,
                "n_cases": len(ft_df),
                "avg_n_edges": ft_df["n_edges"].mean(),
                "avg_density": ft_df["edge_density"].mean(),
                "svc_ac1": ft_df["Svc-AC@1"].mean(),
                "svc_ac3": ft_df["Svc-AC@3"].mean(),
                "svc_ac5": ft_df["Svc-AC@5"].mean(),
                "mtr_ac1": ft_df["Mtr-AC@1"].mean(),
                "mtr_ac3": ft_df["Mtr-AC@3"].mean(),
                "mtr_ac5": ft_df["Mtr-AC@5"].mean(),
            }
            fault_eval_rows.append(row)
            log_and_print(
                f"  {ft:<12} {len(ft_df):>6} {row['avg_n_edges']:>10.1f} "
                f"{row['avg_density']:>10.4f} "
                f"{row['svc_ac1']:>10.4f} {row['svc_ac3']:>10.4f} "
                f"{row['svc_ac5']:>10.4f} {row['mtr_ac1']:>10.4f}"
            )

        # 保存评估结果
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(join(args.output, "pc_metrics.csv"), index=False)

        fault_eval_df = pd.DataFrame(fault_eval_rows)
        fault_eval_df.to_csv(join(args.output, "pc_eval_by_fault.csv"), index=False)

        log_and_print(f"\n  评估指标已保存至: {join(args.output, 'pc_metrics.csv')}")
        log_and_print(f"  故障类型评估已保存至: {join(args.output, 'pc_eval_by_fault.csv')}")

    log_and_print(f"\n  汇总结果已保存至: {summary_path}")
    log_and_print(f"\n{'=' * 70}")
    log_and_print(f"  全部流程完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
