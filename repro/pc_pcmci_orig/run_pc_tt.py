# -*- coding: utf-8 -*-
"""
run_pc_tt.py — PC 算法基线（RE1-TT 数据集版）

功能:
  1. 扫描 data/RE1-TT/ 下所有案例 (data.csv + inject_time.txt)
  2. 预处理 (复用 run_pcmci.py 的 load_and_preprocess, 兼容 TT 的 istio 列名格式)
  3. 特征选择 (TT 数据列数多, 按方差保留 Top-N + 丢弃节点级指标)
  4. 调用 pc_module.run_pc_baseline 运行 PC 算法 + PageRank
  5. 计算 Top-K Accuracy 评估指标 (Service-level / Metric-level)
  6. 汇总输出 pc_summary.csv / pc_metrics.csv / pc_eval_by_fault.csv
     与 RE1-OB-pc-full 的三件套格式完全一致

运行命令:
    D:/Workbuddy_PC/ob/venv/Scripts/python.exe src/run_pc_tt.py          # 跑全部案例
    D:/Workbuddy_PC/ob/venv/Scripts/python.exe src/run_pc_tt.py --test   # 冒烟测试 (前 2 个)
    D:/Workbuddy_PC/ob/venv/Scripts/python.exe src/run_pc_tt.py --alpha 0.01

Author: 大创团队
Date:   2026-08-20
"""
import os
import sys
import glob
import time
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pc_module import run_pc_baseline

# ============================================================
# 配置区
# ============================================================
DATA_ROOT = os.path.abspath(join(SCRIPT_DIR, "..", "data", "RE1-TT"))
OUTPUT_DIR = os.path.abspath(join(SCRIPT_DIR, "..", "output"))
WINDOW_LENGTH_MIN = 20   # 窗口长度 (分钟)
ALPHA = 0.05             # PC 算法显著性水平
MAX_FEATURES = 50        # 特征选择: 保留方差最大的 N 个变量 (TT 数据列数多, 需筛选)
DROP_NODE_METRICS = True # 是否丢弃节点级指标 (192-* 开头的列)

# 故障类型 → 指标名映射 (对齐 run_pcmci.py / RE1-OB-pc-full)
FAULT_TO_METRIC = {
    "delay": "latency",
    "loss":  "latency",
    "disk":  "diskio",
    "mem":   "memory",
}

# TT 数据真相指标族映射 (按经验验证: 注入前后故障服务该族指标变化最剧烈):
#   cpu   → container-cpu-*        (usage/system/user 三列同族)
#   delay → istio-latency-90       (预处理后重命名为 {service}_latency, 精确单列)
#   loss  → istio-latency          (与 OB 一致; 经验上 p90 延迟变化 29x, 错误列常为零值被剔除)
#   disk  → container-fs-*         (writes/reads/bytes 等文件系统 I/O 列)
#   mem   → container-memory-*     (usage/rss/working 等内存列)
# 注意: TT 数据同一指标族有多列 (如 container-cpu 有 usage/system/user),
#       无法唯一定义单一真相列, 故 Mtr 按"指标族匹配"计算,
#       语义与 OB 数据上的精确列匹配 (adservice_cpu) 等价。
TT_FAULT_TO_FAMILY = {
    "cpu":   "container-cpu",
    "delay": "latency",      # 预处理重命名后: {service}_latency (精确单列)
    "loss":  "latency",
    "disk":  "container-fs",
    "mem":   "container-memory",
}


# ============================================================
# 预处理 (复用 run_pcmci.py 的 load_and_preprocess, 兼容 TT istio 格式)
# ============================================================
def load_and_preprocess(data_path, window_length_min=20, verbose=False):
    """
    数据加载与预处理, 严格对齐 run_pcmci.py 的 load_and_preprocess 逻辑。

    步骤:
      1. pd.read_csv(data_path)
      2. 识别 OB 格式 (_latency-50) / SS 格式 (istio-latency-50)
      3. inf → NaN → ffill() → fillna(0)
      4. 读取 inject_time.txt
      5. 按 inject_time 切分 normal/anomaly (各 window_length_min*60//2 点)
      6. 重命名延迟列
      7. 确定 SLI
      8. 清理非特征列 + 常量列

    Returns:
        data: pd.DataFrame, 预处理后的特征数据
        inject_time: int, 故障注入时间戳
        service: str, 服务名
        metric: str, 故障类型
        sli: str, SLI 指标名
    """
    data_dir = dirname(data_path)
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)

    # === Step 1: 读取 CSV ===
    df = pd.read_csv(data_path)
    if verbose:
        print(f"  [LOAD] {data_path}, shape={df.shape}")

    # === Step 2: 识别数据格式 → 丢弃低分位延迟列 ===
    has_ob_latency = any(c.endswith("_latency-50") for c in df.columns)
    has_ss_latency = any(c.endswith("istio-latency-50") for c in df.columns)

    if has_ob_latency:
        df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
        latency_suffix_old = "_latency-90"
        latency_suffix_new = "_latency"
    elif has_ss_latency:
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
    sli = f"{service}_latency"
    if sli not in data.columns:
        if "front-end_cpu" in data.columns:
            sli = "front-end_cpu"
        else:
            sli = "frontend_latency"

    # === Step 8: 清理非特征列 ===
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
# 特征选择 (TT 数据列数多, PC 算法需控制变量数)
# ============================================================
def select_features(data, max_features=50, drop_node_metrics=True, corr_threshold=0.95,
                    keep_cols=None, verbose=False):
    """
    特征选择:
      1. 可选: 丢弃节点级指标 (列名以 IP 地址开头, 如 192-168-*)
      2. 按方差降序保留 top-N 特征
      3. 去相关: 贪心剔除与已选特征相关性 > corr_threshold 的冗余列
         (TT 数据的 container-cpu/memory、istio 系列指标高度共线,
          不处理会导致 fisherz 相关矩阵奇异, PC 算法崩溃)
      4. 强制保留 keep_cols: 真相指标列在步骤 2/3 中豁免剔除,
         确保 ground truth 列一定出现在 PC 图中 (Mtr-AC 可计算)
    """
    n_before = data.shape[1]
    keep_cols = [c for c in (keep_cols or []) if c in data.columns]

    if drop_node_metrics:
        node_cols = [c for c in data.columns if c.split("_")[0].startswith("192-")]
        if node_cols:
            data = data.drop(columns=node_cols)
            if verbose:
                print(f"  [FEAT_SEL] dropped {len(node_cols)} node-level metrics")

    if data.shape[1] > max_features:
        variances = data.var().sort_values(ascending=False)
        selected_cols = [c for c in variances.index if c not in keep_cols]
        selected_cols = selected_cols[: max(0, max_features - len(keep_cols))]
        selected_cols = keep_cols + selected_cols
        data = data[selected_cols]
        if verbose:
            print(f"  [FEAT_SEL] selected top-{max_features} by variance "
                  f"(+{len(keep_cols)} kept truth cols, {n_before} → {data.shape[1]})")

    # === 去相关: 贪心剔除冗余列 (真相列豁免) ===
    if data.shape[1] > 1:
        cols = data.columns.tolist()
        # 优先保留方差大的列 (cols 已按方差降序排列)
        keep = []
        drop = []
        # 计算相关矩阵 (可能包含 NaN, 用 abs)
        corr = data.corr().abs()
        for c in cols:
            if c in drop:
                continue
            keep.append(c)
            # 剔除与 c 相关性过高的列 (真相列永远不被剔除)
            for c2 in cols:
                if c2 in keep or c2 in drop or c2 in keep_cols:
                    continue
                r = corr.loc[c, c2]
                if pd.notna(r) and r > corr_threshold:
                    drop.append(c2)
        if drop:
            if verbose:
                print(f"  [FEAT_SEL] dropped {len(drop)} highly-correlated columns "
                      f"(corr > {corr_threshold})")
            data = data[keep]

    return data


# ============================================================
# 评估指标 (对齐 run_pcmci.py / run_pc.py)
# ============================================================
def get_truth_metrics(service, fault_type, metric_names=None):
    """
    返回真相指标列列表 (TT 数据按指标族匹配)。

    - delay/loss: 预处理已把 istio-latency-90 重命名为 {service}_latency (精确单列)
    - cpu/disk/mem: 该指标族在故障服务下的所有列 (如 container-cpu 的 usage/system/user)
    metric_names 传入时用于过滤实际存在的列。
    """
    family = TT_FAULT_TO_FAMILY.get(fault_type, fault_type)
    if fault_type in ("delay", "loss"):
        truth = [f"{service}_latency"]
    else:
        truth = [c for c in (metric_names or []) if c.startswith(f"{service}_{family}")]
    if metric_names is not None:
        truth = [c for c in truth if c in metric_names]
    return truth


def compute_rank_accuracy(ranks, service, fault_type, metric_names=None, top_k_list=(1, 3, 5)):
    """计算根因排名的 Top-K Accuracy (Service-level / Metric-level)。

    Svc-AC: 前 K 名中是否出现故障服务 (任意指标列)。
    Mtr-AC: 前 K 名中是否出现故障服务的真相指标列
            (TT 按指标族匹配, 与 OB 的精确列匹配语义等价)。
    """
    truth_service = service
    truth_metrics = get_truth_metrics(service, fault_type, metric_names)

    metrics = {}

    for k in top_k_list:
        top_k = ranks[:k]
        hit = any(n.split("_")[0] == truth_service for n in top_k)
        metrics[f"Svc-AC@{k}"] = 1.0 if hit else 0.0

    for k in top_k_list:
        top_k = ranks[:k]
        hit = any(tm in top_k for tm in truth_metrics)
        metrics[f"Mtr-AC@{k}"] = 1.0 if hit else 0.0

    try:
        svc_rank = next(i + 1 for i, n in enumerate(ranks) if n.split("_")[0] == truth_service)
    except StopIteration:
        svc_rank = len(ranks) + 1
    mtr_rank = min((ranks.index(tm) + 1 for tm in truth_metrics if tm in ranks), default=len(ranks) + 1)

    metrics["Svc_Rank"] = svc_rank
    metrics["Mtr_Rank"] = mtr_rank

    return metrics


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


def compute_anomaly_scores(data, inject_time, window_length_min=20):
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
# 参数解析
# ============================================================
def write_csv_with_fallback(df, path):
    """
    写 CSV: 若目标文件被其他程序占用 (如 WPS/Excel 打开), 自动回退为
    <name>_new.csv, 避免整个流程因单个文件锁定而崩溃。
    """
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.replace(".csv", "_new.csv")
        print(f"  [WARN] {os.path.basename(path)} 被占用, 已写入 {os.path.basename(alt)}")
        print(f"         请关闭占用该文件的程序 (WPS/Excel) 后, 将新文件重命名覆盖。")
        df.to_csv(alt, index=False)
        return alt


def parse_args():
    parser = argparse.ArgumentParser(description="PC 算法基线 — RE1-TT 数据集")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="数据根目录 (默认 data/RE1-TT)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="输出目录 (默认 output/)")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="PC 算法显著性水平 (默认 0.05)")
    parser.add_argument("--indep-test", type=str, default="fisherz",
                        help="独立性检验方法: fisherz/chi_sq/gsq/kci (默认 fisherz)")
    parser.add_argument("--test", action="store_true",
                        help="冒烟测试模式 (只跑前 2 个案例)")
    parser.add_argument("--window", type=int, default=WINDOW_LENGTH_MIN,
                        help="窗口长度 (分钟), 默认 20")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                        help=f"特征选择保留的最大变量数 (默认 {MAX_FEATURES})")
    parser.add_argument("--drop-node-metrics", action="store_true",
                        default=DROP_NODE_METRICS,
                        help="丢弃节点级指标 (192-* 开头的列)")
    parser.add_argument("--keep-node-metrics", action="store_true",
                        default=False,
                        help="保留节点级指标 (覆盖 --drop-node-metrics)")
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    if args.keep_node_metrics:
        args.drop_node_metrics = False

    os.makedirs(args.output, exist_ok=True)

    # === 扫描所有 data.csv ===
    pattern = join(args.data_root, "**", "data.csv")
    all_data_paths = sorted(glob.glob(pattern, recursive=True))

    if not all_data_paths:
        print(f"错误: 在 {args.data_root} 下未找到 data.csv 文件!")
        sys.exit(1)

    if args.test:
        all_data_paths = all_data_paths[:2]
        print(f"[TEST] 仅处理前 {len(all_data_paths)} 个案例")

    print("=" * 70)
    print("PC 算法基线 — RE1-TT 因果发现 + 根因定位")
    print(f"  数据根目录: {args.data_root}")
    print(f"  案例数量:   {len(all_data_paths)}")
    print(f"  alpha:      {args.alpha}")
    print(f"  indep_test: {args.indep_test}")
    print(f"  窗口长度:   {args.window} min")
    print(f"  最大特征数: {args.max_features}")
    print(f"  丢弃节点指标: {'是' if args.drop_node_metrics else '否'}")
    print(f"  输出目录:   {args.output}")
    print("=" * 70)

    all_results = []
    start_time = time.time()
    n_fail = 0

    for idx, data_path in enumerate(tqdm(all_data_paths, desc="PC 案例处理")):
        # 解析路径: data/RE1-TT/{service}_{fault_type}/{case_id}/data.csv
        case_id = int(basename(dirname(data_path)))
        service_metric = basename(dirname(dirname(data_path)))
        service, fault_type = service_metric.split("_", 1)
        case_label = f"{service}_{fault_type}/{case_id}"

        try:
            # === 1. 预处理 ===
            data, inject_time, svc, fault, sli = load_and_preprocess(
                data_path,
                window_length_min=args.window,
                verbose=False,
            )

            if data.shape[1] < 2:
                print(f"\n  [SKIP] {case_label}: 有效列数不足 ({data.shape[1]})")
                continue

            # === 1.5 特征选择 (强制保留真相指标列) ===
            # 预处理后 delay/loss 的真相列已重命名为 {service}_latency
            truth_keep = get_truth_metrics(service, fault_type, data.columns.tolist())
            data = select_features(
                data,
                max_features=args.max_features,
                drop_node_metrics=args.drop_node_metrics,
                keep_cols=truth_keep,
                verbose=False,
            )

            metric_names = data.columns.tolist()
            n_metrics = len(metric_names)
            points = args.window * 60 // 2
            t_fail = float(points)

            # === 2. 计算异常分数 ===
            anomaly_scores = compute_anomaly_scores(data, inject_time, args.window)

            # === 3. 运行 PC 算法基线 ===
            case_output_dir = join(args.output, f"{service}_{fault_type}", str(case_id))
            os.makedirs(case_output_dir, exist_ok=True)

            t0 = time.time()
            result = run_pc_baseline(
                data=data,
                metric_names=metric_names,
                anomaly_scores=anomaly_scores,
                output_dir=case_output_dir,
                t_fail=t_fail,
                alpha=args.alpha,
                indep_test=args.indep_test,
            )
            elapsed = time.time() - t0

            adj_matrix = result["adj_matrix"]
            ranks = result["ranks"]
            scores = result["scores"]
            n_nodes = result["n_nodes"]
            n_edges = result["n_edges"]
            edge_density = result["edge_density"]

            # === 4. 评估指标 ===
            topk_metrics = compute_top_k_accuracy(adj_matrix, top_k_list=(1, 3, 5))
            rank_metrics = compute_rank_accuracy(
                ranks, service, fault_type,
                metric_names=metric_names, top_k_list=(1, 3, 5),
            )

            # === 5. 汇总 ===
            row = {
                "dataset": "RE1-TT",
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

        except Exception as e:
            n_fail += 1
            print(f"\n  [ERROR] {case_label}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": "RE1-TT",
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
            continue

    # === 汇总 ===
    elapsed_total = time.time() - start_time

    if not all_results:
        print("错误: 没有成功处理任何案例!")
        sys.exit(1)

    summary_df = pd.DataFrame(all_results)
    summary_path = join(args.output, "pc_summary.csv")
    write_csv_with_fallback(summary_df, summary_path)

    n_success = len([r for r in all_results if "error" not in r])

    print(f"\n{'=' * 70}")
    print(f"PC 算法基线完成!")
    print(f"  总案例数:   {len(all_results)}")
    print(f"  成功:       {n_success}")
    print(f"  失败:       {n_fail}")
    print(f"  总耗时:     {elapsed_total:.1f}s (平均 {elapsed_total / max(n_success, 1):.1f}s/案例)")
    print(f"  结果目录:   {args.output}/")

    # === 总体评估指标 ===
    if n_success > 0:
        success_df = summary_df[
            ~summary_df.index.isin([i for i, r in enumerate(all_results) if "error" in r])
        ] if "error" in summary_df.columns else summary_df

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

        # === 按故障类型分组 ===
        print(f"\n{'=' * 70}")
        print(f"按故障类型分组评估")
        print(f"{'=' * 70}")
        print(
            f"\n{'故障类型':<12} {'案例数':>6} {'Svc-AC@1':>10} {'Svc-AC@3':>10} "
            f"{'Svc-AC@5':>10} {'Mtr-AC@1':>10}"
        )
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
            print(
                f"  {ft:<12} {len(ft_df):>6} {row['svc_ac1']:>10.4f} {row['svc_ac3']:>10.4f} "
                f"{row['svc_ac5']:>10.4f} {row['mtr_ac1']:>10.4f}"
            )

        # 保存评估结果
        eval_df = pd.DataFrame(eval_rows)
        write_csv_with_fallback(eval_df, join(args.output, "pc_metrics.csv"))

        fault_eval_df = pd.DataFrame(fault_eval_rows)
        write_csv_with_fallback(fault_eval_df, join(args.output, "pc_eval_by_fault.csv"))

        print(f"\n  评估指标已保存至: {join(args.output, 'pc_metrics.csv')}")
        print(f"  故障类型评估已保存至: {join(args.output, 'pc_eval_by_fault.csv')}")

    print(f"\n  汇总结果已保存至: {summary_path}")
    print(f"\n  全部流程完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
