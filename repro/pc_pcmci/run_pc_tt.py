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

from common_eval import FAULT_TO_METRIC  # noqa: E402  (统一口径)
from common_features import select_features as _unified_select  # noqa: E402  (统一变量筛选)

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
# 【2026-09-25 统一口径】指标映射表已统一到 common_eval.FAULT_TO_METRIC
# （原先每个 runner 各写一份，disk/mem 的写法互不相同）

# TT 数据真相指标族映射 (按经验验证: 注入前后故障服务该族指标变化最剧烈):
#   cpu   → container-cpu-*        (usage/system/user 三列同族)
#   delay → istio-latency-90       (预处理后重命名为 {service}_latency, 精确单列)
#   loss  → istio-latency          (与 OB 一致; 经验上 p90 延迟变化 29x, 错误列常为零值被剔除)
# 【2026-09-25 统一口径】原 TT_FAULT_TO_FAMILY（cpu→container-cpu / disk→container-fs /
#   mem→container-memory 的「指标族前缀匹配」）已删除。
#   它让 TT 的指标级比其他数据集宽松（一族里任一列进 top-k 即算命中），
#   与 OB/SS 的「精确列名匹配」语义不同。现统一为官方精确匹配：
#   节点名必须精确等于 f"{service}_{common_eval.FAULT_TO_METRIC[fault]}"。
#   ⇒ RE1-TT 的 Mtr 因此大幅下降（旧 0.520 → 官方口径 0.088），这是**如实结果**：
#     长名节点（ts-auth-service_container-cpu-usage-seconds-total）本来就无法等于
#     ts-auth-service_cpu，官方命名规则在 TT 上失效（属官方既有局限，论文标 N/A）。


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
# 【2026-09-25 统一口径】评估函数改为共享实现 common_eval.py
#   服务级: 先 '-db' 归一、再对服务实体序列去重 (官方 main.py:380)
#   指标级: 节点名精确等于 f"{service}_{FAULT_TO_METRIC[fault]}" (官方 main.py:394)
#   变量筛选已改用 common_features（见上方 import），本地旧实现均已移除。
# ============================================================
from common_eval import compute_rank_accuracy as _official_rank_acc


def compute_rank_accuracy(ranks, service, fault_type, metric_names=None, top_k_list=(1, 3, 5)):
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
    parser.add_argument("--normal-only", action="store_true",
                        help="仅用故障前正常窗口建模（默认使用全窗口，与官方 main.py:213 一致）")
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

            # === 1.5 特征选择 ===
            # 【2026-09-25 修正｜P0-1 真值泄漏】
            # 原实现: truth_keep = get_truth_metrics(service, fault_type, ...)
            #         再用 keep_cols=truth_keep 强制保留「当前案例的真值指标列」，
            #         且 select_features 的去相关步骤对这些列豁免剔除。
            #         这等于人为保证真值节点出现在建图候选集里 —— 真值泄漏，
            #         会使 Mtr-AC 系统性虚高（TT 的 Mtr-AC@5=0.52 vs SS 的 0.10 即由此而来）。
            # 现改为【通用保留规则】: 所有指标的 latency 列一律保留。
            #         该规则与具体案例 / 真值无关，对 125 个案例一视同仁，
            #         且与 PCMCI 侧 select_features(preserve_patterns=["_latency"]) 对齐。
            # 【2026-09-25 统一口径】改用 common_features 的统一规则
            # （通用保留所有 *_latency 列，与案例真值无关；不再做案例级真值保留）
            data = _unified_select(
                data,
                max_features=args.max_features,
                drop_node_metrics=args.drop_node_metrics,
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
                use_full_window=not args.normal_only,
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
