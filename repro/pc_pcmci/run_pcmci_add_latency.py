# -*- coding: utf-8 -*-
"""
run_pcmci_add_latency.py — 方法二：PCMCI 原 50 特征 + latency 类特征 重跑

【背景】
  原 PCMCI 基线评测（RE1-TT / RE1-SS）经特征筛选后，50 个候选节点全部为
  container-memory-* 类指标，latency 指标被筛掉了。对 delay / loss 这两类故障，
  PCMCI 的候选边里先天缺失最关键的 latency 指标，导致 Mtr-AC 恒为 0。

【本脚本的修改（方法二）】
  在原 50 特征的基础上，加入每个服务的 istio-latency-90 列
  （重命名为 {service}_latency，与 OB 口径对齐），然后重跑 PCMCI。

  特征集规模：
    RE1-TT: 50 (原 mem 特征) + 29 (istio-latency-90) = 79 节点
    RE1-SS: 50 (原 mem 特征) +  8 (istio-latency-90) = 58 节点

【设计要点】
  * 原 50 特征直接取自旧输出 node_names.json，保证“原封不动”再加 latency；
  * 预处理口径与 run_pcmci.py 完全一致（inject_time 切分 / ffill / 丢常量列）；
  * PageRank 与评估函数与 run_pcmci.py 逐行一致，保证新旧可比；
  * 对 latency 指标引入白名单保护，避免其在故障前因波动过小被误删；
  * 支持断点续跑（默认跳过已完成的案例），支持多进程并行。

用法:
    python run_pcmci_add_latency.py --test              # 冒烟测试（每数据集 1 案例）
    python run_pcmci_add_latency.py                     # 全量重跑 TT + SS 的 delay/loss
    python run_pcmci_add_latency.py --datasets RE1-TT   # 只跑某数据集
    python run_pcmci_add_latency.py --workers 8         # 指定并行度
    python run_pcmci_add_latency.py --force             # 忽略已有结果全部重跑

Author: 大创团队
Date:   2026-09-14
"""
import os

# 并行时避免 BLAS 线程超订（必须在 import numpy 之前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import glob
import json
import time
import argparse
import warnings
import traceback
import multiprocessing as mp
from os.path import basename, dirname, join

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_eval import FAULT_TO_METRIC  # noqa: E402  (统一口径)

from pcmci_module import pcmci  # noqa: E402

# ============================================================
# 配置区
# ============================================================
ROOT = "D:/WorkBuddy_PCMCI_OB"

DATASETS = {
    "RE1-OB": {
        "data_root": "D:/WorkBuddy_PCMCI_OB/data/RE1-OB",
        "old_output": "D:/WorkBuddy_PCMCI_OB/output/RE1-OB",
    },
}

OUTPUT_ROOT = join(ROOT, "output", "pcmci_latency_fix")

TAU_MAX = 3
# 【2026-09-25 统一口径】官方 PCMCI 默认 alpha = 0.20 (见 graph_construction/pcmci.py)
ALPHA = 0.20
WINDOW_LENGTH_MIN = 20
FAULT_TYPES = ["cpu", "delay", "disk", "loss", "mem"]
N_WORKERS = 6

# RE1-OB 列名适配：
#   delay / loss 的 csv 列叫 {svc}_latency-90 / -50，需要重命名为 {svc}_latency（与旧基线对齐）
#   cpu / disk / mem 的 csv 列已叫 {svc}_latency，endswith("_latency-90") 不匹配 → no-op
LATENCY_SUFFIX = "_latency-90"
LATENCY_NEW_NAME = "_latency"

# 【2026-09-25 统一口径】指标映射表已统一到 common_eval.FAULT_TO_METRIC
# （原先每个 runner 各写一份，disk/mem 的写法互不相同）


# ============================================================
# 预处理（口径与 run_pcmci.py 一致）
# ============================================================
def load_case(data_path, window_length_min=WINDOW_LENGTH_MIN):
    data_dir = dirname(data_path)
    case_id = int(basename(data_dir))
    service, fault_type = basename(dirname(data_dir)).split("_", 1)

    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    with open(join(data_dir, "inject_time.txt"), "r") as f:
        inject_time = int(f.readlines()[0].strip())

    points = window_length_min * 60 // 2
    normal_df = df[df["time"] < inject_time].tail(points)
    anomal_df = df[df["time"] >= inject_time].head(points)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    drop_cols = [c for c in data.columns
                 if c.lower() in ("time", "time.1") or c.startswith("time")]
    data = data.drop(columns=drop_cols, errors="ignore")

    return data, inject_time, service, fault_type, case_id


def build_feature_set(data, base_nodes, verbose=False):
    """
    构造方法二特征集 = 原 50 特征 + 各服务 istio-latency-90（重命名 {svc}_latency）
    并对 latency 指标引入白名单保护，避免其在故障前因波动过小被常量过滤误删。
    """
    col_set = set(data.columns)

    # --- 原 50 特征（取交集，防止缺失） ---
    base = [c for c in base_nodes if c in col_set]
    missing = [c for c in base_nodes if c not in col_set]

    # --- latency 类特征映射 ---
    latency_pairs = []
    for c in data.columns:
        if c.endswith(LATENCY_SUFFIX):
            svc = c[: -len(LATENCY_SUFFIX)]
            new_name = f"{svc}{LATENCY_NEW_NAME}"
            if new_name in base or c in base:
                continue
            latency_pairs.append((c, new_name))

    feat_cols = base + [c for c, _ in latency_pairs]
    feat_df = data[feat_cols].copy()
    if latency_pairs:
        feat_df = feat_df.rename(columns=dict(latency_pairs))

    # ================== ✅ 关键修正：latency 白名单 ==================
    # latency 指标即使 std≈0，也绝不删除
    latency_cols = [c for c in feat_df.columns if c.endswith("_latency")]

    const_cols = [
        c for c in feat_df.columns
        if c not in latency_cols and feat_df[c].std() == 0
    ]
    # ===============================================================

    if const_cols:
        feat_df = feat_df.drop(columns=const_cols)

    info = {
        "n_base_requested": len(base_nodes),
        "n_base_used": len(base),
        "n_base_missing": len(missing),
        "n_latency_added": len(latency_pairs),
        "n_const_dropped": len(const_cols),
        "n_features": feat_df.shape[1],
    }
    if verbose:
        print(f"  [FEAT] {info}")
        if missing:
            print(f"  [FEAT] missing base cols ({len(missing)}): {missing[:5]}")
        if const_cols:
            print(f"  [FEAT] const dropped ({len(const_cols)}): {const_cols[:5]}")
    return feat_df, info


# ============================================================
# PageRank 排序（与 run_pcmci.py 逐行一致）
# ============================================================
def run_pagerank(adj_matrix, node_names):
    n = adj_matrix.shape[0]
    adj_float = np.array(adj_matrix, dtype=float)

    if adj_float.sum() == 0:
        return list(node_names), [0.0] * n

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj_float[i, j] != 0:
                G.add_edge(j, i)

    pr_scores = nx.pagerank(G, alpha=0.85)
    scored = [(node_names[i], pr_scores.get(i, 0.0)) for i in range(n)]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored_sorted], [x[1] for x in scored_sorted]
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


def visualize_causal_graph(adj_matrix, node_names, output_path, title=""):
    try:
        n = adj_matrix.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        adj_float = np.array(adj_matrix, dtype=float)
        for i in range(n):
            for j in range(n):
                if adj_float[i, j] != 0:
                    G.add_edge(j, i)

        plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(n, 1)))
        nx.draw(G, pos, with_labels=True,
                labels={i: node_names[i] for i in range(n)},
                node_size=500, font_size=6, arrows=True,
                edge_color="steelblue", alpha=0.7, node_color="lightblue")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"  [WARN] 可视化失败: {e}")
        return False


# ============================================================
# 单案例处理（worker）
# ============================================================
def process_case(task):
    ds_name, data_path, old_output, out_dir, make_graph = task
    t_start = time.time()
    try:
        data, inject_time, service, fault_type, case_id = load_case(data_path)

        old_names_path = join(old_output, f"{service}_{fault_type}", f"{case_id}_node_names.json")
        with open(old_names_path, "r", encoding="utf-8") as f:
            base_nodes = json.load(f)

        feat_df, feat_info = build_feature_set(data, base_nodes, verbose=False)
        if feat_df.shape[1] < 2:
            raise RuntimeError(f"有效特征数不足: {feat_df.shape[1]}")

        t0 = time.time()
        adj = pcmci(feat_df, tau_max=TAU_MAX, alpha=ALPHA)
        elapsed = time.time() - t0

        node_names = feat_df.columns.to_list()
        n_nodes = len(node_names)
        adj_arr = np.array(adj)
        n_edges = int(adj_arr.sum())

        case_dir = join(out_dir, ds_name, f"{service}_{fault_type}")
        os.makedirs(case_dir, exist_ok=True)
        np.save(join(case_dir, f"{case_id}_adj_matrix.npy"), adj_arr)
        with open(join(case_dir, f"{case_id}_node_names.json"), "w", encoding="utf-8") as f:
            json.dump(node_names, f, ensure_ascii=False)

        ranks, scores = run_pagerank(adj_arr, node_names)
        pd.DataFrame({
            "rank": range(1, len(ranks) + 1),
            "node": ranks,
            "pagerank_score": scores,
        }).to_csv(join(case_dir, f"{case_id}_ranks.csv"), index=False)

        rank_metrics = compute_rank_accuracy(ranks, service, fault_type)

        if make_graph:
            visualize_causal_graph(
                adj_arr, node_names,
                join(case_dir, f"{case_id}_causal_graph.png"),
                title=f"PCMCI(+latency) - {service}_{fault_type}/#{case_id}",
            )

        return {
            "dataset": ds_name,
            "service": service,
            "fault_type": fault_type,
            "case_id": case_id,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "edge_density": round(n_edges / max(n_nodes * (n_nodes - 1), 1), 4),
            "pcmci_sec": round(elapsed, 2),
            "total_sec": round(time.time() - t_start, 2),
            "top1_node": ranks[0] if ranks else "",
            "top3_nodes": "|".join(ranks[:3]),
            "ground_truth_metric": f"{service}_{FAULT_TO_METRIC.get(fault_type, fault_type)}",
            **{f"feat_{k}": v for k, v in feat_info.items()},
            **rank_metrics,
        }

    except Exception as e:
        return {
            "dataset": ds_name,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-800:],
            "total_sec": round(time.time() - t_start, 2),
        }


# ============================================================
# 旧结果基线
# ============================================================
def compute_old_baseline(ds_name, old_output):
    rows = []
    for ranks_path in sorted(glob.glob(join(old_output, "*", "*_ranks.csv"))):
        case_dir = dirname(ranks_path)
        service_fault = basename(case_dir)
        if "_" not in service_fault:
            continue
        service, fault_type = service_fault.split("_", 1)
        if fault_type not in FAULT_TYPES:
            continue
        case_id = int(basename(ranks_path).split("_")[0])
        try:
            rdf = pd.read_csv(ranks_path)
            ranks = rdf.sort_values("rank")["node"].tolist()
        except Exception:
            continue
        m = compute_rank_accuracy(ranks, service, fault_type)
        rows.append({
            "dataset": ds_name, "service": service, "fault_type": fault_type,
            "case_id": case_id, "n_nodes": len(ranks),
            "variant": "old_50mem", **m,
        })
    return pd.DataFrame(rows)


# ============================================================
# 主流程
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="PCMCI + latency 特征重跑（方法二）")
    p.add_argument("--datasets", type=str, default=",".join(DATASETS.keys()),
                   help="逗号分隔的数据集名，默认 RE1-TT,RE1-SS")
    p.add_argument("--workers", type=int, default=N_WORKERS, help="并行进程数")
    p.add_argument("--tau_max", type=int, default=TAU_MAX)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--test", action="store_true", help="冒烟测试：每数据集只跑 1 个案例")
    p.add_argument("--limit", type=int, default=0, help="每数据集最多处理 N 个案例（0=全部）")
    p.add_argument("--force", action="store_true", help="忽略已有结果，强制重跑")
    p.add_argument("--no-graph", action="store_true", help="不做因果图可视化")
    return p.parse_args()


def main():
    global TAU_MAX, ALPHA
    args = parse_args()
    TAU_MAX = args.tau_max
    ALPHA = args.alpha

    ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in ds_names:
        if d not in DATASETS:
            print(f"错误：未知数据集 {d}（可选: {list(DATASETS)}）")
            sys.exit(1)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 72)
    print("PCMCI + latency 特征重跑（方法二 · 白名单修正版 · RE1-OB 全 125 案例）")
    print(f"  特征集:      旧基线 node_names.json 原节点 + 各服务 latency（delay/loss: *_latency-90 → {{svc}}_latency；cpu/disk/mem: 原生 *_latency）")
    print(f"  故障类型:    {FAULT_TYPES}")
    print(f"  tau_max:     {TAU_MAX}   alpha: {ALPHA}")
    print(f"  窗口长度:    {WINDOW_LENGTH_MIN} min")
    print(f"  并行度:      {args.workers}")
    print(f"  输出目录:    {OUTPUT_ROOT}")
    print("=" * 72)

    grand_start = time.time()
    all_new = []
    all_old = []

    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        data_root, old_output = cfg["data_root"], cfg["old_output"]

        data_paths = sorted(glob.glob(join(data_root, "**", "data.csv"), recursive=True))
        data_paths = [p for p in data_paths
                      if basename(dirname(dirname(p))).split("_", 1)[-1] in FAULT_TYPES]
        if not data_paths:
            print(f"\n[WARN] {ds_name}: 未找到符合 {FAULT_TYPES} 的 data.csv，跳过")
            continue

        if args.test:
            data_paths = data_paths[:1]
        elif args.limit:
            data_paths = data_paths[:args.limit]

        out_dir = OUTPUT_ROOT
        pending = []
        for dp in data_paths:
            svc_fault = basename(dirname(dirname(dp)))
            cid = basename(dirname(dp))
            if not args.force and os.path.exists(join(out_dir, ds_name, svc_fault, f"{cid}_ranks.csv")):
                continue
            make_graph = (len(pending) < 1) and not args.no_graph
            pending.append((ds_name, dp, old_output, out_dir, make_graph))

        print(f"\n{'=' * 72}")
        print(f"[{ds_name}] 案例总数: {len(data_paths)} | 待处理: {len(pending)} | 已完成: {len(data_paths)-len(pending)}")
        print(f"{'=' * 72}")

        results = []
        if pending:
            if args.workers <= 1:
                for t in tqdm(pending, desc=f"{ds_name}", mininterval=5, miniters=1):
                    results.append(process_case(t))
            else:
                with mp.Pool(processes=min(args.workers, len(pending))) as pool:
                    for r in tqdm(pool.imap_unordered(process_case, pending), total=len(pending), desc=ds_name, mininterval=5, miniters=1):
                        results.append(r)
        all_new.extend(results)

        # ---- 汇总本次 + 全部已落盘 ----
        rows = []
        for dp in data_paths:
            svc_fault = basename(dirname(dirname(dp)))
            cid = basename(dirname(dp))
            service, fault_type = svc_fault.split("_", 1)
            rk = join(out_dir, ds_name, svc_fault, f"{cid}_ranks.csv")
            if not os.path.exists(rk):
                continue
            ranks = pd.read_csv(rk).sort_values("rank")["node"].tolist()
            m = compute_rank_accuracy(ranks, service, fault_type)
            rows.append({
                "dataset": ds_name, "service": service, "fault_type": fault_type,
                "case_id": int(cid), "n_nodes": len(ranks), "variant": "new_add_latency", **m,
            })
        ds_new = pd.DataFrame(rows)
        ds_new.to_csv(join(OUTPUT_ROOT, f"summary_latency_fix_{ds_name}.csv"), index=False)

        ds_old = compute_old_baseline(ds_name, old_output)
        all_old.append(ds_old)

        print(f"\n[{ds_name}] 新旧对比（n={len(ds_new)}，覆盖全部 5 种故障）")
        hdr = f"{'指标':<14}{'旧(原基线)':>14}{'新(+latency)':>14}{'Δ':>10}"
        print(hdr)
        print("-" * len(hdr))
        cmp_rows = []
        for met in ["Svc-AC@1", "Svc-AC@3", "Svc-AC@5", "Mtr-AC@1", "Mtr-AC@3", "Mtr-AC@5"]:
            o = ds_old[met].mean() if len(ds_old) else float("nan")
            n = ds_new[met].mean() if len(ds_new) else float("nan")
            print(f"{met:<14}{o:>12.4f}{n:>14.4f}{n-o:>+10.4f}")
            cmp_rows.append({"dataset": ds_name, "metric": met, "old_50mem": round(o, 4),
                             "new_add_latency": round(n, 4), "delta": round(n - o, 4)})
        for met in ["Svc_Rank", "Mtr_Rank"]:
            o = ds_old[met].mean() if len(ds_old) else float("nan")
            n = ds_new[met].mean() if len(ds_new) else float("nan")
            print(f"{met:<14}{o:>12.2f}{n:>14.2f}{n-o:>+10.2f}")
            cmp_rows.append({"dataset": ds_name, "metric": met, "old_50mem": round(o, 2),
                             "new_add_latency": round(n, 2), "delta": round(n - o, 2)})

        print(f"{'Avg n_nodes':<14}{ds_old['n_nodes'].mean():>12.1f}{ds_new['n_nodes'].mean():>14.1f}"
              f"{ds_new['n_nodes'].mean()-ds_old['n_nodes'].mean():>+10.1f}")

        pd.DataFrame(cmp_rows).to_csv(join(OUTPUT_ROOT, f"compare_{ds_name}.csv"), index=False)

        per_svc = ds_new.groupby("service").agg(
            n=("case_id", "count"),
            Svc_AC1=("Svc-AC@1", "mean"), Svc_AC3=("Svc-AC@3", "mean"), Svc_AC5=("Svc-AC@5", "mean"),
            Mtr_AC1=("Mtr-AC@1", "mean"), Mtr_AC3=("Mtr-AC@3", "mean"), Mtr_AC5=("Mtr-AC@5", "mean"),
        ).round(3).reset_index()
        per_svc.to_csv(join(OUTPUT_ROOT, f"per_service_{ds_name}.csv"), index=False)

    total_elapsed = time.time() - grand_start

    new_all = pd.DataFrame(all_new)
    if len(new_all):
        has_err_col = "error" in new_all.columns
        err = new_all[new_all["error"].notna()] if has_err_col else pd.DataFrame()
        ok = new_all[new_all["error"].isna()] if has_err_col else new_all
        print(f"\n{'=' * 72}")
        print(f"本次运行：处理 {len(new_all)} 个案例，成功 {len(ok)}，失败 {len(err)}")
        if len(err):
            print("失败案例：")
            for _, r in err.iterrows():
                print("   -", r.get("dataset"), r.get("error"))
        print(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    all_rows = []
    for ds_name in ds_names:
        f = join(OUTPUT_ROOT, f"summary_latency_fix_{ds_name}.csv")
        if os.path.exists(f):
            all_rows.append(pd.read_csv(f))
    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(join(OUTPUT_ROOT, "summary_latency_fix_ALL.csv"), index=False)

        # 自适应表头：每个数据集一列 + ALL
        col_w = 14
        ds_labels = [d for d in ds_names if (full["dataset"] == d).any()]
        hdr = f"{'指标':<16}" + "".join(f"{d:>{col_w}}" for d in ds_labels) + f"{'ALL':>{col_w}}"
        sep = "-" * (16 + col_w * (len(ds_labels) + 1))
        print(f"\n{'=' * 72}")
        print(f"最终结果（方法二: 原节点 + latency · 白名单修正）")
        print(f"{'=' * 72}")
        print(hdr)
        print(sep)
        for met in ["Svc-AC@1", "Svc-AC@3", "Svc-AC@5",
                    "Mtr-AC@1", "Mtr-AC@3", "Mtr-AC@5", "Svc_Rank", "Mtr_Rank"]:
            vals = [full[full["dataset"] == d][met].mean() if (full["dataset"] == d).any() else float("nan")
                    for d in ds_labels]
            al = full[met].mean()
            row = f"{met:<16}" + "".join(f"{v:>{col_w}.4f}" for v in vals) + f"{al:>{col_w}.4f}"
            print(row)

        # 按故障类型拆分（不同故障根因不同）
        print(f"\n--- 按故障类型细分（全 {len(full)} 案例） ---")
        if "fault_type" in full.columns:
            ft_hdr = f"{'故障':<10}{'n':>6}" + "".join(f"{m:>{col_w}}" for m in ["Svc-AC@1", "Svc-AC@3", "Svc-AC@5", "Mtr-AC@5", "Svc_Rank", "Mtr_Rank"])
            print(ft_hdr)
            print("-" * len(ft_hdr))
            for ft in FAULT_TYPES:
                sub = full[full["fault_type"] == ft]
                if len(sub) == 0:
                    continue
                row = (f"{ft:<10}{len(sub):>6}"
                       + f"{sub['Svc-AC@1'].mean():>{col_w}.4f}"
                       + f"{sub['Svc-AC@3'].mean():>{col_w}.4f}"
                       + f"{sub['Svc-AC@5'].mean():>{col_w}.4f}"
                       + f"{sub['Mtr-AC@5'].mean():>{col_w}.4f}"
                       + f"{sub['Svc_Rank'].mean():>{col_w}.2f}"
                       + f"{sub['Mtr_Rank'].mean():>{col_w}.2f}")
                print(row)

    cmp_files = [join(OUTPUT_ROOT, f"compare_{d}.csv") for d in ds_names]
    cmp_files = [f for f in cmp_files if os.path.exists(f)]
    if cmp_files:
        pd.concat([pd.read_csv(f) for f in cmp_files], ignore_index=True).to_csv(
            join(OUTPUT_ROOT, "compare_ALL.csv"), index=False)

    print(f"\n输出目录: {OUTPUT_ROOT}")
    print("完成。")


if __name__ == "__main__":
    main()