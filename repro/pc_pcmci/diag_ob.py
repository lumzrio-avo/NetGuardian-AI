# -*- coding: utf-8 -*-
"""diag_ob.py — 诊断 OB 上 PC 输出空图（0 条边）的原因

只读脚本：读数据 → 跑与 runner 完全相同的预处理与变量筛选 → 检查相关矩阵
→ 实际调用 causallearn 的 pc()，报告边数或异常。**不写任何结果目录**。

用法（WSL）:
    ~/RCAEval/env/bin/python diag_ob.py /home/lumzrio/RCAEval/data/RE1/RE1-OB [案例数=3]
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_rca_full_ob import load_and_preprocess          # noqa: E402
from common_features import select_features              # noqa: E402


def show_col(name, s):
    return (f"{name}: std={s.std():.3e} min={s.min():.3e} "
            f"max={s.max():.3e} nunique={s.nunique()}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    paths = sorted(glob.glob(os.path.join(root, "*_*", "*", "data.csv")))
    if not paths:
        print("!! 没找到 {service}_{fault}/{case}/data.csv，检查路径:", root)
        return 1
    print(f"共 {len(paths)} 个案例，抽前 {n_cases} 个诊断")
    print(f"数据根目录: {root}\n")

    for p in paths[:n_cases]:
        svc_fault = os.path.basename(os.path.dirname(os.path.dirname(p)))
        case = os.path.basename(os.path.dirname(p))
        print("=" * 80)
        print(f"■ 案例 {svc_fault}/{case}")
        print("=" * 80)

        try:
            data, inject_time, svc, fault, sli = load_and_preprocess(p, verbose=False)
        except Exception as e:
            print(f"  [ERR] load_and_preprocess 失败: {type(e).__name__}: {e}\n")
            continue

        print(f"  load_and_preprocess → shape={data.shape}  (service={svc}, fault={fault})")

        normal_end = 480
        const_full = [c for c in data.columns if data[c].std() == 0]
        normal_part = data.iloc[:normal_end]
        const_normal = [c for c in data.columns if normal_part[c].std() == 0]
        newly = sorted(set(const_normal) - set(const_full))
        print(f"  常量列(全窗口)={len(const_full)}   常量列(仅正常段前480)={len(const_normal)}")
        print(f"  ★ 只因改成全窗口才被保留的列 ({len(newly)}): {newly}")

        try:
            sel = select_features(data, max_features=50, drop_node_metrics=True, verbose=False)
        except Exception as e:
            print(f"  [ERR] select_features 失败: {type(e).__name__}: {e}\n")
            continue
        print(f"  select_features → shape={sel.shape}")
        keep = [c for c in sel.columns if "_latency" in c]
        print(f"    其中强制保留的 latency 列 {len(keep)} 个: {keep}")

        X = sel.to_numpy(dtype=float)
        n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
        if n_bad:
            print(f"  [WARN] 含 NaN/Inf 共 {n_bad} 个")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 相关矩阵诊断 ---
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=1.0)
        eig = np.linalg.eigvalsh(C)
        rk = int(np.linalg.matrix_rank(C))
        cond = eig.max() / max(eig.min(), 1e-300)
        print(f"  相关矩阵: shape={C.shape} rank={rk} "
              f"min_eig={eig.min():.3e} max_eig={eig.max():.3e} cond={cond:.3e}")
        if rk < C.shape[0]:
            print(f"  ❌ 相关矩阵【奇异】: rank {rk} < {C.shape[0]}  → fisherz 必然失败 → 触发抖动重试")
        elif cond > 1e8:
            print(f"  ⚠️  相关矩阵【接近奇异】(cond={cond:.1e})")

        cols = list(sel.columns)
        dup = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = abs(C[i, j])
                if r > 0.999:
                    dup.append((cols[i], cols[j], r))
        dup.sort(key=lambda t: -t[2])
        print(f"  |r| > 0.999 的列对: {len(dup)}")
        for a, b, r in dup[:10]:
            tag = "  ← 两列都是 latency（豁免了去冗余！）" if ("_latency" in a and "_latency" in b) else ""
            print(f"     r={r:.6f}  {a}  ~  {b}{tag}")

        # --- 实跑 PC ---
        try:
            from causallearn.search.ConstraintBased.PC import pc as _pc
        except Exception as e:
            print(f"  [SKIP] 没有 causallearn: {e}\n")
            continue

        try:
            cg = _pc(X, alpha=0.05, indep_test="fisherz", show_progress=False)
            g = np.asarray(cg.G.graph)
            print(f"  ✅ PC(直接) 完成: shape={g.shape}  "
                  f"边数(==-1 计数)={int((g == -1).sum())}  非零元素={int((g != 0).sum())}")
        except Exception as e:
            print(f"  ❌ PC(直接) 抛异常: {type(e).__name__}: {e}")
            rng = np.random.default_rng(42)
            jit = rng.normal(0, 1e-6, size=X.shape) * (np.std(X, axis=0, keepdims=True) + 1e-10)
            try:
                cg = _pc(X + jit, alpha=0.05, indep_test="fisherz", show_progress=False)
                g = np.asarray(cg.G.graph)
                print(f"     抖动重试后: 边数={int((g == -1).sum())}  "
                      f"（这就是 runner 里 fallback 之前那一步）")
            except Exception as e2:
                print(f"     抖动重试仍失败: {type(e2).__name__}: {e2}  → runner 会降级为全零图")

        # --- 走 runner 的真实函数路径（pc_module.run_pc_baseline）---
        # 这一步是决定性的：直接跑 runner 用的那个函数，看它给多少边
        try:
            import shutil
            import tempfile
            from pc_module import run_pc_baseline
            tmpd = tempfile.mkdtemp(prefix="diagpc_")
            for ufw in (True, False):
                res = run_pc_baseline(
                    data=sel,
                    metric_names=list(sel.columns),
                    anomaly_scores=None,
                    output_dir=os.path.join(tmpd, f"w{int(ufw)}"),
                    t_fail=600.0,
                    alpha=0.05,
                    indep_test="fisherz",
                    use_full_window=ufw,
                )
                print(f"  run_pc_baseline(use_full_window={ufw}) → "
                      f"n_nodes={res['n_nodes']}  n_edges={res['n_edges']}  "
                      f"density={res['edge_density']}")
            shutil.rmtree(tmpd, ignore_errors=True)
        except Exception as e:
            import traceback
            print(f"  [ERR] run_pc_baseline 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
        print()

    print("=" * 80)
    print("判读要点：")
    print("  1) 若 rank < 列数（奇异）→ 根因是共线，需在筛选阶段加去共线保护")
    print("  2) 若出现「两列都是 latency」的高相关对 → 正是 keep 列豁免去冗余造成的漏洞")
    print("  3) 若 PC 直接调用就返回 0 边（无异常）→ 是算法/数据本身的问题，另作处理")
    print("  4) 看最后两行 run_pc_baseline：它 = runner 真正走的路径。若这里正常、")
    print("     而冒烟日志里是 0 边 → 说明冒烟跑的是旧代码/旧输出目录，重跑即可")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
