# -*- coding: utf-8 -*-
"""
阶段二 · 第 2 步（a）：从 1125 张因果图提取拓扑特征

输入：`结果/阶段二分析/逐例明细.csv`（案例清单，1125 行）+ 各方法的邻接矩阵文件
输出：`结果/阶段二分析/图特征.csv`

方向约定（三方法已统一）：
    adj[i, j] = 1  ⟺  i 是 j 的原因（行 = 原因，列 = 结果）
    依据：Granger `run_granger.py` 的 _run_pagerank（G.add_edge(j, i) 当 adj[i,j]!=0）
          PC      `pc_module.py:39-57` `_pc_graph_to_adj_matrix`（cg_graph[i,j]==-1 → adj[i,j]=1）
          PCMCI   `pcmci_module.py` matrix[cause, effect] != 0 → cause→effect

清洗（统一执行）：
    1. 二值化（!= 0 → 1）
    2. 对角线清零 —— PCMCI 的 `np.where(p值矩阵)` 会带入自环（OB 实测 9 个）

用法: python 阶段二_2a_图特征.py
"""
import csv
import json
import os
import sys
import time
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd

def _find_result_dir():
    """从脚本位置向上定位含 Granger/ 的「结果」目录（**不依赖 cwd**，避免写错位置）"""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, 'Granger')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError('无法定位「结果」目录（应含 Granger/）；脚本位于 %s'
                       % os.path.dirname(os.path.abspath(__file__)))


HERE = _find_result_dir()
ANA = os.path.join(HERE, '阶段二分析')
DETAIL = os.path.join(ANA, '逐例明细.csv')


# ============================================================
# 图读取（三种格式）
# ============================================================
def read_graph(method, ds, svc, fault, case):
    """返回 (adj: int 0/1 ndarray, names: list|None)；失败返回 (None, None)"""
    if method == 'Granger':
        base = os.path.join(HERE, 'Granger', 'output_granger_merged', ds,
                            '%s_%s' % (svc, fault), case)
        ap, npath = base + '_adj.npy', base + '_nodes.json'
        if not os.path.exists(ap):
            return None, None
        adj = np.load(ap)
        names = json.load(open(npath, encoding='utf-8')) if os.path.exists(npath) else None

    elif method == 'PC':
        d = os.path.join(HERE, 'pc_pcmci_v2', 'RE1-%s-pc' % ds.split('-')[1],
                         '%s_%s' % (svc, fault), case)
        p = os.path.join(d, 'pc_adj_matrix.csv')
        if not os.path.exists(p):
            return None, None
        df = pd.read_csv(p, index_col=0)
        adj = df.to_numpy()
        names = [str(c) for c in df.columns]

    elif method == 'PCMCI':
        root = os.path.join(HERE, 'pc_pcmci_v2', 'RE1-%s-pcmci' % ds.split('-')[1])
        # A) OB 布局：{svc}_{fault}/{case}/pcmci_adj_matrix.npy（无独立节点名）
        pa = os.path.join(root, '%s_%s' % (svc, fault), case, 'pcmci_adj_matrix.npy')
        if os.path.exists(pa):
            adj = np.load(pa)
            names = None
        else:
            # B) SS/TT 布局：**/{svc}_{fault}/{case}_adj_matrix.npy
            found = None
            for dirpath, _, files in os.walk(root):
                if os.path.basename(dirpath) == '%s_%s' % (svc, fault):
                    if '%s_adj_matrix.npy' % case in files:
                        found = dirpath
                        break
            if not found:
                return None, None
            adj = np.load(os.path.join(found, '%s_adj_matrix.npy' % case))
            np_ = os.path.join(found, '%s_node_names.json' % case)
            names = json.load(open(np_, encoding='utf-8')) if os.path.exists(np_) else None
    else:
        return None, None

    # ---- 统一清洗 ----
    adj = (np.asarray(adj) != 0).astype(np.int8)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return None, None
    np.fill_diagonal(adj, 0)
    return adj, names


# ============================================================
# 特征计算
# ============================================================
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2.0 * (cum / cum[-1]).sum()) / n)


def avg_shortest_path(G):
    """有向图平均最短路径（只统计可达的有序对；不含自环）"""
    if G.number_of_nodes() <= 1:
        return 0.0
    total, cnt = 0, 0
    for s in G.nodes():
        for t, d in nx.single_source_shortest_path_length(G, s).items():
            if t != s:
                total += d
                cnt += 1
    return float(total / cnt) if cnt else 0.0


def compute_features(adj, names, svc):
    n = adj.shape[0]
    n_edges = int(adj.sum())
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # from_numpy_array 会把 adj[i,j]!=0 视为边 i→j ✓（与我们的约定一致）

    indeg = np.array([d for _, d in G.in_degree()], dtype=float)
    outdeg = np.array([d for _, d in G.out_degree()], dtype=float)

    f = OrderedDict()
    f['n_nodes'] = n
    f['n_edges'] = n_edges
    f['edge_density'] = float(n_edges / max(n * (n - 1), 1))
    f['avg_degree'] = float(n_edges / n) if n else 0.0

    f['in_degree_max'] = float(indeg.max()) if n else 0.0
    f['in_degree_mean'] = float(indeg.mean()) if n else 0.0
    f['in_degree_std'] = float(indeg.std()) if n else 0.0
    f['out_degree_max'] = float(outdeg.max()) if n else 0.0
    f['out_degree_mean'] = float(outdeg.mean()) if n else 0.0
    f['out_degree_std'] = float(outdeg.std()) if n else 0.0
    f['degree_gini'] = gini(np.concatenate([indeg, outdeg]))

    f['avg_shortest_path'] = avg_shortest_path(G)

    # 最大强连通分量占比
    sccs = list(nx.strongly_connected_components(G))
    f['largest_scc_ratio'] = float(max((len(c) for c in sccs), default=0) / n) if n else 0.0

    # 互惠边比例：双向边数 / 总边数
    try:
        f['reciprocity'] = float(nx.reciprocity(G)) if n_edges else 0.0
    except Exception:
        f['reciprocity'] = np.nan
    # 双向边显式占比（更直观：A→B 且 B→A）
    both = int(((adj == 1) & (adj.T == 1)).sum() // 2)
    f['bidirectional_ratio'] = float(2 * both / n_edges) if n_edges else 0.0

    try:
        f['transitivity'] = float(nx.transitivity(G))
    except Exception:
        f['transitivity'] = np.nan

    # ---- D 组：真值节点的局部结构（需要节点名）----
    if names is not None and len(names) == n:
        tgt = [i for i, nm in enumerate(names)
               if str(nm).split('_')[0].replace('-db', '') == svc]
        if tgt:
            td = indeg[tgt] + outdeg[tgt]
            f['truth_degree_mean'] = float(td.mean())
            f['truth_degree_max'] = float(td.max())
            f['truth_isolated'] = int((td == 0).all())
            f['truth_n_nodes'] = len(tgt)
        else:
            f['truth_degree_mean'] = np.nan
            f['truth_degree_max'] = np.nan
            f['truth_isolated'] = np.nan
            f['truth_n_nodes'] = 0
    else:
        f['truth_degree_mean'] = np.nan
        f['truth_degree_max'] = np.nan
        f['truth_isolated'] = np.nan
        f['truth_n_nodes'] = np.nan

    return f


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    cases = list(csv.DictReader(open(DETAIL, encoding='utf-8-sig')))
    print('=' * 96)
    print('阶段二 · 第 2 步(a)：提取因果图拓扑特征')
    print('=' * 96)
    print('  案例清单: %d 行（来自 逐例明细.csv）' % len(cases))

    rows, fail = [], []
    for i, c in enumerate(cases, 1):
        m, ds, svc, fault, case = c['method'], c['dataset'], c['service'], c['fault'], c['case']
        adj, names = read_graph(m, ds, svc, fault, case)
        if adj is None:
            fail.append((m, ds, svc, fault, case))
            continue
        f = compute_features(adj, names, svc)
        row = OrderedDict([('method', m), ('dataset', ds), ('service', svc),
                           ('fault', fault), ('case', case)])
        row.update(f)
        rows.append(row)
        if i % 150 == 0:
            print('    ... %d / %d  (%.0fs)' % (i, len(cases), time.time() - t0))

    print('  ✅ 成功 %d 张，失败 %d 张  (耗时 %.0fs)' % (len(rows), len(fail), time.time() - t0))
    for x in fail[:10]:
        print('     ✗ %s' % '/'.join(x))

    out = os.path.join(ANA, '图特征.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n  → %s（%d 行 × %d 列）' % (os.path.basename(out), len(rows), len(rows[0])))

    # ---------- 特征概览 ----------
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ('method', 'dataset', 'service', 'fault', 'case')]
    print('\n' + '=' * 96)
    print('特征概览（按方法 × 数据集 分组的均值）')
    print('=' * 96)
    show = ['n_nodes', 'n_edges', 'edge_density', 'in_degree_max', 'degree_gini',
            'avg_shortest_path', 'largest_scc_ratio', 'bidirectional_ratio', 'truth_isolated']
    print('%-8s %-8s %8s %8s %11s %12s %11s %15s %15s %17s %13s' %
          ('method', 'dataset', 'n_nodes', 'n_edges', 'edge_density', 'in_deg_max',
           'degree_gini', 'avg_sht_path', 'largest_scc_r', 'bidirectional_r', 'truth_isolated'))
    print('-' * 130)
    for m in ['Granger', 'PC', 'PCMCI']:
        for ds in ['RE1-OB', 'RE1-SS', 'RE1-TT']:
            sub = df[(df.method == m) & (df.dataset == ds)]
            if not sub.empty:
                print('%-8s %-8s %8.1f %8.1f %11.4f %12.1f %11.3f %15.3f %15.3f %17.3f %13.3f' % (
                    m, ds, sub.n_nodes.mean(), sub.n_edges.mean(), sub.edge_density.mean(),
                    sub.in_degree_max.mean(), sub.degree_gini.mean(),
                    sub.avg_shortest_path.mean(), sub.largest_scc_ratio.mean(),
                    sub.bidirectional_ratio.mean(), sub.truth_isolated.mean()))
    print('\n✅ 完成。下一步：阶段二_2b_特征分析.py')


if __name__ == '__main__':
    main()
