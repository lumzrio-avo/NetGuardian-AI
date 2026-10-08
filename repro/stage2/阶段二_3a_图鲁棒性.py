# -*- coding: utf-8 -*-
"""
阶段二 · 第 3 步(a) — Layer 1（v2）：因果图的鲁棒性

【v2 相对 v1 的改动】
  ① 新增「定向扰动」—— 这是 v2 的核心
       v1 发现：随机删边几乎不影响性能（删掉一半边，整体 AC@1 仅 −1.6pt）。
       ⇒ 随机误差不是定位失败的原因。那就反过来问：**如果是真值节点周围的边被删呢？**
       v2 让四种扰动删除【完全相同数量】的边，只改变"删在哪里"：
         drop          随机删边（v1 已有，作对照）
         add           随机加边（对照）
         target_truth  优先删「与真值节点相连」的边
         target_rand   优先删「与随机 K 个节点相连」的边（K = 真值节点数，位置对照）
       ⇒ 若 target_truth 显著劣于 target_rand，即证明「真值节点的连接」是性能命门
         （把第 2 步 ρ=−0.479 的「相关」升级为「干预证据」）
  ② 补回 PCMCI·OB 的 125 例（该布局无独立节点名文件，从 pcmci_summary.csv 的 metric 列恢复）
  ③ 加 ratio=0 内部基线（重算基线与 逐例明细 有 2.1% 的 tie 排序差异，改用内部基线自洽）
  ④ REPEAT 3 → 20（v1 的噪声与效应同量级）
  ⑤ PageRank 改 scipy.sparse 稀疏幂迭代（稠密版在 REPEAT=20 下要跑 1 小时以上）

【口径】（与结果/Granger/run_granger.py:317-340 逐行一致）
  adj[i, j] = 1  ⟺  j 是 i 的原因  →  建边 j→i
  PageRank alpha=0.85 → 按分数降序；无边图回退原始节点顺序
  服务级排名 / 未命中哨兵 n+1 → 复用 pc_pcmci_fixed/common_eval.py

用法: python 阶段二_3a_图鲁棒性.py
"""
import csv
import importlib.util
import os
import random
import sys
import time

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

# ------------------------------------------------------------
def _find_result_dir():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, 'Granger')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError('无法定位「结果」目录；脚本位于 %s' % os.path.dirname(os.path.abspath(__file__)))


HERE = _find_result_dir()
ANA = os.path.join(HERE, '阶段二分析')
DETAIL = os.path.join(ANA, '逐例明细.csv')

sys.path.insert(0, os.path.join(HERE, '..', 'pc_pcmci_fixed'))
from common_eval import compute_rank_accuracy  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'g2a', os.path.join(HERE, '阶段二_2a_图特征.py'))
_g2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g2a)
_read_graph = _g2a.read_graph

MODES = ['drop', 'add', 'target_truth', 'target_rand']
RATIOS = [0.1, 0.3, 0.5]
REPEAT = 20
SEED = 20261001
METHODS = ['Granger', 'PC', 'PCMCI']
DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']


# ============================================================
# 图读取：在 2a 的基础上补 PCMCI·OB 的节点名
# ============================================================
def read_graph_v2(method, ds, svc, fault, case):
    adj, names = _read_graph(method, ds, svc, fault, case)
    if adj is None:
        return None, None
    if names is None or len(names) != adj.shape[0]:
        # PCMCI·OB 布局：只有 pcmci_adj_matrix.npy + pcmci_summary.csv（含 metric 列）
        if method == 'PCMCI':
            root = os.path.join(HERE, 'pc_pcmci_v2', 'RE1-%s-pcmci' % ds.split('-')[1])
            sm = os.path.join(root, '%s_%s' % (svc, fault), case, 'pcmci_summary.csv')
            if os.path.exists(sm):
                rows = list(csv.DictReader(open(sm, encoding='utf-8-sig')))
                try:
                    rows.sort(key=lambda r: int(r['rank']))
                except Exception:
                    return adj, None
                cand = [str(r['metric']) for r in rows]
                if len(cand) == adj.shape[0]:
                    names = cand
    if names is None or len(names) != adj.shape[0]:
        return adj, None
    return adj, [str(x) for x in names]


# ============================================================
# PageRank（scipy 稀疏幂迭代；与 networkx 等价）
# ============================================================
def pagerank_scores(adj, alpha=0.85, tol=1e-10, max_iter=200):
    """adj[i,j]=1 ⟺ j→i；M[i,j]=1/outdeg[j]；dangling 分数均分（networkx 同款）"""
    n = adj.shape[0]
    A = adj.astype(np.float64)
    outdeg = A.sum(axis=0)
    dangling = (outdeg == 0)
    M = csr_matrix(A / np.where(dangling, 1.0, outdeg)[None, :])
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (M @ r + r[dangling].sum() / n) + (1.0 - alpha) / n
        if np.abs(r_new - r).sum() < tol * n:
            return r_new
        r = r_new
    return r


def pagerank_rank(adj, names):
    if adj.sum() == 0:
        return list(names)
    r = pagerank_scores(adj)
    order = np.argsort(-r, kind='stable')      # 稳定排序：同分保持索引序
    return [names[i] for i in order]


def pagerank_rank_nx(adj, names):
    """参考实现（networkx），仅用于自检对拍"""
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    ri, ci = np.nonzero(adj)
    for i, j in zip(ri.tolist(), ci.tolist()):
        G.add_edge(j, i)
    pr = nx.pagerank(G, alpha=0.85)
    return [x[0] for x in sorted(((names[i], pr.get(i, 0.0)) for i in range(n)),
                                 key=lambda x: x[1], reverse=True)]


# ============================================================
# 扰动
# ============================================================
def truth_nodes(names, svc):
    return [i for i, nm in enumerate(names) if nm.split('_')[0].replace('-db', '') == svc]


def edge_mask_of_nodes(n, rows, cols, nodes):
    """布尔掩码：标记 rows/cols 中「至少一端落在 nodes 里」的边"""
    m = np.zeros(n, dtype=bool)
    m[np.asarray(nodes, dtype=int)] = True
    return m[rows] | m[cols]


def perturb(adj, ratio, mode, rng, names=None, svc=None, truth_idx=None):
    """返回扰动后的邻接矩阵副本。四种模式删除/新增的【边数完全相同】，只有位置不同。"""
    A = adj.copy()
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    n_edges = len(rows)
    if ratio <= 0 or n_edges == 0:
        return A

    if mode == 'add':
        k = int(round(ratio * n_edges))
        got, tries, cap = 0, 0, k * 60 + 2000
        while got < k and tries < cap:
            i = rng.randrange(n)
            j = rng.randrange(n)
            tries += 1
            if i != j and A[i, j] == 0:
                A[i, j] = 1
                got += 1
        return A

    k_total = int(round(ratio * n_edges))
    if mode == 'drop':
        sel = rng.sample(range(n_edges), min(k_total, n_edges))
    else:
        if mode == 'target_truth':
            nodes = truth_idx
        else:                                   # target_rand：随机同数量节点
            k_nodes = min(len(truth_idx) if truth_idx else 1, n)
            nodes = rng.sample(range(n), k_nodes)
        if not nodes:
            return A
        mask = edge_mask_of_nodes(n, rows, cols, nodes)
        pri = np.nonzero(mask)[0].tolist()
        if not pri:
            sel = rng.sample(range(n_edges), min(k_total, n_edges))
        else:
            take = min(k_total, len(pri))
            sel = rng.sample(pri, take)
            if take < k_total:
                rest = np.nonzero(~mask)[0].tolist()
                sel += rng.sample(rest, min(k_total - take, len(rest)))
    A[rows[sel], cols[sel]] = 0
    return A


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 例（冒烟用）')
    ap.add_argument('--repeat', type=int, default=REPEAT, help='每档重复次数')
    args = ap.parse_args()
    rep_n = args.repeat

    t0 = time.time()
    cases = list(csv.DictReader(open(DETAIL, encoding='utf-8-sig')))
    if args.limit:
        cases = cases[:args.limit]
    print('=' * 100)
    print('阶段二 · 第 3 步(a) Layer 1 v2：因果图鲁棒性（随机 + 定向扰动）')
    print('=' * 100)
    print('  案例 %d | 模式 %s | 档位 %s | 重复 %d' % (len(cases), MODES, RATIOS, rep_n))

    rng = random.Random(SEED)

    # ---------- 加载 + 自检 ----------
    loaded, skip, mismatch = [], [], []
    nx_ok, nx_bad = 0, []
    for c in cases:
        m, ds, svc, fault, case = c['method'], c['dataset'], c['service'], c['fault'], c['case']
        adj, names = read_graph_v2(m, ds, svc, fault, case)
        if adj is None:
            skip.append((m, ds, svc, fault, case, '无邻接矩阵'))
            continue
        if names is None:
            skip.append((m, ds, svc, fault, case, '无节点名'))
            continue

        # 自检 1：稀疏幂迭代 vs networkx（前 30 例逐位对拍）
        if len(loaded) < 30:
            if pagerank_rank(adj, names) == (pagerank_rank_nx(adj, names) if adj.sum() else list(names)):
                nx_ok += 1
            else:
                nx_bad.append((m, ds, svc, fault, case))

        # 自检 2：未扰动重算 vs 逐例明细（基线真值）
        base_rank = compute_rank_accuracy(pagerank_rank(adj, names), svc, fault)['Svc_Rank']
        if int(c['svc_rank']) != int(base_rank):
            mismatch.append((m, ds, svc, fault, case, c['svc_rank'], base_rank))

        n = adj.shape[0]
        ti = truth_nodes(names, svc)
        loaded.append(dict(method=m, dataset=ds, service=svc, fault=fault, case=case,
                           adj=adj, names=names, n=n, truth_idx=ti,
                           base_rank=int(c['svc_rank']), base_found=int(c['svc_found'])))

    print('\n  [加载] 成功 %d 例，跳过 %d 例  (%.0fs)' % (len(loaded), len(skip), time.time() - t0))
    if skip:
        from collections import Counter
        for r, cnt in Counter(x[5] for x in skip).items():
            print('     跳过 %-12s %d 例' % (r, cnt))

    print('  [自检 1] 稀疏幂迭代 vs networkx 逐位一致：%d/30' % nx_ok)
    for x in nx_bad[:5]:
        print('     ⚠️ %s/%s/%s_%s/%s' % x)
    print('  [自检 2] 与 逐例明细 基线对拍一致率 = %.4f (%d/%d)'
          % (1 - len(mismatch) / max(len(loaded), 1), len(loaded) - len(mismatch), len(loaded)))

    # ---------- 扰动 ----------
    detail = []
    for idx, rec in enumerate(loaded, 1):
        adj, names, n = rec['adj'], rec['names'], rec['n']
        svc, fault, ti = rec['service'], rec['fault'], rec['truth_idx']
        n_edges = int(adj.sum())

        def measure(Ap):
            r = compute_rank_accuracy(pagerank_rank(Ap, names), svc, fault)['Svc_Rank']
            r = min(int(r), n + 1)
            return r / (n + 1.0), int(Ap.sum()), r

        # 内部基线（ratio=0）
        b_norm, b_edges, b_rank = measure(adj)

        for mode in MODES:
            for ratio in RATIOS:
                if mode.startswith('target') and not ti:
                    continue                      # 真值节点不在图里，定向模式无意义
                norms, ac1, ac5, ac10, ted = [], [], [], [], []
                for _ in range(rep_n):
                    Ap = perturb(adj, ratio, mode, rng, names, svc, ti)
                    nr, ne, rr = measure(Ap)
                    norms.append(nr)
                    ted.append(ne)
                    ac1.append(1.0 if rr <= 1 else 0.0)
                    ac5.append(1.0 if rr <= 5 else 0.0)
                    ac10.append(1.0 if rr <= 10 else 0.0)
                detail.append(dict(
                    method=rec['method'], dataset=rec['dataset'], service=svc, fault=fault,
                    case=rec['case'], mode=mode, ratio=ratio,
                    n_nodes=n, n_edges=n_edges, n_edges_after=round(float(np.mean(ted)), 2),
                    n_truth=len(ti), base_found=rec['base_found'],
                    base_norm=round(b_norm, 6), base_rank=b_rank,
                    norm_rank=round(float(np.mean(norms)), 6),
                    norm_rank_std=round(float(np.std(norms)), 6),
                    AC1=round(float(np.mean(ac1)), 6),
                    AC5=round(float(np.mean(ac5)), 6),
                    AC10=round(float(np.mean(ac10)), 6),
                    base_AC1=round(1.0 if b_rank <= 1 else 0.0, 6),
                    base_AC5=round(1.0 if b_rank <= 5 else 0.0, 6),
                ))
        if idx % 100 == 0:
            print('     ... %d / %d  (%.0fs)' % (idx, len(loaded), time.time() - t0))

    print('\n  ✅ 扰动完成：%d 条逐例结果  (%.0fs)' % (len(detail), time.time() - t0))

    # ---------- 聚合 ----------
    def agg(rows):
        if not rows:
            return None
        return dict(
            n=len(rows),
            mean_edges=float(np.mean([r['n_edges'] for r in rows])),
            mean_edges_after=float(np.mean([r['n_edges_after'] for r in rows])),
            base_AC1=float(np.mean([r['base_AC1'] for r in rows])),
            AC1=float(np.mean([r['AC1'] for r in rows])),
            base_AC5=float(np.mean([r['base_AC5'] for r in rows])),
            AC5=float(np.mean([r['AC5'] for r in rows])),
            base_norm=float(np.mean([r['base_norm'] for r in rows])),
            norm_rank=float(np.mean([r['norm_rank'] for r in rows])),
        )

    combos = [('ALL', None)]
    combos += [(ds, ds) for ds in DATASETS]
    combos += [('ALL·' + m, m) for m in METHODS]
    combos += [('%s·%s' % (m, ds), None) for m in METHODS for ds in DATASETS]

    agg_rows = []
    for label, key in combos:
        for mode in MODES:
            for ratio in RATIOS:
                sel = [r for r in detail if r['mode'] == mode and r['ratio'] == ratio]
                if label != 'ALL':
                    if label.startswith('ALL·'):
                        sel = [r for r in sel if r['method'] == key]
                    elif '·' in label:
                        mm, dd = label.split('·')
                        sel = [r for r in sel if r['method'] == mm and r['dataset'] == dd]
                    else:
                        sel = [r for r in sel if r['dataset'] == key]
                a = agg(sel)
                if a:
                    a.update(group=label, mode=mode, ratio=ratio)
                    agg_rows.append(a)

    # ---------- 落盘 ----------
    p1 = os.path.join(ANA, '鲁棒性_图扰动_逐例_v2.csv')
    with open(p1, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    print('  → %s（%d 行）' % (os.path.basename(p1), len(detail)))

    cols = ['group', 'mode', 'ratio', 'n', 'mean_edges', 'mean_edges_after',
            'base_AC1', 'AC1', 'AC5', 'base_AC5', 'base_norm', 'norm_rank']
    p2 = os.path.join(ANA, '鲁棒性_图扰动_v2.csv')
    with open(p2, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in agg_rows:
            w.writerow({k: (round(a[k], 4) if isinstance(a[k], float) else a[k]) for k in cols})
    print('  → %s（%d 行）' % (os.path.basename(p2), len(agg_rows)))

    # ---------- 报告 ----------
    def table(title, filt, show_edges=False):
        print('\n' + '=' * 104)
        print(title)
        print('=' * 104)
        hdr = '%-16s %-14s %-6s %7s %9s %9s %10s %10s' % (
            'group', 'mode', 'ratio', 'n', 'base_AC1', 'AC1', 'ΔAC1', 'ΔAC1%')
        if show_edges:
            hdr += ' %9s %9s' % ('edges', '→after')
        print(hdr)
        print('-' * 104)
        for a in agg_rows:
            if not filt(a):
                continue
            d = a['AC1'] - a['base_AC1']
            pct = (d / a['base_AC1'] * 100) if a['base_AC1'] > 0 else float('nan')
            line = '%-16s %-14s %-6.2f %7d %9.4f %9.4f %10.4f %9.1f%%' % (
                a['group'], a['mode'], a['ratio'], a['n'],
                a['base_AC1'], a['AC1'], d, pct)
            if show_edges:
                line += ' %9.1f %9.1f' % (a['mean_edges'], a['mean_edges_after'])
            print(line)

    table('【表 1】整体（ALL）: 四种扰动 × 三档', lambda a: a['group'] == 'ALL', True)

    print('\n' + '=' * 104)
    print('【表 2】★ 核心对照：删同样多的边，删在"真值节点周围" vs "随机节点周围"（ALL）')
    print('=' * 104)
    print('%-8s %9s %9s %9s %9s %10s   %s' %
          ('ratio', 'drop', 'target_rand', 'target_truth', 'truth−rand', 'base_AC1', '解读'))
    print('-' * 104)
    for ratio in RATIOS:
        d = {a['mode']: a for a in agg_rows if a['group'] == 'ALL' and a['ratio'] == ratio}
        if not all(k in d for k in ('drop', 'target_rand', 'target_truth')):
            continue
        diff = d['target_truth']['AC1'] - d['target_rand']['AC1']
        note = '⚠️ 定向更差' if diff < -0.02 else ('定向更好?' if diff > 0.02 else '≈ 无差异')
        print('%-8.2f %9.4f %9.4f %9.4f %9.4f %10.4f   %s' % (
            ratio, d['drop']['AC1'], d['target_rand']['AC1'],
            d['target_truth']['AC1'], diff, d['drop']['base_AC1'], note))

    table('【表 3】各方法（ALL·方法）', lambda a: a['group'].startswith('ALL·'))

    print('\n' + '=' * 104)
    print('【表 4】分数据集（数据集 × 模式，ratio=0.5）')
    print('=' * 104)
    print('%-10s %-14s %7s %9s %9s %10s' % ('dataset', 'mode', 'n', 'base_AC1', 'AC1', 'ΔAC1'))
    print('-' * 104)
    for a in agg_rows:
        if a['group'] in DATASETS and abs(a['ratio'] - 0.5) < 1e-9:
            print('%-10s %-14s %7d %9.4f %9.4f %10.4f' % (
                a['group'], a['mode'], a['n'], a['base_AC1'], a['AC1'],
                a['AC1'] - a['base_AC1']))

    print('\n✅ 完成。下一步：阶段二_3b_适用边界矩阵.py')


if __name__ == '__main__':
    main()
