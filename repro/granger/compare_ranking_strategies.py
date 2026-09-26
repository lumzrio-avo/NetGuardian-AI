"""
瓶颈诊断：固定因果图，只换排序策略，看定位性能变化。

目的：回答"Granger 的性能瓶颈在因果图质量还是排序策略"。
  - 若换排序策略能显著提升 AC@k  ⇒ 瓶颈在排序策略
  - 若换排序策略基本不动        ⇒ 瓶颈在因果图质量

输入（需先重跑一次生成 adj）:
  output_granger_merged/{dataset}/{service}_{fault}/{case_id}_adj.npy
  output_granger_merged/{dataset}/{service}_{fault}/{case_id}_nodes.json
  （{case_id}.csv 提供基线 PageRank 的排名，用于一致性校验）

评估口径：严格对齐 RCAEval 官方 main.py
  - 服务级名次 = 节点实体去掉 '-db' 后**去重**，取真值服务首次出现的名次
  - AC@k = 名次 <= k 的比例；Avg@5 = sum(AC@1..AC@5)/5

用法:
  python compare_ranking_strategies.py
  python compare_ranking_strategies.py --datasets RE1-OB RE1-SS
"""
import os
import csv
import json
import glob
import platform
import argparse
import numpy as np
import networkx as nx

FAULT_TO_METRIC = {'delay': 'latency', 'loss': 'latency', 'disk': 'diskio'}
ALL_DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']
# 除基线外的策略（pagerank_causal 会先从 adj 重算并与 CSV 排名校验一致性）
STRATEGIES = ['pagerank_causal', 'pagerank_reverse', 'out_degree', 'in_degree',
              'degree', 'betweenness', 'closeness', 'eigenvector']


def to_local(p):
    if platform.system() != 'Windows' and len(p) > 2 and p[1] == ':':
        return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')
    return p


# ------------------------------------------------------------
# 排序策略：输入邻接矩阵，返回“节点下标 → 名次(1-based)”的排序
# ------------------------------------------------------------

def make_graphs(adj):
    """
    adj[i, j] == 1 表示 'j Granger-causes i'（对齐官方 granger.py 的注释 test j -> i）。

    G_causal: 边 j→i，即 因 → 果（我们的 run_granger.py 采用的方向）
    G_reverse: 边 i→j，即 果 → 因
    """
    n = adj.shape[0]
    effects, causes = np.nonzero(adj)          # adj[i,j]=1 → i=果, j=因
    G_causal = nx.DiGraph()
    G_causal.add_nodes_from(range(n))
    G_causal.add_edges_from(zip(causes.tolist(), effects.tolist()))
    G_reverse = G_causal.reverse()
    return G_causal, G_reverse


def compute_strategies(adj):
    """返回 {策略名: [节点下标按得分降序]}"""
    n = adj.shape[0]
    G, Gr = make_graphs(adj)
    has_edge = G.number_of_edges() > 0
    out = {}

    def order_by(score):
        return sorted(range(n), key=lambda i: (-score[i], i))

    pr_c = nx.pagerank(G, alpha=0.85) if has_edge else {i: 0.0 for i in range(n)}
    pr_r = nx.pagerank(Gr, alpha=0.85) if has_edge else {i: 0.0 for i in range(n)}
    out['pagerank_causal'] = order_by(pr_c)
    out['pagerank_reverse'] = order_by(pr_r)
    out['out_degree'] = order_by([G.out_degree(i) for i in range(n)])
    out['in_degree'] = order_by([G.in_degree(i) for i in range(n)])
    out['degree'] = order_by([G.degree(i) for i in range(n)])

    U = G.to_undirected()
    if has_edge:
        # betweenness / closeness 复杂度约 O(n*m)，大图（TT 761 节点 / 10.7 万边）会非常慢
        if n <= 400:
            bc = nx.betweenness_centrality(U)
            cc = nx.closeness_centrality(U)
        else:
            bc = nx.betweenness_centrality(U, k=200, seed=42)   # 抽样近似
            cc = {}                                             # 直接跳过
        out['betweenness'] = order_by([bc.get(i, 0.0) for i in range(n)])
        out['closeness'] = order_by([cc.get(i, 0.0) for i in range(n)])
        try:
            eig = nx.eigenvector_centrality_numpy(U)
        except Exception:
            eig = {i: 0.0 for i in range(n)}
        out['eigenvector'] = order_by([eig.get(i, 0.0) for i in range(n)])
    else:
        for k in ['betweenness', 'closeness', 'eigenvector']:
            out[k] = list(range(n))
    return out


# ------------------------------------------------------------
# 官方口径的服务级名次
# ------------------------------------------------------------

def service_rank(node_order, node_names, service):
    """node_order: 节点下标序列；返回真值服务在**去重后**序列中的名次（1-based）。"""
    seen, dedup = set(), []
    for idx in node_order:
        ent = node_names[idx].split('_')[0].replace('-db', '')
        if ent not in seen:
            seen.add(ent)
            dedup.append(ent)
    return dedup.index(service) + 1 if service in dedup else None


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', type=str,
                    default=to_local(r'D:\北交威\大创\结果\Granger\output_granger_merged'))
    ap.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    out_csv = args.out or os.path.join(args.merged, 'ranking_strategy_comparison.csv')

    # hits[ds][strategy][k] = 命中数
    hits = {}
    counts = {}
    mismatch = {'total': 0, 'same': 0, 'nodata': 0}

    for ds in args.datasets:
        base = os.path.join(args.merged, ds)
        if not os.path.isdir(base):
            print(f'[SKIP] {ds}: 目录不存在')
            continue
        hits[ds] = {s: {k: 0 for k in range(1, 6)} for s in STRATEGIES}
        counts[ds] = 0

        for adj_path in sorted(glob.glob(os.path.join(base, '*', '*_adj.npy'))):
            svc_fault = os.path.basename(os.path.dirname(adj_path))
            if '_' not in svc_fault:
                continue
            service = svc_fault.split('_', 1)[0]
            case_id = os.path.basename(adj_path).replace('_adj.npy', '')
            nodes_path = os.path.join(os.path.dirname(adj_path), f'{case_id}_nodes.json')
            if not os.path.exists(nodes_path):
                continue

            adj = np.load(adj_path)
            with open(nodes_path, encoding='utf-8') as f:
                node_names = json.load(f)
            if adj.shape[0] != len(node_names):
                print(f'[WARN] 维度不匹配: {adj_path}')
                continue

            orders = compute_strategies(adj)

            # 一致性校验：用 adj 重算的 pagerank_causal 应与已保存的 case CSV 排名一致
            csv_path = os.path.join(os.path.dirname(adj_path), f'{case_id}.csv')
            if os.path.exists(csv_path):
                with open(csv_path, encoding='utf-8-sig') as f:
                    saved = [r['node'] for r in csv.DictReader(f) if r.get('node')]
                recomputed = [node_names[i] for i in orders['pagerank_causal']]
                mismatch['total'] += 1
                # 只比较名次前 10，避免并列打分的噪声
                if len(saved) >= 10 and len(recomputed) >= 10:
                    if saved[:10] == recomputed[:10]:
                        mismatch['same'] += 1
            else:
                mismatch['nodata'] += 1

            counts[ds] += 1
            for s, order in orders.items():
                r = service_rank(order, node_names, service)
                if r is None:
                    continue
                for k in range(1, 6):
                    if r <= k:
                        hits[ds][s][k] += 1

    # ---------- 输出 ----------
    rows = []
    print('=' * 100)
    print('瓶颈诊断：固定因果图，只换排序策略（服务级，官方去重口径）')
    print('=' * 100)
    for ds in args.datasets:
        if ds not in counts or counts[ds] == 0:
            continue
        n = counts[ds]
        print(f'\n--- {ds}  (n={n}) ---')
        print(f'{"策略":<20}{"AC@1":>8}{"AC@3":>8}{"AC@5":>8}{"Avg@5":>9}   vs 基线')
        print('-' * 100)
        base_ac5 = hits[ds]['pagerank_causal'][5] / n
        for s in STRATEGIES:
            ac = {k: hits[ds][s][k] / n for k in range(1, 6)}
            avg5 = sum(ac[k] for k in range(1, 6)) / 5
            delta = ac[5] - base_ac5
            tag = '（基线）' if s == 'pagerank_causal' else f'{delta:+.4f}'
            print(f'{s:<20}{ac[1]:>8.4f}{ac[3]:>8.4f}{ac[5]:>8.4f}{avg5:>9.4f}   {tag}')
            rows.append([ds, s, n, round(ac[1], 4), round(ac[3], 4), round(ac[5], 4),
                         round(avg5, 4), '' if s == 'pagerank_causal' else round(delta, 4)])

    # 跨数据集汇总（加权）
    if len([d for d in args.datasets if counts.get(d)]) > 1:
        print(f'\n--- ALL（三数据集加权）---')
        print(f'{"策略":<20}{"AC@1":>8}{"AC@3":>8}{"AC@5":>8}{"Avg@5":>9}')
        print('-' * 100)
        tot = sum(counts[d] for d in args.datasets if counts.get(d))
        for s in STRATEGIES:
            ac = {k: sum(hits[d][s][k] for d in args.datasets if counts.get(d)) / tot
                  for k in range(1, 6)}
            avg5 = sum(ac[k] for k in range(1, 6)) / 5
            print(f'{s:<20}{ac[1]:>8.4f}{ac[3]:>8.4f}{ac[5]:>8.4f}{avg5:>9.4f}')
            rows.append(['ALL', s, tot, round(ac[1], 4), round(ac[3], 4),
                         round(ac[5], 4), round(avg5, 4), ''])

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'strategy', 'n_cases', 'svc_ac1', 'svc_ac3',
                    'svc_ac5', 'svc_avg5', 'delta_ac5_vs_baseline'])
        w.writerows(rows)

    print('\n' + '=' * 100)
    print(f'一致性校验（用 adj 重算的 PageRank vs 已保存 CSV 的前 10 名）: '
          f'{mismatch["same"]}/{mismatch["total"]} 一致，{mismatch["nodata"]} 个缺 CSV')
    print(f'结果已写入: {out_csv}')
    print('\n判读: 若某策略 AC@5 明显高于基线 ⇒ 瓶颈在排序策略；'
          '\n      若所有策略都差不多 ⇒ 瓶颈在因果图质量。')


if __name__ == '__main__':
    main()
