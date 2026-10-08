# -*- coding: utf-8 -*-
"""
阶段二 · 第 1 步：主结果表 + 补充指标 + 统计检验

产出（写入 `结果/阶段二分析/`）：
  1. 逐例明细.csv          每个 (方法, 数据集, 服务, 故障, 案例) 一行的原始排名
  2. 主结果表.csv          方法 × 数据集 × 指标的宽表（含 Avg@5 / Avg@10 / MRR）
  3. 主结果表_分故障.csv   同上，但按故障类型细分
  4. wilcoxon_svc.csv      三对方法在服务级排名上的配对检验
  5. wilcoxon_mtr_ob.csv   指标级检验（仅 RE1-OB，因 SS/TT 口径不同不可比）
  6. 说明.md               结果解读与口径声明

口径：
  - 服务级：`-db` 归一 + 去重（官方 main.py:380）
  - 指标级：节点名精确等于 f"{service}_{FAULT_TO_METRIC[fault]}"（官方 main.py:394）
  - 未命中（不在图里）→ 记为 n_nodes + 1（哨兵），保证 AC@k / MRR / 检验口径统一

用法: python 阶段二_主结果表.py
"""
import csv
import glob
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pc_pcmci_fixed'))
from common_eval import compute_rank_accuracy  # noqa: E402

V2 = os.path.join(HERE, 'pc_pcmci_v2')
OUT = os.path.join(HERE, '阶段二分析')

GRANGER = {ds: os.path.join(HERE, 'Granger', 'output_granger_merged', ds)
           for ds in ('RE1-OB', 'RE1-SS', 'RE1-TT')}
PC = {ds: os.path.join(V2, 'RE1-%s-pc' % ds.split('-')[1]) for ds in ('RE1-OB', 'RE1-SS', 'RE1-TT')}
PCMCI = {ds: os.path.join(V2, 'RE1-%s-pcmci' % ds.split('-')[1]) for ds in ('RE1-OB', 'RE1-SS', 'RE1-TT')}

DS_ORDER = ['RE1-OB', 'RE1-SS', 'RE1-TT']
FAULT_ORDER = ['cpu', 'mem', 'delay', 'loss', 'disk']
METHODS = ['Granger', 'PC', 'PCMCI']


# ============================================================
# 加载：{(ds, svc, fault, case): [节点名按名次升序]}
# ============================================================
def _read_rank_file(path, name_col):
    try:
        rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    except Exception:
        return None
    if not rows or 'rank' not in rows[0] or name_col not in rows[0]:
        return None
    try:
        rows.sort(key=lambda r: int(r['rank']))
    except Exception:
        return None
    return [r[name_col] for r in rows]


def load_granger():
    out = {}
    for ds, root in GRANGER.items():
        for p in glob.glob(os.path.join(root, '*_*', '*.csv')):
            parts = p.replace('\\', '/').split('/')
            fname, svc_fault = parts[-1], parts[-2]
            if not fname[:-4].isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'node')
            if nodes:
                out[(ds, svc, fault, fname[:-4])] = nodes
    return out


def load_pc():
    out = {}
    for ds, root in PC.items():
        for p in glob.glob(os.path.join(root, '*_*', '*', 'pc_summary.csv')):
            parts = p.replace('\\', '/').split('/')
            case, svc_fault = parts[-2], parts[-3]
            if not case.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'metric')
            if nodes:
                out[(ds, svc, fault, case)] = nodes
    return out


def load_pcmci():
    """两种布局：OB 是 {svc}_{fault}/{case}/pcmci_summary.csv；SS/TT 是 **/{svc}_{fault}/{case}_ranks.csv"""
    out = {}
    for ds, root in PCMCI.items():
        for p in glob.glob(os.path.join(root, '*_*', '*', 'pcmci_summary.csv')):
            parts = p.replace('\\', '/').split('/')
            case, svc_fault = parts[-2], parts[-3]
            if not case.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'metric')
            if nodes:
                out[(ds, svc, fault, case)] = nodes
        for p in glob.glob(os.path.join(root, '**', '*_*', '*_ranks.csv'), recursive=True):
            parts = p.replace('\\', '/').split('/')
            base, svc_fault = parts[-1][:-len('_ranks.csv')], parts[-2]
            if not base.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'node')
            if nodes:
                out.setdefault((ds, svc, fault, base), nodes)
    return out


# ============================================================
# 逐例计算
# ============================================================
def build_detail(method, nodes_map):
    """返回 [{ds, method, service, fault, case, n_nodes, svc_rank, mtr_rank,
              svc_capped, mtr_capped, svc_found, mtr_found}]"""
    rows = []
    for (ds, svc, fault, case), nodes in sorted(nodes_map.items()):
        n = len(nodes)
        r = compute_rank_accuracy(nodes, svc, fault)
        s, m = r['Svc_Rank'], r['Mtr_Rank']
        s_found = (s is not None) and (s <= n)
        m_found = (m is not None) and (m <= n)
        rows.append({
            'method': method, 'dataset': ds, 'service': svc, 'fault': fault, 'case': case,
            'n_nodes': n,
            'svc_rank': s, 'mtr_rank': m,
            # 未命中 → 哨兵 n+1，便于 AC@k / MRR / 检验统一处理
            'svc_capped': s if s_found else n + 1,
            'mtr_capped': m if m_found else n + 1,
            'svc_found': int(s_found), 'mtr_found': int(m_found),
        })
    return rows


def metrics(ranks):
    """ranks: list[int]（已用哨兵填充）→ 指标字典"""
    n = len(ranks)
    a = np.asarray(ranks, dtype=float)
    out = OrderedDict()
    for k in range(1, 11):
        out['AC@%d' % k] = float((a <= k).mean())
    out['Avg@5'] = float(np.mean([out['AC@%d' % k] for k in range(1, 6)]))
    out['Avg@10'] = float(np.mean([out['AC@%d' % k] for k in range(1, 11)]))
    out['MRR'] = float(np.mean(1.0 / a))
    out['MedianRank'] = float(np.median(a))
    return out


# ============================================================
# Wilcoxon 配对检验
# ============================================================
def paired_test(pairs_a, pairs_b):
    """a、b 为对齐后的两列（同一批案例）。返回检验结果 dict。
       差值 d = a - b < 0 表示 a 的排名更靠前（更好）。"""
    a = np.asarray(pairs_a, dtype=float)
    b = np.asarray(pairs_b, dtype=float)
    d = a - b
    n = len(d)
    wins = int((d < 0).sum())
    ties = int((d == 0).sum())
    losses = int((d > 0).sum())
    if n == 0 or np.all(d == 0):
        return dict(n=n, wins=wins, ties=ties, losses=losses,
                    median_diff=0.0, W=np.nan, p=np.nan, p_less=np.nan, note='无差异')
    try:
        stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        _, p_less = wilcoxon(a, b, zero_method='wilcox', alternative='less')
    except Exception as e:
        return dict(n=n, wins=wins, ties=ties, losses=losses,
                    median_diff=float(np.median(d)), W=np.nan, p=np.nan, p_less=np.nan,
                    note=str(e))
    return dict(n=n, wins=wins, ties=ties, losses=losses,
                median_diff=float(np.median(d)), W=float(stat), p=float(p),
                p_less=float(p_less), note='')


def stars(p):
    if p != p:
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUT, exist_ok=True)

    print('=' * 100)
    print('阶段二 · 第 1 步：主结果表 + 补充指标 + 统计检验')
    print('=' * 100)

    # ---------- 加载 ----------
    detail = []
    for method, loader in [('Granger', load_granger), ('PC', load_pc), ('PCMCI', load_pcmci)]:
        m = loader()
        rows = build_detail(method, m)
        detail += rows
        print('  [LOAD] %-8s %3d 例' % (method, len(rows)))

    # 每方法每数据集 125 例校验
    cnt = defaultdict(int)
    for r in detail:
        cnt[(r['method'], r['dataset'])] += 1
    bad = [(k, v) for k, v in cnt.items() if v != 125]
    if bad:
        print('\n  ⚠️ 案例数异常: %s' % bad)
    else:
        print('  ✅ 案例数校验通过：三方法 × 三数据集 均为 125 例（合计 %d 行）' % len(detail))

    # ---------- 1) 逐例明细 ----------
    p1 = os.path.join(OUT, '逐例明细.csv')
    cols = ['method', 'dataset', 'service', 'fault', 'case', 'n_nodes',
            'svc_rank', 'mtr_rank', 'svc_capped', 'mtr_capped', 'svc_found', 'mtr_found']
    with open(p1, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in detail:
            w.writerow({k: r[k] for k in cols})
    print('\n  → %s（%d 行）' % (os.path.basename(p1), len(detail)))

    # ---------- 2) 主结果表 ----------
    def agg(rows):
        out = OrderedDict()
        out['n'] = len(rows)
        for k, v in metrics([r['svc_capped'] for r in rows]).items():
            out['Svc-' + k] = v
        for k, v in metrics([r['mtr_capped'] for r in rows]).items():
            out['Mtr-' + k] = v
        out['Svc_found'] = round(float(np.mean([r['svc_found'] for r in rows])), 4)
        out['Mtr_found'] = round(float(np.mean([r['mtr_found'] for r in rows])), 4)
        return out

    summaries = {}
    for method in METHODS:
        for ds in DS_ORDER:
            sel = [r for r in detail if r['method'] == method and r['dataset'] == ds]
            summaries[(method, ds)] = agg(sel)
        summaries[(method, 'ALL')] = agg([r for r in detail if r['method'] == method])

    key_cols = ['n', 'Svc-AC@1', 'Svc-AC@3', 'Svc-AC@5', 'Svc-AC@10',
                'Svc-Avg@5', 'Svc-Avg@10', 'Svc-MRR', 'Svc-MedianRank', 'Svc_found',
                'Mtr-AC@1', 'Mtr-AC@5', 'Mtr-MRR', 'Mtr_found']

    p2 = os.path.join(OUT, '主结果表.csv')
    with open(p2, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['method', 'dataset'] + key_cols)
        for method in METHODS:
            for ds in DS_ORDER + ['ALL']:
                s = summaries.get((method, ds))
                if not s:
                    continue
                w.writerow([method, ds] + ['%.4f' % s.get(c, 0) if c != 'n' else s['n']
                                           for c in key_cols])
    print('  → %s' % os.path.basename(p2))

    # ---------- 3) 分故障表 ----------
    p3 = os.path.join(OUT, '主结果表_分故障.csv')
    with open(p3, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['method', 'dataset', 'fault'] + key_cols)
        for method in METHODS:
            for ds in DS_ORDER:
                for fault in FAULT_ORDER:
                    sel = [r for r in detail if r['method'] == method
                           and r['dataset'] == ds and r['fault'] == fault]
                    if not sel:
                        continue
                    s = agg(sel)
                    w.writerow([method, ds, fault] +
                               ['%.4f' % s.get(c, 0) if c != 'n' else s['n'] for c in key_cols])
    print('  → %s' % os.path.basename(p3))

    # ---------- 4) Wilcoxon（服务级）----------
    # 按 (ds, svc, fault, case) 对齐
    idx = {m: {} for m in METHODS}
    for r in detail:
        idx[r['method']][(r['dataset'], r['service'], r['fault'], r['case'])] = r['svc_capped']

    pairs = [('Granger', 'PC'), ('Granger', 'PCMCI'), ('PC', 'PCMCI')]
    strata = [('ALL', None)] + [(ds, ds) for ds in DS_ORDER]

    p4 = os.path.join(OUT, 'wilcoxon_svc.csv')
    wl = [['method_A', 'method_B', 'stratum', 'n_pair', 'A_win', 'tie', 'A_lose',
           'median_rank_diff_A_minus_B', 'W', 'p_two_sided', 'p_one_sided_A_better',
           'sig', 'direction']]
    for A, B in pairs:
        for label, ds in strata:
            keys = [k for k in idx[A] if k in idx[B] and (ds is None or k[0] == ds)]
            keys.sort()
            a = [idx[A][k] for k in keys]
            b = [idx[B][k] for k in keys]
            res = paired_test(a, b)
            d = 'A 更好' if res['median_diff'] < 0 else ('B 更好' if res['median_diff'] > 0 else '无差异')
            wl.append([A, B, label, res['n'], res['wins'], res['ties'], res['losses'],
                       round(res['median_diff'], 3),
                       (round(res['W'], 1) if res['W'] == res['W'] else ''),
                       (round(res['p'], 6) if res['p'] == res['p'] else ''),
                       (round(res['p_less'], 6) if res['p_less'] == res['p_less'] else ''),
                       stars(res['p']), d])
    with open(p4, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(wl)
    print('  → %s' % os.path.basename(p4))

    # ---------- 5) Wilcoxon（指标级，仅 OB）----------
    idx_m = {m: {} for m in METHODS}
    for r in detail:
        idx_m[r['method']][(r['dataset'], r['service'], r['fault'], r['case'])] = r['mtr_capped']

    p5 = os.path.join(OUT, 'wilcoxon_mtr_ob.csv')
    wl2 = [['method_A', 'method_B', 'dataset', 'n_pair', 'A_win', 'tie', 'A_lose',
            'median_rank_diff_A_minus_B', 'W', 'p_two_sided', 'p_one_sided_A_better',
            'sig', 'direction']]
    for A, B in pairs:
        for ds in DS_ORDER:
            keys = sorted(k for k in idx_m[A] if k in idx_m[B] and k[0] == ds)
            res = paired_test([idx_m[A][k] for k in keys], [idx_m[B][k] for k in keys])
            d = 'A 更好' if res['median_diff'] < 0 else ('B 更好' if res['median_diff'] > 0 else '无差异')
            wl2.append([A, B, ds, res['n'], res['wins'], res['ties'], res['losses'],
                        round(res['median_diff'], 3),
                        (round(res['W'], 1) if res['W'] == res['W'] else ''),
                        (round(res['p'], 6) if res['p'] == res['p'] else ''),
                        (round(res['p_less'], 6) if res['p_less'] == res['p_less'] else ''),
                        stars(res['p']), d])
    with open(p5, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(wl2)
    print('  → %s' % os.path.basename(p5))

    # ---------- 控制台报告 ----------
    def line(s):
        print('-' * 100)
        print('%-9s %-8s %5s %8s %8s %8s %9s %9s %8s %8s' % s)

    print('\n' + '=' * 100)
    print('【服务级 Svc】AC@k / Avg@5 / Avg@10 / MRR')
    print('=' * 100)
    line(('method', 'dataset', 'n', 'AC@1', 'AC@3', 'AC@5', 'Avg@5', 'Avg@10', 'MRR', 'MedRank'))
    for method in METHODS:
        for ds in DS_ORDER + ['ALL']:
            s = summaries[(method, ds)]
            print('%-9s %-8s %5d %8.3f %8.3f %8.3f %9.4f %9.4f %8.4f %8.1f' % (
                method, ds, s['n'], s['Svc-AC@1'], s['Svc-AC@3'], s['Svc-AC@5'],
                s['Svc-Avg@5'], s['Svc-Avg@10'], s['Svc-MRR'], s['Svc-MedianRank']))
        print()

    print('=' * 100)
    print('【指标级 Mtr】AC@k / Avg@5 / Avg@10 / MRR   （⚠️ SS/TT 因命名口径不同，跨方法不可比）')
    print('=' * 100)
    line(('method', 'dataset', 'n', 'AC@1', 'AC@3', 'AC@5', 'Avg@5', 'Avg@10', 'MRR', 'Found'))
    for method in METHODS:
        for ds in DS_ORDER + ['ALL']:
            s = summaries[(method, ds)]
            print('%-9s %-8s %5d %8.3f %8.3f %8.3f %9.4f %9.4f %8.4f %8.3f' % (
                method, ds, s['n'], s['Mtr-AC@1'], s['Mtr-AC@3'], s['Mtr-AC@5'],
                s['Mtr-Avg@5'], s['Mtr-Avg@10'], s['Mtr-MRR'], s['Mtr_found']))
        print()

    print('=' * 100)
    print('【Wilcoxon 配对检验 · 服务级排名】（差值 = A 的排名 − B 的排名，负值表示 A 更好）')
    print('=' * 100)
    print('%-9s %-9s %-7s %6s %6s %4s %6s %9s %10s %11s %5s  %s' %
          ('A', 'B', '分层', 'n', 'A胜', '平', 'A负', '中位差', 'p(双侧)', 'p(单侧A更好)', 'sig', '结论'))
    print('-' * 116)
    for row in wl[1:]:
        print('%-9s %-9s %-7s %6d %6d %4d %6d %9.3f %10s %11s %5s  %s' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
            (('%.4g' % row[9]) if isinstance(row[9], float) else '-'),
            (('%.4g' % row[10]) if isinstance(row[10], float) else '-'),
            row[11], row[12]))

    print('\n  sig 说明: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著')

    print('\n' + '=' * 100)
    print('【Wilcoxon 配对检验 · 指标级排名】（⚠️ 仅 RE1-OB 三方法口径一致）')
    print('=' * 100)
    print('%-9s %-9s %-8s %6s %6s %4s %6s %9s %10s %11s %5s  %s' %
          ('A', 'B', '数据集', 'n', 'A胜', '平', 'A负', '中位差', 'p(双侧)', 'p(单侧A更好)', 'sig', '结论'))
    print('-' * 116)
    for row in wl2[1:]:
        print('%-9s %-9s %-8s %6d %6d %4d %6d %9.3f %10s %11s %5s  %s' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
            (('%.4g' % row[9]) if isinstance(row[9], float) else '-'),
            (('%.4g' % row[10]) if isinstance(row[10], float) else '-'),
            row[11], row[12]))

    print('\n✅ 全部产出已写入: %s' % OUT)


if __name__ == '__main__':
    main()
