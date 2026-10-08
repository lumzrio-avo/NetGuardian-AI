# -*- coding: utf-8 -*-
"""
Layer 1 v2 后处理：排除不可信的 (method, dataset) 层，重算聚合并补连续指标

排除理由：PCMCI·OB 的节点名无法与邻接矩阵索引对齐
（pcmci_summary.csv 的行序是排名序，不是 adj 索引序），
且其图本身是缺陷产物（近完全图、PageRank 分数差 1e-17、排名近似随机），
对它的扰动实验结果没有解释价值。

产出：阶段二分析/鲁棒性_图扰动_v2_clean.csv
"""
import csv
import os
from collections import defaultdict

import numpy as np

ANA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '阶段二分析')
SRC = os.path.join(ANA, '鲁棒性_图扰动_逐例_v2.csv')
DST = os.path.join(ANA, '鲁棒性_图扰动_v2_clean.csv')

EXCLUDE = {('PCMCI', 'RE1-OB')}     # 节点名索引无法对齐
MODES = ['drop', 'add', 'target_truth', 'target_rand']
RATIOS = [0.1, 0.3, 0.5]
DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']
METHODS = ['Granger', 'PC', 'PCMCI']


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
    keep = [r for r in rows if (r['method'], r['dataset']) not in EXCLUDE]
    print('原始 %d 行 → 保留 %d 行（排除 %d 行 PCMCI·OB）'
          % (len(rows), len(keep), len(rows) - len(keep)))

    def agg(sel):
        if not sel:
            return None
        f = lambda k: float(np.mean([float(r[k]) for r in sel]))
        return dict(
            n=len(sel),
            base_AC1=f('base_AC1'), AC1=f('AC1'), AC5=f('AC5'),
            base_norm=f('base_norm'), norm_rank=f('norm_rank'),
            norm_std=f('norm_rank_std'), edges=f('n_edges'),
        )

    combos = [('ALL', None)] + [(d, d) for d in DATASETS] \
        + [('ALL·' + m, m) for m in METHODS] \
        + [('%s·%s' % (m, d), None) for m in METHODS for d in DATASETS]

    out = []
    for label, key in combos:
        for mode in MODES:
            for ratio in RATIOS:
                sel = [r for r in keep if r['mode'] == mode
                       and abs(float(r['ratio']) - ratio) < 1e-9]
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
                    out.append(a)

    cols = ['group', 'mode', 'ratio', 'n', 'base_AC1', 'AC1', 'AC5',
            'base_norm', 'norm_rank', 'norm_std', 'edges']
    with open(DST, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in out:
            w.writerow({k: (round(a[k], 5) if isinstance(a[k], float) else a[k]) for k in cols})
    print('→ %s（%d 行）\n' % (os.path.basename(DST), len(out)))

    # ---- 核心表：定向 vs 随机（ALL）----
    print('=' * 92)
    print('【核心对照】删同样多的边，只改「删在哪里」（ALL，n=%d）' % out[0]['n'])
    print('=' * 92)
    print('%-6s %9s %9s %12s %11s %10s %11s' %
          ('ratio', 'drop', 'target_rand', 'target_truth', 'truth−rand', 'base_AC1', 'base_norm'))
    print('-' * 92)
    for ratio in RATIOS:
        d = {x['mode']: x for x in out if x['group'] == 'ALL' and x['ratio'] == ratio}
        print('%-6.2f %9.4f %9.4f %12.4f %11.4f %10.4f %11.4f' % (
            ratio, d['drop']['AC1'], d['target_rand']['AC1'], d['target_truth']['AC1'],
            d['target_truth']['AC1'] - d['target_rand']['AC1'],
            d['drop']['base_AC1'], d['drop']['base_norm']))

    # ---- 名次（连续指标）----
    print('\n' + '=' * 92)
    print('【归一化名次 norm_rank】越小越好（0=排第1，1=完全未命中）')
    print('=' * 92)
    print('%-6s %11s %11s %13s' % ('ratio', 'base', 'drop', 'target_truth'))
    print('-' * 92)
    for ratio in RATIOS:
        d = {x['mode']: x for x in out if x['group'] == 'ALL' and x['ratio'] == ratio}
        print('%-6.2f %11.4f %11.4f %13.4f' % (
            ratio, d['drop']['base_norm'], d['drop']['norm_rank'], d['target_truth']['norm_rank']))

    # ---- 各方法 ----
    print('\n' + '=' * 92)
    print('【各方法】target_truth vs target_rand（ratio=0.50）')
    print('=' * 92)
    print('%-9s %7s %11s %13s %11s %11s' %
          ('method', 'n', 'base_AC1', 'truth_AC1', 'rand_AC1', 'drop_AC1'))
    print('-' * 92)
    for m in METHODS:
        d = {x['mode']: x for x in out if x['group'] == 'ALL·' + m and abs(x['ratio'] - 0.5) < 1e-9}
        if len(d) < 4 or 'target_truth' not in d:
            continue
        g = d['drop']
        print('%-9s %7d %11.4f %13.4f %11.4f %11.4f' % (
            m, g['n'], g['base_AC1'], d['target_truth']['AC1'],
            d['target_rand']['AC1'], g['AC1']))

    # ---- 分数据集 ----
    print('\n' + '=' * 92)
    print('【分数据集】target_truth（ratio=0.50）')
    print('=' * 92)
    print('%-10s %7s %11s %13s %11s' % ('dataset', 'n', 'base_AC1', 'truth_AC1', 'rand_AC1'))
    print('-' * 92)
    for ds in DATASETS:
        d = {x['mode']: x for x in out if x['group'] == ds and abs(x['ratio'] - 0.5) < 1e-9}
        if 'target_truth' not in d:
            continue
        print('%-10s %7d %11.4f %13.4f %11.4f' % (
            ds, d['target_truth']['n'], d['target_truth']['base_AC1'],
            d['target_truth']['AC1'], d['target_rand']['AC1']))


if __name__ == '__main__':
    main()
