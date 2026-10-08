# -*- coding: utf-8 -*-
"""
阶段二 · 第 2 步（b）：图拓扑特征 × 定位性能 的分析

输入：`结果/阶段二分析/图特征.csv` + `逐例明细.csv`
输出（全部在 `结果/阶段二分析/`）：
    相关_分层.csv            A 组全局结构特征 × 9 分层 → ρ, p, q(FDR)   ← 主结果
    相关_合并.csv            层内标准化后合并（n=1125），消除规模差异
    机制_真值节点.csv         真值节点局部特征（强信号所在）
    PCMCI_度分布对比.csv      PCMCI 为何失效：度分布均匀性对比
    特征相关矩阵.csv / 图_*.png
    说明.md

分析设计：
  因变量   norm_rank = svc_capped / (n_nodes + 1)   ∈ (0,1]，越小越好
  分层     9 层（方法 × 数据集，各 125 张图）
  两种口径
      ① 分层 Spearman  —— 层内看，规模相近、可比
      ② 层内标准化后合并 Spearman —— 保留全部样本、消除层间规模差异
  三重校正 BH-FDR

  ⚠️ 共线性处理：
      avg_degree ≡ in_degree_mean ≡ out_degree_mean（数学恒等，n_edges/n_nodes）
      reciprocity ≡ bidirectional_ratio（同一概念两种算法）
      → 各保留一个，避免 OLS 的 VIF 爆炸

用法: python 阶段二_2b_特征分析.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _find_result_dir():
    """从脚本位置向上定位含 Granger/ 的「结果」目录（不依赖 cwd）"""
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

# A 组：全局结构特征（已去恒等/重复项）
GLOBAL_FEATS = ['n_nodes', 'n_edges', 'edge_density', 'avg_degree',
                'in_degree_max', 'in_degree_std',
                'out_degree_max', 'out_degree_std',
                'degree_gini', 'avg_shortest_path', 'largest_scc_ratio',
                'bidirectional_ratio', 'transitivity']
# B 组：真值节点的局部结构特征（机制分析）
MECH_FEATS = ['truth_degree_mean', 'truth_degree_max', 'truth_degree_ratio']
METHODS = ['Granger', 'PC', 'PCMCI']
DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']
LAYERS = [(m, d) for m in METHODS for d in DATASETS]


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(n) + 1.0)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(qv, 0, 1)
    q[ok] = out
    return q


def stars(p, q=None):
    v = q if (q is not None and q == q) else p
    if v != v:
        return 'n/a'
    for th, s in ((0.001, '***'), (0.01, '**'), (0.05, '*')):
        if v < th:
            return s
    return 'ns'


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.unique(x[ok]).size < 2:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def main():
    print('=' * 106)
    print('阶段二 · 第 2 步(b)：图拓扑特征 × 定位性能')
    print('=' * 106)

    feat = pd.read_csv(os.path.join(ANA, '图特征.csv'))
    det = pd.read_csv(os.path.join(ANA, '逐例明细.csv'))
    key = ['method', 'dataset', 'service', 'fault', 'case']
    df = feat.merge(det[key + ['svc_capped']], on=key, how='inner')
    df['norm_rank'] = df['svc_capped'] / (df['n_nodes'] + 1.0)
    # 真值节点度 / 全图平均度（>1 表示真值节点比平均连得多）
    df['truth_degree_ratio'] = df['truth_degree_mean'] / df['avg_degree'].replace(0, np.nan)
    print('  合并后: %d 行' % len(df))

    gfeats = [f for f in GLOBAL_FEATS if f in df.columns]
    mfeats = [f for f in MECH_FEATS if f in df.columns]

    # ---------- 0) 共线性诊断 ----------
    corr = df[gfeats + mfeats].corr(method='spearman')
    corr.to_csv(os.path.join(ANA, '特征相关矩阵.csv'), encoding='utf-8-sig')
    high = [(a, b, round(float(corr.loc[a, b]), 3))
            for i, a in enumerate(corr.index) for b in corr.columns[i + 1:]
            if abs(corr.loc[a, b]) > 0.85]
    print('  → 特征相关矩阵.csv   （|ρ|>0.85 的高共线对: %d 组）' % len(high))
    for a, b, v in high[:8]:
        print('       %-22s ~ %-22s ρ=%s' % (a, b, v))

    # ---------- 1) 分层 Spearman（全局特征）----------
    def layer_spearman(feats, fname):
        recs = []
        for m, ds in LAYERS:
            sub = df[(df.method == m) & (df.dataset == ds)]
            for f in feats:
                rho, p, n = spearman(sub[f], sub['norm_rank'])
                recs.append(dict(method=m, dataset=ds, feature=f, n=n, rho=rho, p=p))
        t = pd.DataFrame(recs)
        t['q'] = bh_fdr(t['p'].to_numpy())
        t['sig'] = [stars(p, q) for p, q in zip(t['p'], t['q'])]
        t.to_csv(os.path.join(ANA, fname), index=False, encoding='utf-8-sig')
        return t

    layer = layer_spearman(gfeats, '相关_分层.csv')
    print('  → 相关_分层.csv（%d 条）' % len(layer))

    # ---------- 2) 层内标准化后合并 ----------
    zdf = df.copy()
    grp = zdf.groupby(['method', 'dataset'])

    def zscore(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    zdf['norm_rank_z'] = grp['norm_rank'].transform(zscore)
    for f in gfeats + mfeats:
        zdf[f + '_z'] = grp[f].transform(zscore)

    recs2 = []
    for f in gfeats + mfeats:
        rho, p, n = spearman(zdf[f + '_z'], zdf['norm_rank_z'])
        recs2.append(dict(feature=f, group='全局结构' if f in gfeats else '真值局部',
                          n=n, rho=rho, p=p))
    comb = pd.DataFrame(recs2)
    comb['q'] = bh_fdr(comb['p'].to_numpy())
    comb['sig'] = [stars(p, q) for p, q in zip(comb['p'], comb['q'])]
    comb = comb.sort_values('rho')
    comb.to_csv(os.path.join(ANA, '相关_合并.csv'), index=False, encoding='utf-8-sig')
    print('  → 相关_合并.csv（层内标准化，n=1125）')

    # ---------- 3) 真值节点机制特征（分层）----------
    mech = layer_spearman([f for f in mfeats if df[f].notna().any()], '机制_真值节点.csv')
    print('  → 机制_真值节点.csv')

    # ---------- 4) PCMCI 为何失效：度分布均匀性 ----------
    rows = []
    for m, ds in LAYERS:
        sub = df[(df.method == m) & (df.dataset == ds)]
        rows.append(dict(
            method=m, dataset=ds, n=len(sub),
            mean_density=round(sub.edge_density.mean(), 4),
            mean_gini=round(sub.degree_gini.mean(), 4),
            mean_path=round(sub.avg_shortest_path.mean(), 3),
            mean_bidir=round(sub.bidirectional_ratio.mean(), 3),
            std_in_degree=round(sub.in_degree_std.mean(), 2),
            mean_truth_deg=round(sub.truth_degree_mean.mean(), 2),
            mean_norm_rank=round(sub.norm_rank.mean(), 4),
        ))
    pcm = pd.DataFrame(rows)
    pcm.to_csv(os.path.join(ANA, 'PCMCI_度分布对比.csv'), index=False, encoding='utf-8-sig')
    print('  → PCMCI_度分布对比.csv')

    # ---------- 5) 作图 ----------
    # 5.1 密度 vs 名次
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds in zip(axes, DATASETS):
        for m, color in zip(METHODS, ['#2E86AB', '#E67E22', '#C0392B']):
            s = df[(df.method == m) & (df.dataset == ds)]
            if s.empty:
                continue
            ax.scatter(s['edge_density'], s['norm_rank'], s=13, alpha=0.55,
                       label=m, color=color, edgecolors='none')
        ax.set_title('%s：边密度 vs 归一化名次' % ds, fontsize=12)
        ax.set_xlabel('edge density'); ax.set_ylabel('normalized rank (越小越好)')
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle('因果图边密度与定位性能（各 125 例）', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(ANA, '图_密度vs名次.png'), dpi=150); plt.close(fig)

    # 5.2 真值节点度比 vs 名次（强信号）
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds in zip(axes, DATASETS):
        for m, color in zip(METHODS, ['#2E86AB', '#E67E22', '#C0392B']):
            s = df[(df.method == m) & (df.dataset == ds)].dropna(subset=['truth_degree_ratio'])
            if s.empty:
                continue
            ax.scatter(s['truth_degree_ratio'], s['norm_rank'], s=13, alpha=0.55,
                       label='%s (ρ=%.2f)' % (m, spearman(s['truth_degree_ratio'], s['norm_rank'])[0]),
                       color=color, edgecolors='none')
        ax.set_title('%s：真值节点度比 vs 归一化名次' % ds, fontsize=12)
        ax.set_xlabel('truth_degree / avg_degree'); ax.set_ylabel('normalized rank')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('真值节点在图中的连接强度 vs 定位性能', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(ANA, '图_真值度比vs名次.png'), dpi=150); plt.close(fig)

    # 5.3 特征相关热力图
    fig, ax = plt.subplots(figsize=(11, 9))
    cm = corr.to_numpy()
    im = ax.imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if abs(cm[i, j]) >= 0.6 and i != j:
                ax.text(j, i, '%.1f' % cm[i, j], ha='center', va='center', fontsize=6,
                        color='white' if abs(cm[i, j]) > 0.8 else 'black')
    ax.set_title('特征间 Spearman 相关（|ρ|≥0.6 标注）', fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(os.path.join(ANA, '图_特征相关热力图.png'), dpi=150); plt.close(fig)

    # 5.4 分层 rho 条形图
    piv = layer.pivot_table(index='feature', columns=['method', 'dataset'], values='rho').reindex(gfeats)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(piv)); w = 0.9 / piv.shape[1]
    for k, col in enumerate(piv.columns):
        ax.barh(x + k * w, piv[col].to_numpy(), height=w, label='%s·%s' % (col[0], col[1][4:]))
    ax.set_yticks(x + 0.45); ax.set_yticklabels(piv.index, fontsize=9)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Spearman ρ（负 = 该特征越大，名次越靠前）')
    ax.set_title('全局结构特征与归一化名次的相关系数（分层）', fontsize=12)
    ax.legend(fontsize=7, ncol=3, loc='lower right'); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(ANA, '图_分层rho.png'), dpi=150); plt.close(fig)
    print('  → 4 张图')

    # ---------- 6) 控制台报告 ----------
    def show_layer(t, title):
        print('\n' + '=' * 106)
        print(title)
        print('=' * 106)
        print('%-24s %-8s %8s %9s %9s %6s   %s' % ('feature', 'dataset', 'n', 'rho', 'p', 'q', 'sig'))
        print('-' * 106)
        for f in t['feature'].unique():
            for m in METHODS:
                for ds in DATASETS:
                    r = t[(t.feature == f) & (t.method == m) & (t.dataset == ds)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    pv = r['p']
                    print('%-24s %-8s %8d %9.3f %9s %6s   %s' % (
                        f, '%s·%s' % (m[:4], ds[4:]), r['n'], r['rho'],
                        ('%.3g' % pv) if pv == pv else '-',
                        ('%.3g' % r['q']) if r['q'] == r['q'] else '-', r['sig']))
            print()

    show_layer(layer, '【表 1】全局结构特征 × 归一化名次（分层 Spearman；q 为 BH-FDR 校正后）')

    print('=' * 106)
    print('【表 2】层内标准化后合并（n=1125，已消除层间规模差异）')
    print('=' * 106)
    print('%-24s %-10s %8s %9s %9s %6s   %s' % ('feature', '组', 'n', 'rho', 'p', 'q', 'sig'))
    print('-' * 106)
    for _, r in comb.iterrows():
        print('%-24s %-10s %8d %9.3f %9s %6s   %s' % (
            r['feature'], r['group'], r['n'], r['rho'],
            ('%.3g' % r['p']) if r['p'] == r['p'] else '-',
            ('%.3g' % r['q']) if r['q'] == r['q'] else '-', r['sig']))

    show_layer(mech, '【表 3】真值节点局部特征 × 归一化名次（机制分析）')

    print('=' * 106)
    print('【表 4】为什么 PCMCI 失效：度分布的区分度')
    print('=' * 106)
    print('%-9s %-8s %10s %9s %8s %10s %11s %12s %13s' %
          ('method', 'dataset', 'density', 'gini', 'avgPath', 'bidir%', 'inDegStd',
           'truthDeg', 'normRank'))
    print('-' * 106)
    for _, r in pcm.iterrows():
        print('%-9s %-8s %10.4f %9.4f %8.2f %10.1f %11.2f %12.2f %13.4f' % (
            r['method'], r['dataset'], r['mean_density'], r['mean_gini'], r['mean_path'],
            r['mean_bidir'] * 100, r['std_in_degree'], r['mean_truth_deg'], r['mean_norm_rank']))

    print('\n✅ 全部产出已写入: %s' % ANA)


if __name__ == '__main__':
    main()
