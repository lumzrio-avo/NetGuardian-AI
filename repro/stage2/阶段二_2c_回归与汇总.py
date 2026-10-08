# -*- coding: utf-8 -*-
"""
阶段二 · 第 2 步(c)：补齐【方案承诺但未执行】的两项

方案原文（`工作细化文档/阶段二-第2步-拓扑特征分析方案.md`）：
  §5「多元：标准化线性回归（主表）」   → 本脚本补做
  §6「`相关_汇总.csv`（显著层数、方向一致性、平均 ρ）」 → 本脚本补做

【为什么补】
  不补的话，只能说"特征与名次**关联**"，不能说"控制其他特征后的**独立**贡献"。
  补完之后论文措辞可以更强一档。

【方法】
  因变量  norm_rank_z = 层内(方法×数据集) z-score 后的归一化名次
  自变量  13 个全局结构特征（层内 z-score），另单独跑 3 个真值节点机制特征
  共线性 任务书列了 33 组 |ρ|>0.85 → 先做**贪心修剪**（每组只留与因变量相关更强的一个），
          再上 OLS，这样既有 p 值又不会 VIF 爆炸
  估计   手写 OLS（本机无 sklearn/statsmodels，且公式足够简单）
          标准化系数 β、稳健起见同时报 VIF 与 R²
  对照   同时跑「单变量 OLS」，与 Spearman 相互印证

产出（写入 阶段二分析/）：
  回归_跨层.csv      多元模型（修剪后）的标准化系数
  回归_单变量.csv    单变量模型
  相关_汇总.csv      每特征：显著层数 / 方向一致性 / 平均 ρ / 合并 ρ

用法: python 结果/阶段二_2c_回归与汇总.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


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

GLOBAL_FEATS = ['n_nodes', 'n_edges', 'edge_density', 'avg_degree',
                'in_degree_max', 'in_degree_std', 'out_degree_max', 'out_degree_std',
                'degree_gini', 'avg_shortest_path', 'largest_scc_ratio',
                'bidirectional_ratio', 'transitivity']
MECH_FEATS = ['truth_degree_mean', 'truth_degree_max', 'truth_degree_ratio']
CORR_THRESH = 0.85


# ============================================================
# 手写 OLS（标准化数据上运行 ⇒ 系数即可解释为标准化 β）
# ============================================================
def ols(X, y):
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else np.nan
    return beta, se, t, pvals, r2


def vif_all(X):
    out = []
    for j in range(X.shape[1]):
        Xj = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(Xj)), Xj])
        _, _, _, _, r2 = ols(A, X[:, j])
        out.append(np.inf if r2 >= 1 else 1.0 / (1.0 - r2))
    return out


def stars(p):
    if p != p:
        return 'n/a'
    for th, s in ((0.001, '***'), (0.01, '**'), (0.05, '*')):
        if p < th:
            return s
    return 'ns'


def prune_collinear(feats, corr, ycorr, thresh=CORR_THRESH):
    """贪心：按 |与因变量相关| 降序，逐个保留不与已留特征高共线的"""
    keep = []
    for f in sorted(feats, key=lambda x: -abs(ycorr.get(x, 0.0))):
        if all(abs(corr.loc[f, k]) <= thresh for k in keep):
            keep.append(f)
    return keep


# ============================================================
def main():
    print('=' * 96)
    print('阶段二 · 第 2 步(c)：补齐多元回归 + 相关汇总')
    print('=' * 96)

    feat = pd.read_csv(os.path.join(ANA, '图特征.csv'))
    det = pd.read_csv(os.path.join(ANA, '逐例明细.csv'))
    key = ['method', 'dataset', 'service', 'fault', 'case']
    df = feat.merge(det[key + ['svc_capped']], on=key, how='inner')
    df['norm_rank'] = df['svc_capped'] / (df['n_nodes'] + 1.0)
    df['truth_degree_ratio'] = df['truth_degree_mean'] / df['avg_degree'].replace(0, np.nan)
    print('  合并后 %d 行' % len(df))

    gfeats = [f for f in GLOBAL_FEATS if f in df.columns]
    mfeats = [f for f in MECH_FEATS if f in df.columns]

    # ---------- 0) 层内标准化 ----------
    grp = df.groupby(['method', 'dataset'])

    def z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    zdf = df.copy()
    zdf['y'] = grp['norm_rank'].transform(z)
    for f in gfeats + mfeats:
        zdf[f + '_z'] = grp[f].transform(z)

    corr = zdf[[f + '_z' for f in gfeats + mfeats]].corr(method='spearman')
    corr.index = [c[:-2] for c in corr.index]
    corr.columns = [c[:-2] for c in corr.columns]
    ycorr = {f: float(zdf[[f + '_z', 'y']].corr(method='spearman').iloc[0, 1])
             for f in gfeats + mfeats}

    # ---------- 1) 共线性修剪 ----------
    kept = prune_collinear(gfeats, corr, ycorr)
    dropped = [f for f in gfeats if f not in kept]
    print('\n  [共线性修剪] |ρ|>%.2f 视为共线，13 个全局特征 → 保留 %d 个' % (CORR_THRESH, len(kept)))
    print('    保留: %s' % ', '.join(kept))
    print('    剔除: %s' % (', '.join(dropped) if dropped else '（无）'))

    def run(feats, tag, dropna=True):
        sub = zdf.dropna(subset=[f + '_z' for f in feats] + ['y'])
        X = sub[[f + '_z' for f in feats]].to_numpy(float)
        y = sub['y'].to_numpy(float)
        # 标准化（层内 z 后仍再标准化一次，保证不同模型 β 可比）
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        y = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
        beta, se, t, p, r2 = ols(X, y)
        v = vif_all(X)
        return [dict(model=tag, feature=f, n=len(sub), beta=float(b), se=float(s),
                     t=float(tt), p=float(pp), vif=float(vv), r2=float(r2))
                for f, b, s, tt, pp, vv in zip(feats, beta, se, t, p, v)]

    multi = run(kept, '多元(修剪后)')
    single = [r for f in gfeats for r in run([f], '单变量')]
    # 机制特征（truth_degree_mean/max/ratio）同源、彼此高度共线，
    # 放进同一个多元模型会让系数符号互相翻转 → 一律**逐个单独建模**
    mech_single = [r for f in mfeats for r in run([f], '机制(单变量)')]

    # ---------- 2) 落盘：回归 ----------
    def dump(rows, name):
        cols = ['model', 'feature', 'n', 'beta', 'se', 't', 'p', 'vif', 'r2', 'sig']
        p_ = os.path.join(ANA, name)
        with open(p_, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                r = dict(r)
                r['sig'] = stars(r['p'])
                w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})
        print('  → %s（%d 行）' % (name, len(rows)))

    dump(multi + mech_single, '回归_跨层.csv')
    dump(single + mech_single, '回归_单变量.csv')

    # ---------- 3) 相关_汇总.csv ----------
    lay = pd.read_csv(os.path.join(ANA, '相关_分层.csv'))
    # 机制特征的分层结果在另一个文件里（2b 脚本分开产出），必须合并进来，否则汇总表会漏掉它们
    mech_lay = pd.read_csv(os.path.join(ANA, '机制_真值节点.csv'))
    lay = pd.concat([lay, mech_lay], ignore_index=True)
    comb = pd.read_csv(os.path.join(ANA, '相关_合并.csv'))
    comb_map = {r['feature']: r for _, r in comb.iterrows()}

    rows = []
    for f in gfeats + mfeats:
        sub = lay[lay.feature == f]
        if sub.empty:
            continue
        sig = sub[sub.q < 0.05]
        dirs = np.sign(sig['rho'].to_numpy()) if len(sig) else np.array([])
        c = comb_map.get(f)
        rows.append(dict(
            feature=f,
            group='全局结构' if f in gfeats else '真值局部',
            n_layers=len(sub),
            n_sig=len(sig),
            dir_consistent='是' if len(dirs) and (np.abs(dirs.sum()) == len(dirs)) else
                           ('否' if len(dirs) else '—'),
            mean_rho_layer=round(float(sub['rho'].mean()), 4),
            min_rho=round(float(sub['rho'].min()), 4),
            max_rho=round(float(sub['rho'].max()), 4),
            merged_rho=round(float(c['rho']), 4) if c is not None else np.nan,
            merged_p=float(c['p']) if c is not None else np.nan,
            merged_q=float(c['q']) if c is not None else np.nan,
            merged_sig=stars(float(c['q'])) if c is not None else 'n/a',
            in_multivar='是' if f in kept else ('否(共线剔除)' if f in gfeats else '—'),
        ))
    cols = ['feature', 'group', 'n_layers', 'n_sig', 'dir_consistent', 'mean_rho_layer',
            'min_rho', 'max_rho', 'merged_rho', 'merged_p', 'merged_q', 'merged_sig', 'in_multivar']
    p_ = os.path.join(ANA, '相关_汇总.csv')
    with open(p_, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('  → 相关_汇总.csv（%d 行）' % len(rows))

    # ---------- 4) 报告 ----------
    print('\n' + '=' * 96)
    print('【表 1】★ 多元回归（共线性修剪后，控制层内规模差异）—— 因变量：归一化名次的层内 z 值')
    print('=' * 96)
    print('%-24s %8s %10s %10s %8s %8s' % ('feature', 'n', 'beta', 'p', 'VIF', 'sig'))
    print('-' * 96)
    for r in multi:
        print('%-24s %8d %10.4f %10.3g %8.2f %8s' % (
            r['feature'], r['n'], r['beta'], r['p'], r['vif'], stars(r['p'])))
    print('  R² = %.4f' % (multi[0]['r2'] if multi else float('nan')))

    print('\n' + '=' * 96)
    print('【表 2】机制特征（真值节点）—— 逐个单独建模（彼此共线，不可同时入模）')
    print('=' * 96)
    print('%-24s %8s %10s %10s %10s' % ('feature', 'n', 'beta', 'p', 'sig'))
    print('-' * 96)
    for r in mech_single:
        print('%-24s %8d %10.4f %10.3g %10s' % (r['feature'], r['n'], r['beta'], r['p'], stars(r['p'])))

    print('\n' + '=' * 96)
    print('【表 3】相关汇总（每特征：显著层数 / 方向一致性 / 合并 ρ）')
    print('=' * 96)
    print('%-24s %-8s %6s %6s %6s %10s %11s %8s %s' %
          ('feature', '组', '层数', '显著', '同向', '层内均ρ', '合并ρ', 'q', '多元'))
    print('-' * 96)
    for r in sorted(rows, key=lambda x: (x['group'], x['merged_rho'])):
        print('%-24s %-8s %6d %6d %6s %10.3f %11.3f %8.2g %s' % (
            r['feature'], r['group'], r['n_layers'], r['n_sig'], r['dir_consistent'],
            r['mean_rho_layer'], r['merged_rho'], r['merged_q'], r['in_multivar']))

    print('\n✅ 完成')


if __name__ == '__main__':
    main()
