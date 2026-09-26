# -*- coding: utf-8 -*-
"""
Granger 结果全面分析脚本
生成: granger_eval_clean.csv + granger_summary.md + 图表
"""
import os, glob, platform
import pandas as pd
import numpy as np


def to_local(p: str) -> str:
    """Windows 路径在 WSL/Linux 下自动转为 /mnt/<drive>/...，便于两边都能跑。"""
    if platform.system() != 'Windows' and len(p) > 2 and p[1] == ':':
        return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')
    return p


BASE = to_local(r'D:\北交威\大创\结果\Granger\output_granger_merged')
OUT_DIR = to_local(r'D:\北交威\大创\结果\Granger\结果分析')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取合并数据
# ============================================================
summary = pd.read_csv(os.path.join(BASE, 'summary.csv'))
eval_df = pd.read_csv(os.path.join(BASE, 'eval.csv'))

# 清洗 summary
summary['rank_of_truth'] = pd.to_numeric(summary['rank_of_truth'], errors='coerce')
summary['rank_of_truth_metric'] = pd.to_numeric(summary['rank_of_truth_metric'], errors='coerce')

print(f"总案例数: {len(summary)}")
print(f"数据集分布: {summary['dataset'].value_counts().to_dict()}")
print(f"故障类型分布: {summary['fault_type'].value_counts().to_dict()}")

# ============================================================
# 2. 构建 granger_eval_clean.csv（汇总表）
# ============================================================

# 从 summary 计算每个案例的 Svc 命中情况
def svc_hit(row, k):
    if pd.isna(row['rank_of_truth']):
        return 0
    return 1 if row['rank_of_truth'] <= k else 0

def mtr_hit(row, k):
    if pd.isna(row['rank_of_truth_metric']):
        return 0
    return 1 if row['rank_of_truth_metric'] <= k else 0

for k in [1, 2, 3, 4, 5]:
    summary[f'svc_hit_{k}'] = summary.apply(lambda r: svc_hit(r, k), axis=1)
    summary[f'mtr_hit_{k}'] = summary.apply(lambda r: mtr_hit(r, k), axis=1)

# 按数据集+故障类型聚合
agg_rows = []
for (ds, ft), group in summary.groupby(['dataset', 'fault_type']):
    n = len(group)
    svc_ac1 = group['svc_hit_1'].mean()
    svc_ac3 = group['svc_hit_3'].mean()
    svc_ac5 = group['svc_hit_5'].mean()
    # 官方 Avg@5 = sum(AC@1..AC@5)/5（原为 mean(AC@1,AC@3,AC@5)，已修正）
    svc_avg5 = sum(group[f'svc_hit_{k}'].mean() for k in range(1, 6)) / 5

    mtr_ac1 = group['mtr_hit_1'].mean()
    mtr_ac3 = group['mtr_hit_3'].mean()
    mtr_ac5 = group['mtr_hit_5'].mean()
    mtr_avg5 = sum(group[f'mtr_hit_{k}'].mean() for k in range(1, 6)) / 5

    # rank 分布
    ranks = group['rank_of_truth'].dropna()
    rank_mean = ranks.mean() if len(ranks) > 0 else np.nan
    rank_median = ranks.median() if len(ranks) > 0 else np.nan
    rank_std = ranks.std() if len(ranks) > 0 else np.nan

    # Mtr 可用性标注
    mtr_note = ''
    if ds in ['RE1-SS', 'RE1-TT']:
        mtr_note = '数据格式限制：列名非 {service}_{metric} 格式，Mtr 无法计算'
    elif ft == 'disk' and ds == 'RE1-OB':
        mtr_note = 'RE1-OB 无 diskio 列，Mtr 无法计算'
    elif ft in ['cpu', 'mem'] and ds == 'RE1-OB':
        mtr_note = '案例数少/排名靠后，Mtr 为 0'

    agg_rows.append({
        'dataset': ds,
        'fault_type': ft,
        'n_cases': n,
        'svc_ac1': round(svc_ac1, 4),
        'svc_ac3': round(svc_ac3, 4),
        'svc_ac5': round(svc_ac5, 4),
        'svc_avg5': round(svc_avg5, 4),
        'mtr_ac1': round(mtr_ac1, 4),
        'mtr_ac3': round(mtr_ac3, 4),
        'mtr_ac5': round(mtr_ac5, 4),
        'mtr_avg5': round(mtr_avg5, 4),
        'rank_mean': round(rank_mean, 2) if not np.isnan(rank_mean) else '',
        'rank_median': round(rank_median, 1) if not np.isnan(rank_median) else '',
        'rank_std': round(rank_std, 2) if not np.isnan(rank_std) else '',
        'top1_hit_rate': round(group['svc_hit_1'].mean(), 4),
        'top3_hit_rate': round(group['svc_hit_3'].mean(), 4),
        'top5_hit_rate': round(group['svc_hit_5'].mean(), 4),
        'mtr_note': mtr_note,
    })

clean_df = pd.DataFrame(agg_rows)

# 添加数据集 OVERALL 行
for ds in ['RE1-OB', 'RE1-SS', 'RE1-TT']:
    ds_group = summary[summary['dataset'] == ds]
    n = len(ds_group)
    svc_ac1 = ds_group['svc_hit_1'].mean()
    svc_ac3 = ds_group['svc_hit_3'].mean()
    svc_ac5 = ds_group['svc_hit_5'].mean()
    # 官方 Avg@5 = sum(AC@1..AC@5)/5
    svc_avg5 = sum(ds_group[f'svc_hit_{k}'].mean() for k in range(1, 6)) / 5

    mtr_ac1 = ds_group['mtr_hit_1'].mean()
    mtr_ac3 = ds_group['mtr_hit_3'].mean()
    mtr_ac5 = ds_group['mtr_hit_5'].mean()
    mtr_avg5 = sum(ds_group[f'mtr_hit_{k}'].mean() for k in range(1, 6)) / 5

    ranks = ds_group['rank_of_truth'].dropna()
    mtr_note = 'SS/TT 列名非标准格式，Mtr 无法计算' if ds in ['RE1-SS', 'RE1-TT'] else ''

    clean_df = pd.concat([clean_df, pd.DataFrame([{
        'dataset': ds,
        'fault_type': 'OVERALL',
        'n_cases': n,
        'svc_ac1': round(svc_ac1, 4),
        'svc_ac3': round(svc_ac3, 4),
        'svc_ac5': round(svc_ac5, 4),
        'svc_avg5': round(svc_avg5, 4),
        'mtr_ac1': round(mtr_ac1, 4),
        'mtr_ac3': round(mtr_ac3, 4),
        'mtr_ac5': round(mtr_ac5, 4),
        'mtr_avg5': round(mtr_avg5, 4),
        'rank_mean': round(ranks.mean(), 2),
        'rank_median': round(ranks.median(), 1),
        'rank_std': round(ranks.std(), 2),
        'top1_hit_rate': round(ds_group['svc_hit_1'].mean(), 4),
        'top3_hit_rate': round(ds_group['svc_hit_3'].mean(), 4),
        'top5_hit_rate': round(ds_group['svc_hit_5'].mean(), 4),
        'mtr_note': mtr_note,
    }])], ignore_index=True)

# 添加全局 OVERALL
all_group = summary
n = len(all_group)
svc_ac1 = all_group['svc_hit_1'].mean()
svc_ac3 = all_group['svc_hit_3'].mean()
svc_ac5 = all_group['svc_hit_5'].mean()
# 官方 Avg@5 = sum(AC@1..AC@5)/5
svc_avg5 = sum(all_group[f'svc_hit_{k}'].mean() for k in range(1, 6)) / 5
mtr_ac1 = all_group['mtr_hit_1'].mean()
mtr_ac3 = all_group['mtr_hit_3'].mean()
mtr_ac5 = all_group['mtr_hit_5'].mean()
mtr_avg5 = sum(all_group[f'mtr_hit_{k}'].mean() for k in range(1, 6)) / 5
ranks = all_group['rank_of_truth'].dropna()

clean_df = pd.concat([clean_df, pd.DataFrame([{
    'dataset': 'ALL',
    'fault_type': 'OVERALL',
    'n_cases': n,
    'svc_ac1': round(svc_ac1, 4),
    'svc_ac3': round(svc_ac3, 4),
    'svc_ac5': round(svc_ac5, 4),
    'svc_avg5': round(svc_avg5, 4),
    'mtr_ac1': round(mtr_ac1, 4),
    'mtr_ac3': round(mtr_ac3, 4),
    'mtr_ac5': round(mtr_ac5, 4),
    'mtr_avg5': round(mtr_avg5, 4),
    'rank_mean': round(ranks.mean(), 2),
    'rank_median': round(ranks.median(), 1),
    'rank_std': round(ranks.std(), 2),
    'top1_hit_rate': round(all_group['svc_hit_1'].mean(), 4),
    'top3_hit_rate': round(all_group['svc_hit_3'].mean(), 4),
    'top5_hit_rate': round(all_group['svc_hit_5'].mean(), 4),
    'mtr_note': 'SS/TT Mtr 因数据格式限制无法计算',
}])], ignore_index=True)

clean_df.to_csv(os.path.join(OUT_DIR, 'granger_eval_clean.csv'), index=False)
print(f"\n✅ granger_eval_clean.csv 已保存: {len(clean_df)} 行")

# ============================================================
# 3. 生成图表
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1: 各故障类型 Svc Avg@5 对比（分数据集）
fig, ax = plt.subplots(figsize=(10, 5))
faults = ['cpu', 'mem', 'delay', 'loss', 'disk']
x = np.arange(len(faults))
width = 0.25

for i, ds in enumerate(['RE1-OB', 'RE1-SS', 'RE1-TT']):
    vals = []
    for ft in faults:
        row = clean_df[(clean_df['dataset'] == ds) & (clean_df['fault_type'] == ft)]
        vals.append(row['svc_avg5'].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=ds)

ax.set_xlabel('故障类型')
ax.set_ylabel('Svc Avg@5')
ax.set_title('Granger+PageRank: 各故障类型在不同数据集上的 Svc Avg@5')
ax.set_xticks(x + width)
ax.set_xticklabels([f.upper() for f in faults])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_fault_type_svc_avg5.png'), dpi=150)
print("✅ fig1_fault_type_svc_avg5.png")

# 图2: 数据集总体 Svc vs Mtr
fig, ax = plt.subplots(figsize=(8, 5))
datasets = ['RE1-OB', 'RE1-SS', 'RE1-TT', 'ALL']
svc_vals = [clean_df[(clean_df['dataset'] == ds) & (clean_df['fault_type'] == 'OVERALL')]['svc_avg5'].values[0] for ds in datasets]
mtr_vals = [clean_df[(clean_df['dataset'] == ds) & (clean_df['fault_type'] == 'OVERALL')]['mtr_avg5'].values[0] for ds in datasets]

x = np.arange(len(datasets))
ax.bar(x - 0.2, svc_vals, 0.35, label='Svc Avg@5')
ax.bar(x + 0.2, mtr_vals, 0.35, label='Mtr Avg@5')
ax.set_xlabel('数据集')
ax.set_ylabel('Avg@5')
ax.set_title('Granger+PageRank: 服务级 vs 指标级定位')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_svc_vs_mtr.png'), dpi=150)
print("✅ fig2_svc_vs_mtr.png")

# 图3: rank_of_truth 分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ds in enumerate(['RE1-OB', 'RE1-SS', 'RE1-TT']):
    ranks = summary[summary['dataset'] == ds]['rank_of_truth'].dropna()
    axes[i].hist(ranks, bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{ds} (n={len(ranks)})')
    axes[i].set_xlabel('rank_of_truth')
    axes[i].set_ylabel('案例数')
    axes[i].axvline(x=5, color='red', linestyle='--', label='Top-5 线')
    axes[i].legend()
plt.suptitle('Granger+PageRank: 根因排名分布')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_rank_distribution.png'), dpi=150)
print("✅ fig3_rank_distribution.png")

# 图4: n_nodes 分布（图规模代理）
fig, ax = plt.subplots(figsize=(8, 5))
for ds in ['RE1-OB', 'RE1-SS', 'RE1-TT']:
    nodes = summary[summary['dataset'] == ds]['n_nodes']
    ax.hist(nodes, bins=15, alpha=0.5, label=ds)
ax.set_xlabel('因果图节点数 (n_nodes)')
ax.set_ylabel('案例数')
ax.set_title('Granger+PageRank: 因果图规模分布')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_graph_size.png'), dpi=150)
print("✅ fig4_graph_size.png")

print(f"\n所有输出已保存到: {OUT_DIR}")
