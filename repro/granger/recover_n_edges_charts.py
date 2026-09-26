"""补充 n_edges 数据到 Granger 分析报告"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = r'D:\北交威\大创\Granger结果\granger_analysis'

# 用户跑出的数据
data = [
    ('RE1-OB', 'adservice', 'delay', 48, 378, 0.1676),
    ('RE1-OB', 'adservice', 'disk',  47, 447, 0.2068),
    ('RE1-OB', 'adservice', 'loss',  50, 365, 0.1490),
    ('RE1-OB', 'checkoutservice', 'cpu', 51, 953, 0.3737),
    ('RE1-SS', 'carts', 'cpu',  438, 14872, 0.0777),
    ('RE1-SS', 'carts', 'delay', 438, 21745, 0.1136),
    ('RE1-SS', 'carts', 'mem',  438, 19869, 0.1038),
    ('RE1-TT', 'ts-auth-service', 'cpu', 1169, 107441, 0.0787),
]
edge_df = pd.DataFrame(data, columns=['dataset','service','fault','n_nodes','n_edges','density'])
edge_df.to_csv(f'{OUT_DIR}/granger_n_edges.csv', index=False)
print('✅ granger_n_edges.csv 已保存')

# 图5: 边数 vs 节点数
fig, ax = plt.subplots(figsize=(8, 5))
for ds, group in edge_df.groupby('dataset'):
    ax.scatter(group['n_nodes'], group['n_edges'], s=200, label=ds)
    for _, row in group.iterrows():
        ax.annotate(row['fault'], (row['n_nodes'], row['n_edges']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('因果图节点数 (n_nodes)')
ax.set_ylabel('因果图边数 (n_edges)')
ax.set_title('Granger 因果图规模：节点数 vs 边数')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_nodes_vs_edges.png', dpi=150)
print('✅ fig5_nodes_vs_edges.png')

# 图6: 边密度对比
fig, ax = plt.subplots(figsize=(9, 5))
fault_order = ['cpu','delay','mem','loss','disk']
x = np.arange(len(fault_order))
width = 0.27
for i, ds in enumerate(['RE1-OB','RE1-SS','RE1-TT']):
    sub = edge_df[edge_df['dataset']==ds]
    vals = []
    for ft in fault_order:
        row = sub[sub['fault']==ft]
        vals.append(row['density'].values[0] if len(row)>0 else 0)
    ax.bar(x + i*width, vals, width, label=ds)
ax.set_xlabel('故障类型')
ax.set_ylabel('边密度 (edge_density)')
ax.set_title('Granger 因果图：边密度对比')
ax.set_xticks(x + width)
ax.set_xticklabels([f.upper() for f in fault_order])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_edge_density.png', dpi=150)
print('✅ fig6_edge_density.png')

# 图7: 综合诊断图（边密度 vs 准确率）
# 关联到之前的 clean_df
clean_df = pd.read_csv(f'{OUT_DIR}/granger_eval_clean.csv')
merged = clean_df.merge(
    edge_df.groupby('dataset').agg(
        avg_nodes=('n_nodes','mean'),
        avg_edges=('n_edges','mean'),
        avg_density=('density','mean'),
    ).reset_index(),
    on='dataset', how='left'
)
merged_overall = merged[merged['fault_type']=='OVERALL']

fig, ax = plt.subplots(figsize=(8, 5))
for _, row in merged_overall.iterrows():
    ax.scatter(row['avg_density']*100, row['svc_avg5']*100, s=300)
    ax.annotate(row['dataset'], (row['avg_density']*100, row['svc_avg5']*100),
                xytext=(10, 5), textcoords='offset points', fontsize=11)
ax.set_xlabel('平均边密度 (%)')
ax.set_ylabel('Svc Avg@5 (%)')
ax.set_title('Granger 性能 vs 因果图边密度')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_density_vs_perf.png', dpi=150)
print('✅ fig7_density_vs_perf.png')

print('\n新增 3 张图 + 1 个 n_edges 数据 CSV 已保存到', OUT_DIR)