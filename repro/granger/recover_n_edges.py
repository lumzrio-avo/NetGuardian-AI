"""
快速重跑几个案例，只提取 n_edges / edge_density
运行环境: WSL (需要 RCAEval + 数据)
"""
import sys, os
sys.path.insert(0, '/home/lumzrio/RCAEval')

import pandas as pd
import numpy as np
from RCAEval.graph_construction.granger import granger

# 代表案例: OB每种故障1个 + SS 2个 + TT 1个
CASES = [
    ('RE1-OB', 'adservice', 'delay', 1),
    ('RE1-OB', 'adservice', 'disk', 1),
    ('RE1-OB', 'adservice', 'loss', 1),
    ('RE1-OB', 'checkoutservice', 'cpu', 4),  # 仅有的2个cpu之一
    ('RE1-SS', 'carts', 'cpu', 1),
    ('RE1-SS', 'carts', 'delay', 1),
    ('RE1-SS', 'carts', 'mem', 1),
    ('RE1-TT', 'ts-auth-service', 'cpu', 1),
]

DATA_ROOT = '/home/lumzrio/RCAEval/data/RE1'

print(f"{'dataset':<8} {'service':<25} {'fault':<8} {'n_nodes':>7} {'n_edges':>7} {'density':>8}")
print('-' * 80)

for dataset, service, fault, case_id in CASES:
    # 读数据
    csv_path = f'{DATA_ROOT}/{dataset}/{service}_{fault}/{case_id}/data.csv'
    inject_path = f'{DATA_ROOT}/{dataset}/{service}_{fault}/{case_id}/inject_time.txt'
    
    if not os.path.exists(csv_path):
        print(f'  SKIP: {csv_path} not found')
        continue
    
    df = pd.read_csv(csv_path)
    
    # 预处理 (对齐 main.py)
    df = df.loc[:, ~df.columns.str.endswith('_latency-50')]
    # SS: 丢弃 lat_50/lat_99（对齐 run_granger.py 的 load_and_preprocess）
    for col in list(df.columns):
        if col.endswith('lat_50') or col.endswith('lat_99'):
            df = df.drop(columns=[col])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    df = df.fillna(0)
    
    # TT: 只保留 ts- 列
    if dataset == 'RE1-TT':
        ts_cols = [c for c in df.columns if c.startswith('ts-')]
        if ts_cols:
            time_col = df['time']
            df = df[ts_cols]
            df['time'] = time_col
    
    # inject_time 切分
    with open(inject_path) as f:
        inject_time = int(f.read().strip())
    
    normal = df[df['time'] < inject_time].tail(600)
    anomal = df[df['time'] >= inject_time].head(600)
    df = pd.concat([normal, anomal], ignore_index=True)
    
    # 重命名 latency-90 → latency (OB only)
    for c in list(df.columns):
        if c.endswith('_latency-90'):
            df = df.rename(columns={c: c.replace('_latency-90', '_latency')})
        if c.endswith('lat_90'):
            df = df.rename(columns={c: c.replace('lat_90', '_latency')})
    
    # 复制官方 preprocess
    from RCAEval.io.time_series import preprocess
    df = preprocess(df, dataset=dataset, dk_select_useful=False)
    
    # Granger
    adj = granger(df)
    n = adj.shape[0]
    n_edges = int(adj.sum())
    density = round(n_edges / max(n * (n - 1), 1), 4)
    
    print(f'{dataset:<8} {service:<25} {fault:<8} {n:>7} {n_edges:>7} {density:>8}')

print('-' * 80)
print('Done!')
