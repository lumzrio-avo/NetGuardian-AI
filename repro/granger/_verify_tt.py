# -*- coding: utf-8 -*-
"""TT 分片结果完整性校验（纯标准库，Windows 可直接跑）"""
import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.path.join(BASE, 'output_granger_TT_shards')
os.chdir(SHARD)

adjs = sorted(glob.glob('shard*/RE1-TT/**/*_adj.npy', recursive=True))
bad, ok, ns = [], 0, []
seen = set()
for adj in adjs:
    d = os.path.dirname(adj)
    base = os.path.basename(adj)[:-len('_adj.npy')]
    nodep = os.path.join(d, base + '_nodes.json')
    csvp = os.path.join(d, base + '.csv')
    if not os.path.exists(nodep):
        bad.append((adj, '缺 nodes.json')); continue
    n = len(json.load(open(nodep, encoding='utf-8')))
    if os.path.getsize(adj) != n * n * 8 + 128:
        bad.append((adj, '大小不符 n=%d' % n)); continue
    if not os.path.exists(csvp):
        bad.append((adj, '缺 csv')); continue
    if sum(1 for _ in open(csvp, encoding='utf-8-sig')) - 1 != n:
        bad.append((adj, 'csv 行数 != n')); continue
    ok += 1
    ns.append(n)
    seen.add((os.path.basename(d), base))

print('完整性: %d / %d 通过' % (ok, len(adjs)))
for a, w in bad[:10]:
    print('  X', a, w)
if ns:
    print('n_nodes: %d ~ %d  均值 %.1f' % (min(ns), max(ns), sum(ns) / len(ns)))

# 案例覆盖
exp = set()
for dd in glob.glob('shard*/RE1-TT/*/'):
    svc = os.path.basename(dd.rstrip('/').rstrip('\\'))
    for f in os.listdir(dd):
        if f.endswith('.csv'):
            exp.add((svc, f[:-4]))
print('案例覆盖: %d  期望 %d  缺 %d  多 %d'
      % (len(seen), len(exp), len(exp - seen), len(seen - exp)))
