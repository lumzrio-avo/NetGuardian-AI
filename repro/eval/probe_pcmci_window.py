# -*- coding: utf-8 -*-
"""
反推 PCMCI 在 RE1-TT 上实际使用的【窗口】（纯标准库，含完整 ffill/fillna 预处理）

流程对齐 run_pcmci.py：
  read_csv → replace(inf,-inf -> nan) → ffill() → fillna(0)
  → 删 time / 隐去 latency-50 → 删常量列(std==0) → 方差 Top-50
"""
import csv
import io
import json
import math
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.normpath(os.path.join(HERE, '..', 'dataset', 'RE1-TT.zip'))
TARGET = os.path.normpath(os.path.join(
    HERE, 'pcmci', 'pcmci结果', 'output_tt', 'RE1-TT',
    'ts-auth-service_cpu', '1_node_names.json'))
CASE = 'ts-auth-service_cpu/1'

target = set(json.load(open(TARGET, encoding='utf-8')))
z = zipfile.ZipFile(ZIP)
members = [n for n in z.namelist() if n.startswith('RE1-TT/%s/' % CASE)]
inject_time = int(z.read([n for n in members if n.endswith('inject_time.txt')][0])
                  .decode().strip().splitlines()[0])
data_entry = [n for n in members if n.endswith('/data.csv')][0]

with z.open(data_entry) as fh:
    reader = csv.reader(io.TextIOWrapper(fh, encoding='utf-8', errors='replace'))
    header = next(reader)
    raw = []
    for r in reader:
        row = []
        for x in r:
            try:
                v = float(x)
                if math.isinf(v):
                    v = float('nan')
            except ValueError:
                v = float('nan')
            row.append(v)
        raw.append(row)

tcol = header.index('time')
# --- ffill → fillna(0) ---
last = [0.0] * len(header)
for row in raw:
    for j in range(len(header)):
        if math.isnan(row[j]):
            row[j] = last[j]
        else:
            last[j] = row[j]

before = [i for i, r in enumerate(raw) if r[tcol] < inject_time]
print('原始 %d 行 × %d 列 | 故障前 %d 行' % (len(raw), len(header), len(before)))


def var(col):
    n = len(col)
    m = sum(col) / n
    return sum((x - m) ** 2 for x in col) / (n - 1)


KEEP = [j for j, c in enumerate(header)
        if c.lower() != 'time' and not c.endswith('_latency-50')]


def pick(idx, label):
    cols = []
    for j in KEEP:
        col = [raw[i][j] for i in idx]
        v = var(col)
        if v == 0.0:
            continue
        cols.append((v, header[j]))
    cols.sort(key=lambda x: -x[0])
    top = set(n for _, n in cols[:50])
    print('  [%-14s] 行=%-4d 删常量后 %-4d 列  Top-50 命中目标 %d / 50'
          % (label, len(idx), len(cols), len(top & target)))
    return top


A = pick(range(len(raw)), '全窗口')
B = pick(before[-480:], '正常窗480')
C = pick(before, '正常窗全部(480)')
E = pick(range(before[0], len(raw)), '故障点到末尾')

print('\n  [诊断] 两种窗口各自 Top-50 集合的重合度: %d / 50' % len(A & B))
print('\n=== 结论 ===')
for lbl, s, in [('全窗口', A), ('正常窗口', B), ('故障点到末尾', E)]:
    print('  %-12s %d / 50' % (lbl, len(s & target)))
