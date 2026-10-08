# -*- coding: utf-8 -*-
"""
合并 Granger 分片结果 → 完整评估（输出格式与 run_granger.py 的 eval.csv 一致）

背景：RE1-SS / RE1-TT 单例太慢（239 s / 811 s），必须用 --shard 分片并行；
      但每个分片各自只覆盖案例子集，需合并后才是完整结果。

用法（WSL 中）:
  ~/RCAEval/env/bin/python /mnt/d/北交威/大创/结果/merge_granger_shards.py --k 1 --dataset RE1-SS
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'pc_pcmci_fixed'))
from common_eval import compute_rank_accuracy  # noqa: E402

FAULT_ORDER = ['cpu', 'mem', 'delay', 'loss', 'disk']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True, help='降采样档位（1/2/4）')
    ap.add_argument('--dataset', default='RE1-SS')
    ap.add_argument('--out-root', default=None, help='默认 $REPO/结果/降采样实验_全量')
    ap.add_argument('--n-shard', type=int, default=6)
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(REPO, '结果', '降采样实验_全量')
    base = os.path.join(out_root, 'ds%d' % args.k)

    cases = {}
    for i in range(1, args.n_shard + 1):
        d = os.path.join(base, '%s-granger-shard%d' % (args.dataset, i))
        if not os.path.isdir(d):
            print('  [WARN] 分片 %d 目录不存在，跳过' % i)
            continue
        for p in glob.glob(os.path.join(d, args.dataset, '*_*', '*.csv')):
            parts = p.replace('\\', '/').split('/')
            fname, svc_fault = parts[-1], parts[-2]
            case = fname[:-4]
            if not case.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            try:
                rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
            except Exception:
                continue
            if not rows or 'rank' not in rows[0] or 'node' not in rows[0]:
                continue
            rows.sort(key=lambda r: int(r['rank']))
            cases[(svc, fault, case)] = [r['node'] for r in rows]

    print('  合并到 %d 例' % len(cases))
    if not cases:
        print('❌ 未收集到任何案例，请检查分片目录是否存在')
        return

    recs = []
    for (svc, fault, case), nodes in sorted(cases.items()):
        r = compute_rank_accuracy(nodes, svc, fault)
        recs.append(dict(service=svc, fault=fault, case=case, n_nodes=len(nodes),
                         svc_rank=r['Svc_Rank'],
                         Svc_AC1=r['Svc-AC@1'], Svc_AC3=r['Svc-AC@3'], Svc_AC5=r['Svc-AC@5'],
                         Mtr_AC1=r['Mtr-AC@1'], Mtr_AC5=r['Mtr-AC@5']))

    merged_dir = os.path.join(base, '%s-granger' % args.dataset)
    os.makedirs(merged_dir, exist_ok=True)

    with open(os.path.join(merged_dir, 'summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    def agg(rs):
        n = len(rs)
        return dict(
            n_cases=n,
            svc_ac1=sum(r['Svc_AC1'] for r in rs) / n,
            svc_ac3=sum(r['Svc_AC3'] for r in rs) / n,
            svc_ac5=sum(r['Svc_AC5'] for r in rs) / n,
            svc_avg5=sum((r['Svc_AC1'] + r['Svc_AC3'] + r['Svc_AC5']) / 3.0 for r in rs) / n,
            mtr_ac1=sum(r['Mtr_AC1'] for r in rs) / n,
            mtr_avg5=sum(r['Mtr_AC5'] for r in rs) / n,
        )

    lines = [['fault_type', 'n_cases', 'svc_ac1', 'svc_ac3', 'svc_ac5', 'svc_avg5',
              'mtr_ac1', 'mtr_avg5']]
    for ft in FAULT_ORDER:
        rs = [r for r in recs if r['fault'] == ft]
        if rs:
            a = agg(rs)
            lines.append([ft] + [a[k] for k in
                                 ('n_cases', 'svc_ac1', 'svc_ac3', 'svc_ac5',
                                  'svc_avg5', 'mtr_ac1', 'mtr_avg5')])
    a = agg(recs)
    lines.append(['OVERALL'] + [a[k] for k in
                                ('n_cases', 'svc_ac1', 'svc_ac3', 'svc_ac5',
                                 'svc_avg5', 'mtr_ac1', 'mtr_avg5')])

    ev = os.path.join(merged_dir, 'eval.csv')
    with open(ev, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(lines)

    print('  → %s' % ev)
    for row in lines:
        print('    ' + '  '.join(
            ('%s' % x) if isinstance(x, str) or isinstance(x, int) else '%.4f' % x for x in row))


if __name__ == '__main__':
    main()
