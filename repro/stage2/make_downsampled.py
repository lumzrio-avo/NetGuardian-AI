# -*- coding: utf-8 -*-
"""
阶段二 · 第 3 步(b) Layer 2 前置 —— 生成「降采样」数据

【用途】
  模拟「监控采集频率下降」的场景，供因果发现重跑。
  做法：在原数据上每 k 个时间点取 1 点（k=2 → 采样率减半；k=4 → 降至 1/4）。

【⚠️ 关键：窗口参数必须同步缩小 k 倍】
  三个 runner 的窗口都是按【点数】取的（隐含 1Hz 假设）：
      Granger : run_granger.py:230   points = window_length_min*60//2
      PC/PCMCI: run_pcmci.py:146     同上
  降采样后点数变少 ⇒ 若窗口参数不变，窗口覆盖的【物理时长】会翻 k 倍，实验就不干净。
      ⇒ k=2 时用 --length 10 / --window 10；k=4 时用 5

【抽样】
  每个 {service}_{fault} 目录抽 N 例（默认 6）⇒ 每数据集 5 故障 × 6 = 30 例
  三个数据集共 90 例 × 2 档 = 180 次因果发现

【用法（在 WSL 中执行）】
    # 1) 先探测数据在哪（不写任何文件）
    python make_downsampled.py --src ~/RCAEval/data/RE1 --dst ~/RCAEval/data_ds --probe-only

    # 2) 确认无误后生成（两档各跑一次）
    python make_downsampled.py --src ~/RCAEval/data/RE1 --dst ~/RCAEval/data_ds --k 2
    python make_downsampled.py --src ~/RCAEval/data/RE1 --dst ~/RCAEval/data_ds --k 4
"""
import argparse
import os
import random
import shutil
from collections import defaultdict

import pandas as pd

DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']
SEED = 20261001


def find_cases(src):
    """返回 [(dataset, svc_fault, case, data_csv_path)]"""
    out = []
    for ds in DATASETS:
        d = os.path.join(src, ds)
        if not os.path.isdir(d):
            continue
        for sf in sorted(os.listdir(d)):
            sd = os.path.join(d, sf)
            if not os.path.isdir(sd):
                continue
            for case in sorted(os.listdir(sd)):
                if not case.isdigit():
                    continue
                p = os.path.join(sd, case, 'data.csv')
                if os.path.exists(p):
                    out.append((ds, sf, case, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='原始数据根（其下应为 RE1-OB / RE1-SS / RE1-TT）')
    ap.add_argument('--dst', required=True, help='输出根（每个 k 用一个独立目录，避免互相覆盖）')
    ap.add_argument('--k', type=int, default=2, help='降采样倍数（每 k 个点取 1 个）')
    ap.add_argument('--per-fault', type=int, default=6, help='每个 {svc}_{fault} 抽几例')
    ap.add_argument('--probe-only', action='store_true', help='只探测数据位置，不生成文件')
    args = ap.parse_args()

    cases = find_cases(args.src)
    if not cases:
        print('❌ 在 %s 下没找到 data.csv' % args.src)
        print('   期望结构: {src}/{RE1-OB|RE1-SS|RE1-TT}/{service}_{fault}/{case_id}/data.csv')
        print('   请用 --src 指向正确位置（可用 probe 定位后再跑）。')
        return

    g = defaultdict(list)
    for c in cases:
        g[(c[0], c[1])].append(c)

    print('探测到 %d 个案例，分布在 %d 个 (数据集, 服务_故障) 组' % (len(cases), len(g)))
    for ds in DATASETS:
        sub = {k: v for k, v in g.items() if k[0] == ds}
        if sub:
            sizes = sorted({len(v) for v in sub.values()})
            print('  %-8s %2d 组 / %3d 例（每组例数 %s）'
                  % (ds, len(sub), sum(len(v) for v in sub.values()), sizes))

    if args.probe_only:
        n_pick = sum(min(args.per_fault, len(v)) for v in g.values())
        print('\n✅ 探测完成。按 --per-fault %d 抽样将得到 %d 例。' % (args.per_fault, n_pick))
        print('   去掉 --probe-only 即可生成。')
        return

    rng = random.Random(SEED)
    # ⚠️ 输出必须按 k 分目录：否则先生成 k=2、再跑 k=4 时，会因为「文件已存在」被全部跳过
    base = os.path.join(args.dst, 'ds%d' % args.k)
    done = skipped = 0
    for (ds, sf), lst in sorted(g.items()):
        for (_ds, _sf, case, src_csv) in rng.sample(lst, min(args.per_fault, len(lst))):
            dst_dir = os.path.join(base, ds, sf, case)
            dst_csv = os.path.join(dst_dir, 'data.csv')
            if os.path.exists(dst_csv):
                skipped += 1
                continue
            os.makedirs(dst_dir, exist_ok=True)
            df = pd.read_csv(src_csv)
            df.iloc[::args.k].reset_index(drop=True).to_csv(dst_csv, index=False)
            inj = os.path.join(os.path.dirname(src_csv), 'inject_time.txt')
            if os.path.exists(inj):
                shutil.copy2(inj, os.path.join(dst_dir, 'inject_time.txt'))
            done += 1

    print('\n✅ 生成 %d 例（跳过已存在 %d 例），输出根：%s' % (done, skipped, base))
    print('   ⚠️ 重跑时窗口参数请用：--length %d / --window %d' % (20 // args.k, 20 // args.k))


if __name__ == '__main__':
    main()
