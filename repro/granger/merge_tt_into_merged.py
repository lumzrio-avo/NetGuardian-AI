# -*- coding: utf-8 -*-
"""
把 output_granger_TT_new/RE1-TT 并入 output_granger_merged/RE1-TT

步骤：
  1) 备份旧的 output_granger_merged/RE1-TT → RE1-TT.pre_step15fix.bak
  2) 拷贝新 TT（125 CSV + 125 adj.npy + 125 nodes.json）覆盖进去
  3) 打印拷贝后的计数，供人工核对

纯标准库，Windows / WSL 均可直接运行。
用法：
    python merge_tt_into_merged.py          # 预览（不写任何文件）
    python merge_tt_into_merged.py --apply  # 实际执行
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(HERE, 'output_granger_merged')
TT_NEW = os.path.join(HERE, 'output_granger_TT_new')
DST = os.path.join(MERGED, 'RE1-TT')
SRC = os.path.join(TT_NEW, 'RE1-TT')
BAK = os.path.join(MERGED, 'RE1-TT.pre_step15fix.bak')

APPLY = '--apply' in sys.argv


def count(root):
    """返回 (csv 数, adj 数, nodes 数, svc_fault 目录数)"""
    if not os.path.isdir(root):
        return None
    c = a = n = 0
    svc = 0
    for d in sorted(os.listdir(root)):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        svc += 1
        for f in os.listdir(dd):
            if f.endswith('_adj.npy'):
                a += 1
            elif f.endswith('_nodes.json'):
                n += 1
            elif f.endswith('.csv') and f[:-4].isdigit():
                c += 1
    return c, a, n, svc


print('=' * 70)
print('源目录:', SRC)
print('目标  :', DST)
print('备份到:', BAK)
print('=' * 70)

src_cnt = count(SRC)
if src_cnt is None:
    sys.exit('!! 源目录不存在')
print('\n[源] TT_new      : csv=%d adj=%d nodes=%d svc_fault=%d' % src_cnt)

bak_exist = os.path.isdir(BAK)
print('[备] 备份是否已存在: %s' % ('是（将复用，不覆盖）' if bak_exist else '否（本次创建）'))

dst_before = count(DST)
print('[前] merged/RE1-TT: csv=%d adj=%d nodes=%d svc_fault=%d' % dst_before)

if not APPLY:
    print('\n（预览模式，未改动任何文件。加 --apply 执行）')
    sys.exit(0)

# --- 1) 备份 ---
if bak_exist:
    print('\n[1/3] 备份已存在，跳过')
else:
    shutil.move(DST, BAK)
    print('\n[1/3] 已备份 → %s' % os.path.basename(BAK))

# --- 2) 拷贝 ---
shutil.copytree(SRC, DST, dirs_exist_ok=True)
print('[2/3] 已拷贝 TT_new → merged/RE1-TT')

# --- 3) 核对 ---
dst_after = count(DST)
print('[3/3] merged/RE1-TT: csv=%d adj=%d nodes=%d svc_fault=%d' % dst_after)
ok = (dst_after[0] == 125 and dst_after[1] == 125
      and dst_after[2] == 125 and dst_after[3] == 25)
print('\n%s' % ('✅ 计数正确（125/125/125，25 个 svc_fault）' if ok
                else '⚠️ 计数异常，请人工检查'))

# 顺带把 TT_new 自带的 per-dataset summary/eval 也放进去（保留痕迹）
for name in ('summary.csv', 'eval.csv'):
    s = os.path.join(SRC, name)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(DST, name + '.from_tt_new.bak'))
print('     （TT_new 的 summary/eval 已另存为 *.from_tt_new.bak）')
print('\n完成。接下来运行：python rebuild_eval_official.py')
