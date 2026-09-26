"""
Step 1.5 修正后的 TT 重跑准备：
把「受 Step 1.5 影响」的案例从分片目录移出，使 run_granger.py --resume 自动重算它们，
其余未受影响的案例原样保留（不重跑）。

判定依据
--------
Step 1.5 会删掉所有不以 "ts-" 开头的列。若某案例被删的列中**有列通过了 drop_constant**
（即该列在窗口内非常量），最终节点集合就会变小 ⇒ 该案例受影响。
因此：

    受影响  ⟺  旧结果的节点数 > 新结果的节点数

其中「旧结果」= output_granger_merged/RE1-TT（Step 1.5 修正前的历史结果，含宿主机 node-* 列），
「新结果」= output_granger_TT_shards（Step 1.5 误命中时的结果）。

两者相等的案例，说明 Step 1.5 没有删掉任何「本可保留」的列 ⇒ 因果图完全相同 ⇒ 无需重跑。
（实测这 68 个案例的排名也与旧结果逐例一致，可交叉验证。）

用法
----
    # 预览（不改任何文件）
    python prepare_rerun_tt_step15.py

    # 实际执行（把受影响的案例移到 _stale_step15/ 下）
    python prepare_rerun_tt_step15.py --apply

执行后用原命令重启分片即可（--resume 会自动跳过未受影响的案例）：

    bash "/mnt/d/北交威/大创/结果/Granger/run_tt_shards.sh"
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_BASE = os.path.join(HERE, 'output_granger_TT_shards')
STALE_DIR = os.path.join(SHARD_BASE, '_stale_step15')
OLD_DIR = os.path.join(HERE, 'output_granger_merged')


def read_new_counts():
    """{(svc_fault, case_id): (shard_dir, n_nodes)}"""
    out = {}
    for p in glob.glob(os.path.join(SHARD_BASE, 'shard*', 'RE1-TT', '*', '*_nodes.json')):
        case_dir = os.path.dirname(p)                      # .../shardN/RE1-TT/svc_fault
        svc_fault = os.path.basename(case_dir)
        shard_dir = os.path.dirname(os.path.dirname(case_dir))   # .../shardN
        case_id = os.path.basename(p).replace('_nodes.json', '')
        n = len(json.load(open(p, encoding='utf-8')))
        out[(svc_fault, case_id)] = (shard_dir, n)
    return out


def read_old_counts():
    """{(svc_fault, case_id): n_nodes}，以 CSV 行数为准"""
    out = {}
    for p in glob.glob(os.path.join(OLD_DIR, 'RE1-TT', '*', '*.csv')):
        stem = os.path.basename(p)[:-4]
        if not stem.isdigit():
            continue
        svc_fault = os.path.basename(os.path.dirname(p))
        with open(p, encoding='utf-8-sig') as f:
            out[(svc_fault, stem)] = sum(1 for _ in f) - 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正移动文件（默认只预览）')
    args = ap.parse_args()

    new = read_new_counts()
    old = read_old_counts()
    print(f'新结果（分片）: {len(new)} 例')
    print(f'旧结果（merged）: {len(old)} 例')

    common = sorted(set(new) & set(old))
    affected = [k for k in common if old[k] > new[k][1]]
    same = [k for k in common if old[k] == new[k][1]]
    other = [k for k in common if old[k] < new[k][1]]

    print(f'\n受影响（旧节点数 > 新节点数）: {len(affected)} 例 —— 需要重跑')
    print(f'未受影响（节点数相同）:        {len(same)} 例 —— 可保留')
    if other:
        print(f'⚠️ 新节点数 > 旧节点数: {len(other)} 例（异常，请检查）')
    only_new = set(new) - set(old)
    only_old = set(old) - set(new)
    if only_new:
        print(f'⚠️ 新结果有而旧结果无: {len(only_new)} 例')
    if only_old:
        print(f'⚠️ 旧结果有而新结果无: {len(only_old)} 例')

    if not args.apply:
        print('\n[预览模式] 未改动任何文件。加 --apply 执行移动。')
        print('\n受影响的案例（前 10 个）:')
        for svc_fault, cid in affected[:10]:
            print(f'    {svc_fault}/{cid}: 旧 {old[(svc_fault, cid)]} → 新 {new[(svc_fault, cid)][1]}')
        return

    moved = 0
    for svc_fault, cid in affected:
        shard_dir, _ = new[(svc_fault, cid)]
        src_dir = os.path.join(shard_dir, 'RE1-TT', svc_fault)
        dst_dir = os.path.join(STALE_DIR, os.path.basename(shard_dir), 'RE1-TT', svc_fault)
        os.makedirs(dst_dir, exist_ok=True)
        for suffix in ('.csv', '_adj.npy', '_nodes.json'):
            src = os.path.join(src_dir, f'{cid}{suffix}')
            if os.path.exists(src):
                shutil.move(src, os.path.join(dst_dir, f'{cid}{suffix}'))
        moved += 1

    # 顺便把各分片的 summary/eval 也移走，让 --resume 走「扫描 CSV 目录」这条干净路径
    for p in glob.glob(os.path.join(SHARD_BASE, 'shard*', 'summary.csv')) + \
             glob.glob(os.path.join(SHARD_BASE, 'shard*', 'eval.csv')):
        shard = os.path.basename(os.path.dirname(p))
        dst = os.path.join(STALE_DIR, shard)
        os.makedirs(dst, exist_ok=True)
        shutil.move(p, os.path.join(dst, os.path.basename(p)))

    print(f'\n✅ 已移出 {moved} 个受影响案例 → {STALE_DIR}')
    print('   重启命令: bash "/mnt/d/北交威/大创/结果/Granger/run_tt_shards.sh"')


if __name__ == '__main__':
    main()
