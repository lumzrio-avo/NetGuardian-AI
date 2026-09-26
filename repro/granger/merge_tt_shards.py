"""
合并 TT 分片结果 + 与旧结果比对 + 按官方口径重算评估。

分三步：
  ①  把 output_granger_TT_shards/shard*/RE1-TT/** 合并到 output_granger_TT_new/RE1-TT/
      （每个案例的 {id}.csv 排名 + {id}_adj.npy 邻接矩阵 + {id}_nodes.json 节点名顺序）
  ②  与新结果比对旧结果（output_granger_merged/RE1-TT）的排名是否变化
      —— 因为旧 TT 结果是用「无 try/except」的版本跑的，需确认补丁没有改变结论
  ③  按官方口径重算 output_granger_TT_new 的 summary.csv / eval.csv

用法（WSL）:
  /home/lumzrio/RCAEval/env/bin/python \
      "/mnt/d/北交威/大创/结果/Granger/merge_tt_shards.py"

本脚本只做「拷贝 + 比对 + 评估」，不修改 output_granger_merged（旧结果保持原样）。
是否把新 TT 结果并入总目录，由你单独决定。
"""
import csv
import glob
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def to_local(p):
    """Windows 路径在 WSL/Linux 下自动转成 /mnt/<drive>/..."""
    if platform.system() != 'Windows' and len(p) > 2 and p[1] == ':':
        return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')
    return p


SHARD_BASE = os.path.join(HERE, 'output_granger_TT_shards')
NEW_DIR = os.path.join(HERE, 'output_granger_TT_new')
OLD_DIR = os.path.join(HERE, 'output_granger_merged')

SUFFIXES = ('_adj.npy', '_nodes.json')


def find_case_files(root, dataset=None):
    """返回 {(svc_fault, case_id): {'csv':.., 'adj':.., 'nodes':..}}

    dataset 不为 None 时只扫描 root/{dataset}/（旧目录里混着 OB/SS，必须限定）。
    """
    found = {}
    base = os.path.join(root, dataset) if dataset else root
    pattern = os.path.join(base, '**', '*.csv')
    for csv_path in glob.glob(pattern, recursive=True):
        svc_fault = os.path.basename(os.path.dirname(csv_path))
        stem = os.path.basename(csv_path)[:-4]
        if not stem.isdigit():
            continue
        case_id = int(stem)
        case_dir = os.path.dirname(csv_path)
        entry = found.setdefault((svc_fault, case_id), {})
        entry['csv'] = csv_path
        for suf in SUFFIXES:
            cand = os.path.join(case_dir, f'{case_id}{suf}')
            if os.path.exists(cand):
                entry[suf] = cand
    return found


def read_rank_order(csv_path):
    """读取案例 CSV 的 node 列顺序（= PageRank 降序排名）"""
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        return [row['node'] for row in csv.DictReader(f) if row.get('node')]


def main():
    print('=' * 78)
    print('合并 TT 分片结果')
    print('=' * 78)

    shard_dirs = sorted(glob.glob(os.path.join(SHARD_BASE, 'shard*')))
    shard_dirs = [d for d in shard_dirs if os.path.isdir(d)]
    if not shard_dirs:
        print(f'未找到任何分片目录: {SHARD_BASE}/shard*')
        sys.exit(1)

    # ---------- ① 收集 + 拷贝 ----------
    cases = {}
    conflicts = []
    for shard in shard_dirs:
        got = find_case_files(shard)
        print(f'  {os.path.basename(shard):<10} {len(got):>4} 个案例')
        for key, files in got.items():
            if key in cases:
                conflicts.append(key)
            cases[key] = files

    if conflicts:
        print(f'\n[ERROR] 有 {len(conflicts)} 个案例被多个分片重复产出（分片逻辑异常）:')
        for svc_fault, case_id in conflicts[:10]:
            print(f'    {svc_fault}/{case_id}')
        sys.exit(1)

    target_root = os.path.join(NEW_DIR, 'RE1-TT')
    os.makedirs(target_root, exist_ok=True)

    n_csv = n_adj = n_nodes = 0
    for (svc_fault, case_id), files in sorted(cases.items()):
        dst_dir = os.path.join(target_root, svc_fault)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(files['csv'], os.path.join(dst_dir, f'{case_id}.csv'))
        n_csv += 1
        for suf, counter in (('_adj.npy', 'adj'), ('_nodes.json', 'nodes')):
            if suf in files:
                shutil.copy2(files[suf], os.path.join(dst_dir, f'{case_id}{suf}'))
                if counter == 'adj':
                    n_adj += 1
                else:
                    n_nodes += 1

    print(f'\n合并结果 → {NEW_DIR}')
    print(f'  排名 CSV:   {n_csv:>4} / 125')
    print(f'  邻接矩阵:   {n_adj:>4} / 125')
    print(f'  节点名顺序: {n_nodes:>4} / 125')

    # 缺哪些
    if n_csv < 125:
        print('\n[WARN] 案例数不足 125，以下案例尚未产出：')
        # 用旧目录作为「应有哪些案例」的参照
        old = find_case_files(OLD_DIR, 'RE1-TT')
        missing = sorted(set(old) - set(cases))
        for svc_fault, case_id in missing:
            print(f'    {svc_fault}/{case_id}')
        print('  → 说明分片还没跑完。等跑完后再执行本脚本即可（--resume 会自动补）。')

    # ---------- ② 与旧结果比对排名 ----------
    print('\n' + '=' * 78)
    print('与旧 TT 结果比对（旧结果用无 try/except 的版本跑的）')
    print('=' * 78)

    old_cases = find_case_files(OLD_DIR, 'RE1-TT')
    same = diff = only_new = 0
    diff_list = []
    for key, files in sorted(cases.items()):
        new_order = read_rank_order(files['csv'])
        old_entry = old_cases.get(key)
        if old_entry is None:
            only_new += 1
            continue
        old_order = read_rank_order(old_entry['csv'])
        if old_order == new_order:
            same += 1
        else:
            diff += 1
            if len(diff_list) < 10:
                # 找出第一个不同的位置
                pos = next((i for i, (a, b) in enumerate(zip(old_order, new_order))
                            if a != b), min(len(old_order), len(new_order)))
                diff_list.append((key, len(old_order), len(new_order), pos))

    print(f'  排名一致:     {same:>4}')
    print(f'  排名不一致:   {diff:>4}')
    print(f'  旧目录没有的: {only_new:>4}')
    if diff_list:
        print('\n  不一致案例抽样（旧节点数 → 新节点数，第一个不同的名次）：')
        for (svc_fault, case_id), n_old, n_new, pos in diff_list:
            print(f'    {svc_fault}/{case_id}: {n_old} → {n_new}, 第 {pos + 1} 名起不同')
        print('\n  注：不一致的案例数**不等于** try/except 补丁的影响。')
        print('      本脚本同时打印了「旧节点数 → 新节点数」；')
        print('      若二者不同，说明输入列集合发生了变化，')
        print('      需先定位原因（例如 run_granger.py 的 Step 1.5「只保留 ts- 前缀列」），')
        print('      再判断是补丁影响还是输入差异。')
    else:
        print('\n  ✅ 排名与节点数均未变化。')

    # ---------- ③ 重算评估 ----------
    print('\n' + '=' * 78)
    print('按官方口径重算评估')
    print('=' * 78)
    rebuild = os.path.join(HERE, 'rebuild_eval_official.py')
    subprocess.run([sys.executable, rebuild, NEW_DIR], check=False)

    print(f'\n新结果目录: {NEW_DIR}')
    print('旧的 output_granger_merged 未被修改。')
    print('确认无误后如需并入总目录，再单独执行拷贝（先备份 RE1-TT）。')


if __name__ == '__main__':
    main()
