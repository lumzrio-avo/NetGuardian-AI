"""
统一重建 OB/SS/TT 的 eval + 合并到 output_granger_merged
"""
import os, glob, shutil, pandas as pd

FAULT_TO_METRIC = {'delay': 'latency', 'loss': 'latency', 'disk': 'diskio'}

DATASETS = {
    'RE1-OB': r'D:\北交威\大创\Granger\output_granger_v4_OB',
    'RE1-SS': r'D:\北交威\大创\Granger\output_granger_ss',
    'RE1-TT': r'D:\北交威\大创\Granger\output_granger_TT',
}
OUT_DIR = r'D:\北交威\大创\Granger\output_granger_merged'
os.makedirs(OUT_DIR, exist_ok=True)


def compute_eval(dataset_name, src_dir, out_dir):
    """读取案例 CSV，生成 summary.csv + eval.csv"""
    base = os.path.join(src_dir, dataset_name)
    if not os.path.isdir(base):
        print(f'  SKIP {dataset_name}: 目录不存在')
        return [], {}

    all_results = []
    stats = {}
    overall_s = {'ac1': 0, 'ac3': 0, 'ac5': 0}
    overall_m = {'ac1': 0, 'ac3': 0, 'ac5': 0}

    for csv_path in sorted(glob.glob(os.path.join(base, '*', '*.csv'))):
        parts = csv_path.replace('\\', '/').split('/')
        svc_fault = parts[-2]
        if '_' not in svc_fault:
            continue
        service, fault_type = svc_fault.split('_', 1)
        try:
            case_id = int(parts[-1].replace('.csv', ''))
        except ValueError:
            continue

        df = pd.read_csv(csv_path)
        nodes = df['node'].tolist()
        n_nodes = len(nodes)

        # Service-level: 第一个匹配 service 前缀的节点
        svc_rank = next((i+1 for i, n in enumerate(nodes)
                         if n.split('_')[0].replace('-db', '') == service.replace('-db', '')), 999)

        # Metric-level
        mapped = FAULT_TO_METRIC.get(fault_type, fault_type)
        truth_node = f'{service}_{mapped}'
        try:
            mtr_rank = nodes.index(truth_node) + 1
        except ValueError:
            mtr_rank = 999

        all_results.append({
            'dataset': dataset_name, 'service': service, 'fault_type': fault_type,
            'case_id': case_id, 'n_nodes': n_nodes,
            'top1': nodes[0] if nodes else '',
            'top3': '|'.join(nodes[:3]) if len(nodes) >= 3 else '',
            'ground_truth_service': service,
            'ground_truth_metric': mapped,
            'rank_of_truth': str(svc_rank) if svc_rank < 999 else 'not_found',
            'rank_of_truth_metric': str(mtr_rank) if mtr_rank < 999 else 'not_found',
        })

        # Copy CSV to merged dir
        dst_dir = os.path.join(out_dir, dataset_name, svc_fault)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(csv_path, os.path.join(dst_dir, f'{case_id}.csv'))

        # Stats
        if fault_type not in stats:
            stats[fault_type] = {'n': 0, 's_ac1': 0, 's_ac3': 0, 's_ac5': 0,
                                  'm_ac1': 0, 'm_ac3': 0, 'm_ac5': 0}
        stats[fault_type]['n'] += 1
        if svc_rank <= 1: stats[fault_type]['s_ac1'] += 1; overall_s['ac1'] += 1
        if svc_rank <= 3: stats[fault_type]['s_ac3'] += 1; overall_s['ac3'] += 1
        if svc_rank <= 5: stats[fault_type]['s_ac5'] += 1; overall_s['ac5'] += 1
        if mtr_rank != 999:
            if mtr_rank <= 1: stats[fault_type]['m_ac1'] += 1; overall_m['ac1'] += 1
            if mtr_rank <= 3: stats[fault_type]['m_ac3'] += 1; overall_m['ac3'] += 1
            if mtr_rank <= 5: stats[fault_type]['m_ac5'] += 1; overall_m['ac5'] += 1

    # Save individual summary
    if all_results:
        pd.DataFrame(all_results).to_csv(os.path.join(src_dir, 'summary.csv'), index=False)
        pd.DataFrame(all_results).to_csv(os.path.join(out_dir, dataset_name, 'summary.csv'), index=False)

    # Print + save individual eval
    total_n = sum(s['n'] for s in stats.values())
    eval_rows = []
    print(f'\n{"="*80}')
    print(f'{dataset_name}  (n={total_n})')
    print(f'{"":12} {"n":>4} {"Svc-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}  {"Mtr-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}')
    print('-' * 80)
    for ft in ['cpu', 'mem', 'delay', 'loss', 'disk', 'socket']:
        if ft not in stats: continue
        s = stats[ft]; n = s['n']
        sac1 = round(s['s_ac1']/n, 4); sac3 = round(s['s_ac3']/n, 4); sac5 = round(s['s_ac5']/n, 4)
        savg5 = round((sac1+sac3+sac5)/3, 4)
        mac1 = round(s['m_ac1']/n, 4); mac3 = round(s['m_ac3']/n, 4); mac5 = round(s['m_ac5']/n, 4)
        mavg5 = round((mac1+mac3+mac5)/3, 4)
        print(f'{ft.upper():<12} {n:>4} {sac1:>8.4f} {sac3:>8.4f} {sac5:>8.4f} {savg5:>8.4f}  {mac1:>8.4f} {mac3:>8.4f} {mac5:>8.4f} {mavg5:>8.4f}')
        eval_rows.append([dataset_name, ft, n, sac1, sac3, sac5, savg5, mac1, mac3, mac5, mavg5])

    osac1 = round(overall_s['ac1']/total_n, 4); osac3 = round(overall_s['ac3']/total_n, 4)
    osac5 = round(overall_s['ac5']/total_n, 4); osavg5 = round((osac1+osac3+osac5)/3, 4)
    omac1 = round(overall_m['ac1']/total_n, 4); omac3 = round(overall_m['ac3']/total_n, 4)
    omac5 = round(overall_m['ac5']/total_n, 4); omavg5 = round((omac1+omac3+omac5)/3, 4)
    eval_rows.append([dataset_name, 'OVERALL', total_n, osac1, osac3, osac5, osavg5, omac1, omac3, omac5, omavg5])
    print(f'{"OVERALL":<12} {total_n:>4} {osac1:>8.4f} {osac3:>8.4f} {osac5:>8.4f} {osavg5:>8.4f}  {omac1:>8.4f} {omac3:>8.4f} {omac5:>8.4f} {omavg5:>8.4f}')

    eval_df = pd.DataFrame(eval_rows, columns=[
        'dataset','fault_type','n_cases','svc_ac1','svc_ac3','svc_ac5','svc_avg5',
        'mtr_ac1','mtr_ac3','mtr_ac5','mtr_avg5'])
    eval_df.to_csv(os.path.join(src_dir, 'eval.csv'), index=False)
    eval_df.to_csv(os.path.join(out_dir, dataset_name, 'eval.csv'), index=False)

    return all_results, eval_rows


# === 1. 重建各数据集 eval + 复制到 merged ===
print('重建各数据集 eval + 合并案例 CSV...')
all_merged_results = []
all_eval_rows = []

for ds_name, ds_dir in DATASETS.items():
    results, eval_rows = compute_eval(ds_name, ds_dir, OUT_DIR)
    all_merged_results.extend(results)
    all_eval_rows.extend(eval_rows)

# === 2. 合并 summary.csv ===
merged_summary = pd.DataFrame(all_merged_results)
merged_summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
print(f'\n{"="*80}')
print(f'合并 summary.csv: {len(all_merged_results)} 案例')

# === 3. 合并 eval.csv（按数据集×故障类型 + 总体）===
merged_eval = pd.DataFrame(all_eval_rows, columns=[
    'dataset','fault_type','n_cases','svc_ac1','svc_ac3','svc_ac5','svc_avg5',
    'mtr_ac1','mtr_ac3','mtr_ac5','mtr_avg5'])

# 添加三数据集合并的 OVERALL 行
total_n = sum(r[2] for r in all_eval_rows if r[1] != 'OVERALL')
# 重新计算总体（加权平均）
total_s_ac1 = sum(r[3] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')
total_s_ac3 = sum(r[4] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')
total_s_ac5 = sum(r[5] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')
total_m_ac1 = sum(r[7] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')
total_m_ac3 = sum(r[8] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')
total_m_ac5 = sum(r[9] * r[2] for r in all_eval_rows if r[1] != 'OVERALL')

g_sac1 = round(total_s_ac1 / total_n, 4)
g_sac3 = round(total_s_ac3 / total_n, 4)
g_sac5 = round(total_s_ac5 / total_n, 4)
g_savg5 = round((g_sac1 + g_sac3 + g_sac5) / 3, 4)
g_mac1 = round(total_m_ac1 / total_n, 4)
g_mac3 = round(total_m_ac3 / total_n, 4)
g_mac5 = round(total_m_ac5 / total_n, 4)
g_mavg5 = round((g_mac1 + g_mac3 + g_mac5) / 3, 4)

merged_eval = pd.concat([merged_eval, pd.DataFrame([['ALL', 'OVERALL', total_n, g_sac1, g_sac3, g_sac5, g_savg5, g_mac1, g_mac3, g_mac5, g_mavg5]])], ignore_index=True)
merged_eval.to_csv(os.path.join(OUT_DIR, 'eval.csv'), index=False)

print(f'\n{"="*80}')
print(f'三数据集合并 (n={total_n})')
print(f'{"":12} {"Svc-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}  {"Mtr-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}')
print(f'{"ALL":<12} {g_sac1:>8.4f} {g_sac3:>8.4f} {g_sac5:>8.4f} {g_savg5:>8.4f}  {g_mac1:>8.4f} {g_mac3:>8.4f} {g_mac5:>8.4f} {g_mavg5:>8.4f}')
print(f'\n合并完成! 输出目录: {OUT_DIR}')
print(f'  - RE1-OB/  ({DATASETS["RE1-OB"]})')
print(f'  - RE1-SS/  ({DATASETS["RE1-SS"]})')
print(f'  - RE1-TT/  ({DATASETS["RE1-TT"]})')
print(f'  - summary.csv  ({total_n} 案例)')
print(f'  - eval.csv     (含各数据集+总体)')
