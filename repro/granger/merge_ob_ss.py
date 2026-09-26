"""Merge OB and SS Granger results and recompute eval."""
import pandas as pd
import glob, os, shutil

FAULT_TO_METRIC = {'delay': 'latency', 'loss': 'latency', 'disk': 'diskio'}
OB_DIR = r'D:\北交威\大创\Granger\output_granger_v4'
SS_DIR = r'D:\北交威\大创\Granger\output_granger_ss'
OUT_DIR = r'D:\北交威\大创\Granger\output_granger_merged'

os.makedirs(OUT_DIR, exist_ok=True)

# --- Copy OB cases ---
ob_root = os.path.join(OB_DIR, 'RE1-OB')
if os.path.isdir(ob_root):
    for root, _, files in os.walk(ob_root):
        for f in files:
            if f.endswith('.csv'):
                src = os.path.join(root, f)
                rel = os.path.relpath(root, ob_root)
                dst_dir = os.path.join(OUT_DIR, 'RE1-OB', rel)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, f))
    print(f'Copied RE1-OB cases')

# --- Copy SS cases ---
ss_root = os.path.join(SS_DIR, 'RE1-SS')
if os.path.isdir(ss_root):
    for root, _, files in os.walk(ss_root):
        for f in files:
            if f.endswith('.csv'):
                src = os.path.join(root, f)
                rel = os.path.relpath(root, ss_root)
                dst_dir = os.path.join(OUT_DIR, 'RE1-SS', rel)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, f))
    print(f'Copied RE1-SS cases')

# --- Merge summaries ---
summaries = []
for path in [os.path.join(OB_DIR, 'summary.csv'), os.path.join(SS_DIR, 'summary.csv')]:
    if os.path.exists(path):
        summaries.append(pd.read_csv(path))
if summaries:
    merged = pd.concat(summaries, ignore_index=True)
    merged.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    print(f'Merged summary: {len(merged)} rows')

# --- Recompute eval from merged CSVs ---
stats_by_ds = {}
overall_s = {'ac1': 0, 'ac3': 0, 'ac5': 0}
overall_m = {'ac1': 0, 'ac3': 0, 'ac5': 0}

for dataset in ['RE1-OB', 'RE1-SS']:
    base = os.path.join(OUT_DIR, dataset)
    if not os.path.isdir(base):
        continue
    for csv_path in glob.glob(os.path.join(base, '*', '*.csv')):
        parts = csv_path.replace('\\', '/').split('/')
        svc_fault = parts[-2]
        service, fault_type = svc_fault.split('_', 1)

        key = (dataset, fault_type)
        if key not in stats_by_ds:
            stats_by_ds[key] = {'n': 0, 's_ac1': 0, 's_ac3': 0, 's_ac5': 0,
                                'm_ac1': 0, 'm_ac3': 0, 'm_ac5': 0}
        stats_by_ds[key]['n'] += 1

        df = pd.read_csv(csv_path)
        nodes = df['node'].tolist()

        svc_rank = next((i+1 for i, n in enumerate(nodes)
                         if n.split('_')[0].replace('-db', '').replace('-', '') ==
                         service.replace('-', '').replace('-db', '')), 999)
        if svc_rank <= 1: stats_by_ds[key]['s_ac1'] += 1; overall_s['ac1'] += 1
        if svc_rank <= 3: stats_by_ds[key]['s_ac3'] += 1; overall_s['ac3'] += 1
        if svc_rank <= 5: stats_by_ds[key]['s_ac5'] += 1; overall_s['ac5'] += 1

        mapped = FAULT_TO_METRIC.get(fault_type, fault_type)
        truth_node = f'{service}_{mapped}'
        try:
            mtr_rank = nodes.index(truth_node) + 1
        except ValueError:
            mtr_rank = 999
        if mtr_rank <= 1: stats_by_ds[key]['m_ac1'] += 1; overall_m['ac1'] += 1
        if mtr_rank <= 3: stats_by_ds[key]['m_ac3'] += 1; overall_m['ac3'] += 1
        if mtr_rank <= 5: stats_by_ds[key]['m_ac5'] += 1; overall_m['ac5'] += 1

total_n = sum(s['n'] for s in stats_by_ds.values())
rows = []

for dataset in ['RE1-OB', 'RE1-SS']:
    print(f'\n{"="*70}\n{dataset}')
    print(f'{"":12} {"n":>5} {"Svc-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}  {"Mtr-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}')
    for ft in ['cpu', 'mem', 'delay', 'loss', 'disk', 'socket']:
        key = (dataset, ft)
        if key not in stats_by_ds:
            continue
        s = stats_by_ds[key]; n = s['n']
        sac1 = round(s['s_ac1']/n, 4); sac3 = round(s['s_ac3']/n, 4)
        sac5 = round(s['s_ac5']/n, 4); savg5 = round((sac1+sac3+sac5)/3, 4)
        mac1 = round(s['m_ac1']/n, 4); mac3 = round(s['m_ac3']/n, 4)
        mac5 = round(s['m_ac5']/n, 4); mavg5 = round((mac1+mac3+mac5)/3, 4)
        print(f'{ft.upper():<12} {n:>5} {sac1:>8.4f} {sac3:>8.4f} {sac5:>8.4f} {savg5:>8.4f}  {mac1:>8.4f} {mac3:>8.4f} {mac5:>8.4f} {mavg5:>8.4f}')
        rows.append([dataset, ft, n, sac1, sac3, sac5, savg5, mac1, mac3, mac5, mavg5])

osac1 = round(overall_s['ac1']/total_n, 4); osac3 = round(overall_s['ac3']/total_n, 4)
osac5 = round(overall_s['ac5']/total_n, 4); osavg5 = round((osac1+osac3+osac5)/3, 4)
omac1 = round(overall_m['ac1']/total_n, 4); omac3 = round(overall_m['ac3']/total_n, 4)
omac5 = round(overall_m['ac5']/total_n, 4); omavg5 = round((omac1+omac3+omac5)/3, 4)
rows.append(['OVERALL', '', total_n, osac1, osac3, osac5, osavg5, omac1, omac3, omac5, omavg5])
print(f'\n{"OVERALL":<12} {total_n:>5} {osac1:>8.4f} {osac3:>8.4f} {osac5:>8.4f} {osavg5:>8.4f}  {omac1:>8.4f} {omac3:>8.4f} {omac5:>8.4f} {omavg5:>8.4f}')

eval_df = pd.DataFrame(rows, columns=[
    'dataset', 'fault_type', 'n_cases', 'svc_ac1', 'svc_ac3', 'svc_ac5', 'svc_avg5',
    'mtr_ac1', 'mtr_ac3', 'mtr_ac5', 'mtr_avg5'])
eval_df.to_csv(os.path.join(OUT_DIR, 'eval.csv'), index=False)
print(f'\nSaved to {OUT_DIR}')
