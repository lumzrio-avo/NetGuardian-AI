"""Generate summary + eval from TT case CSVs."""
import os, pandas as pd

FAULT_TO_METRIC = {'delay': 'latency', 'loss': 'latency', 'disk': 'diskio'}
TT_DIR = r'D:\北交威\大创\Granger\output_granger_TT'

all_results = []
stats = {}
overall_s = {'ac1': 0, 'ac3': 0, 'ac5': 0}

for root, _, files in os.walk(os.path.join(TT_DIR, 'RE1-TT')):
    for f in files:
        if not f.endswith('.csv') or not f[0].isdigit():
            continue
        csv_path = os.path.join(root, f)
        svc_fault = os.path.basename(os.path.dirname(csv_path))
        service, fault_type = svc_fault.split('_', 1)
        case_id = int(f.replace('.csv', ''))

        df = pd.read_csv(csv_path)
        nodes = df['node'].tolist()

        # Service-level: match ts-{name} prefix
        svc_rank = next((i+1 for i, n in enumerate(nodes)
                         if n.split('_')[0] == service), 999)

        # Metric-level: ts-{service}_{mapped_metric}
        mapped = FAULT_TO_METRIC.get(fault_type, fault_type)
        truth_node = f'{service}_{mapped}'
        try:
            mtr_rank = nodes.index(truth_node) + 1
        except ValueError:
            mtr_rank = 999

        all_results.append({
            'dataset': 'RE1-TT', 'service': service, 'fault_type': fault_type,
            'case_id': case_id, 'n_nodes': len(nodes),
            'top1': nodes[0] if nodes else '',
            'top3': '|'.join(nodes[:3]) if len(nodes) >= 3 else '',
            'ground_truth_service': service,
            'ground_truth_metric': FAULT_TO_METRIC.get(fault_type, fault_type),
            'rank_of_truth': str(svc_rank) if svc_rank < 999 else 'not_found',
            'rank_of_truth_metric': str(mtr_rank) if mtr_rank < 999 else 'not_found',
        })

        if fault_type not in stats:
            stats[fault_type] = {'n': 0, 's_ac1': 0, 's_ac3': 0, 's_ac5': 0}
        stats[fault_type]['n'] += 1
        if svc_rank <= 1: stats[fault_type]['s_ac1'] += 1; overall_s['ac1'] += 1
        if svc_rank <= 3: stats[fault_type]['s_ac3'] += 1; overall_s['ac3'] += 1
        if svc_rank <= 5: stats[fault_type]['s_ac5'] += 1; overall_s['ac5'] += 1

# Save summary
pd.DataFrame(all_results).to_csv(os.path.join(TT_DIR, 'summary.csv'), index=False)
print(f'summary.csv: {len(all_results)} cases')

# Build eval
total_n = sum(s['n'] for s in stats.values())
eval_rows = []
print(f'{"":12} {"n":>5} {"Svc-AC@1":>8} {"AC@3":>8} {"AC@5":>8} {"Avg@5":>8}')
for ft in ['cpu', 'mem', 'delay', 'loss', 'disk']:
    if ft not in stats: continue
    s = stats[ft]; n = s['n']
    sac1 = round(s['s_ac1']/n, 4); sac3 = round(s['s_ac3']/n, 4); sac5 = round(s['s_ac5']/n, 4)
    savg5 = round((sac1+sac3+sac5)/3, 4)
    print(f'{ft.upper():<12} {n:>5} {sac1:>8.4f} {sac3:>8.4f} {sac5:>8.4f} {savg5:>8.4f}')
    eval_rows.append([ft, n, sac1, sac3, sac5, savg5])

osac1 = round(overall_s['ac1']/total_n, 4); osac3 = round(overall_s['ac3']/total_n, 4)
osac5 = round(overall_s['ac5']/total_n, 4); osavg5 = round((osac1+osac3+osac5)/3, 4)
eval_rows.append(['OVERALL', total_n, osac1, osac3, osac5, osavg5])
print(f'OVERALL      {total_n:>5} {osac1:>8.4f} {osac3:>8.4f} {osac5:>8.4f} {osavg5:>8.4f}')

pd.DataFrame(eval_rows, columns=['fault_type','n_cases','svc_ac1','svc_ac3','svc_ac5','svc_avg5']).to_csv(os.path.join(TT_DIR, 'eval.csv'), index=False)
print(f'\neval.csv saved to {TT_DIR}')
