"""
按 RCAEval 官方口径重建 Granger 的 summary.csv / eval.csv

修正 merge_all.py 的两处偏差：
  1) 官方 main.py 在计算 service 级 AC@k 之前会先对服务实体去重
     （去重使真值位置前移，AC 提高）。merge_all.py 缺这一步，导致 SS/TT 被系统性低估。
  2) Avg@5 应为 sum(AC@1..AC@5)/5，而不是 mean(AC@1, AC@3, AC@5)。
     merge_all.py 显式写的是 (sac1+sac3+sac5)/3。

本脚本只读取 output_granger_merged/ 下已经保存的案例 CSV，
不重跑任何因果发现，几秒钟即可完成。仅依赖标准库。

对应官方实现:
  RCAEval/main.py                    第 380-397 行 (service / fine-grained 评估构造)
  RCAEval/benchmark/evaluation.py    AC@k / Avg@k 定义
"""
import os
import csv
import glob
import shutil
import platform
import sys


def to_local(p: str) -> str:
    """Windows 路径在 WSL/Linux 下自动转为 /mnt/<drive>/...，便于两边都能跑。"""
    if platform.system() != 'Windows' and len(p) > 2 and p[1] == ':':
        return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')
    return p


MERGED = (to_local(sys.argv[1]) if len(sys.argv) > 1
          else to_local(r'D:\北交威\大创\结果\Granger\output_granger_merged'))
DATASETS = ['RE1-OB', 'RE1-SS', 'RE1-TT']
FAULT_TO_METRIC = {'delay': 'latency', 'loss': 'latency', 'disk': 'diskio'}
FAULT_ORDER = ['cpu', 'mem', 'delay', 'loss', 'disk', 'socket']


# ------------------------------------------------------------
# 官方口径的两个名次函数
# ------------------------------------------------------------

def service_rank(nodes, service):
    """
    service 级名次（官方 main.py 口径）：
      节点实体取 '_' 前一段并去掉 '-db'，然后对实体序列去重，
      返回真值服务在去重后序列中的名次（1-based）。找不到返回 None。

    官方代码:
      s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
      s_ranks = [old[0]] + [old[i] for i in range(1, len(old)) if old[i] not in old[:i]]
    """
    if not nodes:
        return None
    entities = [n.split('_')[0].replace('-db', '') for n in nodes]
    dedup = [entities[0]] + [e for i, e in enumerate(entities) if i and e not in entities[:i]]
    try:
        return dedup.index(service) + 1
    except ValueError:
        return None


def metric_rank(nodes, service, fault_type):
    """
    metric 级名次（官方 main.py 口径）：
      节点拆成 服务_指标 两段，真值为 {service}_{FAULT_TO_METRIC[fault]}。
      官方此处不做去重。

    注意: SS/TT 的列名是 Prometheus 格式（如 carts_istio-latency-90），
    split('_')[1] 永远无法等于 cpu/mem/latency/diskio，
    因此 SS/TT 的 Mtr 恒为 0 —— 这是官方列名约定与数据格式不兼容所致。
    按既定决策：保持官方原样，不做映射（修复留待改进阶段）。
    """
    target_metric = FAULT_TO_METRIC.get(fault_type, fault_type)
    for i, n in enumerate(nodes):
        ent = n.split('_')[0]
        met = n.split('_')[1] if '_' in n else 'unknown'
        if ent == service and met == target_metric:
            return i + 1
    return None


# ------------------------------------------------------------
# 读取案例 CSV
# ------------------------------------------------------------

def collect_cases(merged_dir):
    cases = []
    for ds in DATASETS:
        base = os.path.join(merged_dir, ds)
        if not os.path.isdir(base):
            print(f'  [SKIP] {ds}: 目录不存在')
            continue
        n_ds = 0
        for path in sorted(glob.glob(os.path.join(base, '*', '*.csv'))):
            svc_fault = os.path.basename(os.path.dirname(path))
            if '_' not in svc_fault:
                continue
            service, fault_type = svc_fault.split('_', 1)
            try:
                case_id = int(os.path.basename(path)[:-4])
            except ValueError:
                continue

            with open(path, encoding='utf-8-sig') as f:
                nodes = [r['node'] for r in csv.DictReader(f) if r.get('node')]
            if not nodes:
                continue

            s_rank = service_rank(nodes, service)
            m_rank = metric_rank(nodes, service, fault_type)
            cases.append({
                'dataset':    ds,
                'service':    service,
                'fault_type': fault_type,
                'case_id':    case_id,
                'n_nodes':    len(nodes),
                'top1':       nodes[0],
                'top3':       '|'.join(nodes[:3]) if len(nodes) >= 3 else '',
                'ground_truth_service': service,
                'ground_truth_metric':  FAULT_TO_METRIC.get(fault_type, fault_type),
                'rank_of_truth':        s_rank if s_rank is not None else 'not_found',
                'rank_of_truth_metric': m_rank if m_rank is not None else 'not_found',
                '_s_rank': s_rank,
                '_m_rank': m_rank,
            })
            n_ds += 1
        print(f'  {ds}: {n_ds} 案例')
    return cases


# ------------------------------------------------------------
# 评估（严格按 official accuracy / average 定义）
# ------------------------------------------------------------

SUMMARY_COLS = ['dataset', 'service', 'fault_type', 'case_id', 'n_nodes',
                'top1', 'top3', 'ground_truth_service', 'ground_truth_metric',
                'rank_of_truth', 'rank_of_truth_metric']

EVAL_COLS = ['dataset', 'fault_type', 'n_cases',
             'svc_ac1', 'svc_ac3', 'svc_ac5', 'svc_avg5',
             'mtr_ac1', 'mtr_ac3', 'mtr_ac5', 'mtr_avg5']


def evaluate(cases):
    s_hits = {k: 0 for k in range(1, 6)}
    m_hits = {k: 0 for k in range(1, 6)}
    for c in cases:
        if c['_s_rank'] is not None:
            for k in range(1, 6):
                if c['_s_rank'] <= k:
                    s_hits[k] += 1
        if c['_m_rank'] is not None:
            for k in range(1, 6):
                if c['_m_rank'] <= k:
                    m_hits[k] += 1
    return s_hits, m_hits, len(cases)


def _ac(hits, n, k):
    return round(hits[k] / n, 4) if n else 0.0


def _avg5(hits, n):
    """官方 Avg@5 = sum(AC@1..AC@5) / 5"""
    return round(sum(hits[k] / n for k in range(1, 6)) / 5, 4) if n else 0.0


def eval_row(dataset, fault_type, cases):
    s_hits, m_hits, n = evaluate(cases)
    return [dataset, fault_type, n,
            _ac(s_hits, n, 1), _ac(s_hits, n, 3), _ac(s_hits, n, 5), _avg5(s_hits, n),
            _ac(m_hits, n, 1), _ac(m_hits, n, 3), _ac(m_hits, n, 5), _avg5(m_hits, n)]


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def main():
    print('=' * 78)
    print('按官方口径重建 Granger 评估结果（不重跑因果发现）')
    print('=' * 78)
    print(f'数据目录: {MERGED}\n')

    # 备份（只备份一次，避免覆盖原始备份）
    for fn in ['summary.csv', 'eval.csv']:
        src = os.path.join(MERGED, fn)
        bak = src + '.pre_official_eval.bak'
        if os.path.exists(src) and not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f'  已备份: {fn} -> {os.path.basename(bak)}')
    print()

    print('读取案例 CSV:')
    cases = collect_cases(MERGED)
    if not cases:
        print('未读到任何案例，终止。')
        return

    # === 写 summary.csv ===
    summary_path = os.path.join(MERGED, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in SUMMARY_COLS})

    # === 写 eval.csv ===
    eval_rows = []
    for ds in DATASETS:
        ds_cases = [c for c in cases if c['dataset'] == ds]
        if not ds_cases:
            continue
        for ft in FAULT_ORDER:
            ft_cases = [c for c in ds_cases if c['fault_type'] == ft]
            if ft_cases:
                eval_rows.append(eval_row(ds, ft, ft_cases))
        eval_rows.append(eval_row(ds, 'OVERALL', ds_cases))
    eval_rows.append(eval_row('ALL', 'OVERALL', cases))

    eval_path = os.path.join(MERGED, 'eval.csv')
    with open(eval_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(EVAL_COLS)
        w.writerows(eval_rows)

    # === 同步各数据集自己的 summary/eval，避免目录内数字互相矛盾 ===
    for ds in DATASETS:
        ds_cases = [c for c in cases if c['dataset'] == ds]
        if not ds_cases:
            continue
        ds_dir = os.path.join(MERGED, ds)
        with open(os.path.join(ds_dir, 'summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            for c in ds_cases:
                w.writerow({k: c[k] for k in SUMMARY_COLS})
        ds_rows = [r for r in eval_rows if r[0] == ds]
        with open(os.path.join(ds_dir, 'eval.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(EVAL_COLS)
            w.writerows(ds_rows)

    # === 打印 ===
    print(f'\n{"=" * 78}')
    print(f'{"dataset":<9}{"fault":<9}{"n":>5}{"SvcAC@1":>9}{"AC@3":>8}{"AC@5":>8}{"Avg@5":>8}'
          f'{"MtrAC@1":>9}{"AC@3":>7}{"AC@5":>7}{"Avg@5":>8}')
    print('-' * 78)
    for row in eval_rows:
        print(f'{row[0]:<9}{row[1].upper():<9}{row[2]:>5}'
              f'{row[3]:>9.4f}{row[4]:>8.4f}{row[5]:>8.4f}{row[6]:>8.4f}'
              f'{row[7]:>9.4f}{row[8]:>7.4f}{row[9]:>7.4f}{row[10]:>8.4f}')
    print('=' * 78)
    print(f'\n已写入:\n  {summary_path}\n  {eval_path}')
    print('\n注: SS/TT 的 Mtr 恒为 0 是官方列名约定与 Prometheus 列名不兼容所致，')
    print('    论文中应标注为 N/A（不可计算），修复留待改进阶段。')


if __name__ == '__main__':
    main()
