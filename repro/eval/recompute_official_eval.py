# -*- coding: utf-8 -*-
"""
recompute_official_eval.py — 用【已产出的案例级完整排名】重算三方法的官方口径指标

为什么能直接重算：评估口径（服务级去重、指标级精确匹配）只是**对排名列表的后处理**，
不依赖原始数据、不需要重跑因果发现。三方法的案例级完整排名都在本地。

【2026-09-26 更新】数据源改为 `结果/pc_pcmci_v2/`（PC/PCMCI 的官方口径修复版重跑结果）。
   新目录的布局与旧版不同，且每个配置顶层还有一个 126 行的汇总 `<algo>_summary.csv`
   （列名不同、没有 rank 列），本脚本会自动避开它。

用法:  python recompute_official_eval.py
输出:  官方口径重算_三方法对比.csv + 控制台表格 + 与各 runner 自带 metrics 的自检对照
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pc_pcmci_fixed'))

from common_eval import (FAULT_TO_METRIC, compute_rank_accuracy,  # noqa: E402
                         metric_rank, service_rank)

V2 = os.path.join(HERE, 'pc_pcmci_v2')

# ============================================================
# 案例级完整排名的位置
# ============================================================
GRANGER = {
    'RE1-OB': os.path.join(HERE, 'Granger', 'output_granger_merged', 'RE1-OB'),
    'RE1-SS': os.path.join(HERE, 'Granger', 'output_granger_merged', 'RE1-SS'),
    'RE1-TT': os.path.join(HERE, 'Granger', 'output_granger_merged', 'RE1-TT'),
}
PC = {ds: os.path.join(V2, 'RE1-%s-pc' % ds.split('-')[1]) for ds in ('RE1-OB', 'RE1-SS', 'RE1-TT')}
PCMCI = {ds: os.path.join(V2, 'RE1-%s-pcmci' % ds.split('-')[1]) for ds in ('RE1-OB', 'RE1-SS', 'RE1-TT')}

FAULT_ORDER = ['cpu', 'mem', 'delay', 'loss', 'disk']
DS_ORDER = ['RE1-OB', 'RE1-SS', 'RE1-TT']


def _split_path(p):
    return p.replace('\\', '/').split('/')


def _read_rank_file(path, name_col):
    """读一个案例级排名文件；返回按 rank 升序的节点名列表（不合法则 None）"""
    try:
        rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    except Exception:
        return None
    if not rows or 'rank' not in rows[0] or name_col not in rows[0]:
        return None
    try:
        rows.sort(key=lambda r: int(r['rank']))
    except Exception:
        return None
    return [r[name_col] for r in rows]


# ============================================================
# 读取：统一返回 {(ds, svc, fault, case): [节点名按名次升序]}
# ============================================================
def load_granger():
    """Granger: {ds}/{svc}_{fault}/{case}.csv  (rank,node,pagerank_score)"""
    out = {}
    for ds, root in GRANGER.items():
        for p in glob.glob(os.path.join(root, '*_*', '*.csv')):
            parts = _split_path(p)
            fname = parts[-1]
            svc_fault = parts[-2]
            if not fname[:-4].isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'node')
            if nodes:
                out[(ds, svc, fault, fname[:-4])] = nodes
    return out


def load_pc():
    """PC(v2): {root}/{svc}_{fault}/{case}/pc_summary.csv  (metric,score,rank)"""
    out = {}
    for ds, root in PC.items():
        for p in glob.glob(os.path.join(root, '*_*', '*', 'pc_summary.csv')):
            parts = _split_path(p)
            case, svc_fault = parts[-2], parts[-3]
            if not case.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'metric')
            if nodes:
                out[(ds, svc, fault, case)] = nodes
    return out


def load_pcmci():
    """PCMCI(v2) 两种布局：
         A) {root}/{svc}_{fault}/{case}/pcmci_summary.csv   (metric,score,rank)   —— OB 走 run_rca_full_ob.py
         B) {root}/[中间层]/{svc}_{fault}/{case}_ranks.csv  (rank,node,...)       —— SS/TT 走 run_pcmci.py
       （B 的中间层目录名可能是 runner 硬编码的 'RE1-SS'，与真实数据集无关，故用 ** 通配）
    """
    out = {}
    for ds, root in PCMCI.items():
        for p in glob.glob(os.path.join(root, '*_*', '*', 'pcmci_summary.csv')):
            parts = _split_path(p)
            case, svc_fault = parts[-2], parts[-3]
            if not case.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'metric')
            if nodes:
                out[(ds, svc, fault, case)] = nodes
        for p in glob.glob(os.path.join(root, '**', '*_*', '*_ranks.csv'), recursive=True):
            parts = _split_path(p)
            base = parts[-1][:-len('_ranks.csv')]
            svc_fault = parts[-2]
            if not base.isdigit() or '_' not in svc_fault:
                continue
            svc, fault = svc_fault.split('_', 1)
            nodes = _read_rank_file(p, 'node')
            if nodes:
                out.setdefault((ds, svc, fault, base), nodes)
    return out


# ============================================================
# 旧口径（各 runner 修改前实际用的那套），仅用于量化口径差异
# ============================================================
PC_TT_FAMILY = {
    'cpu': 'container-cpu', 'delay': 'latency', 'loss': 'latency',
    'disk': 'container-fs', 'mem': 'container-memory',
}


def old_service_rank(nodes, service):
    """旧口径：首次出现位置，不去重、不做 '-db' 归一"""
    e = [n.split('_')[0] for n in nodes]
    return (e.index(service) + 1) if service in e else None


def old_metric_rank(nodes, service, fault, method, ds):
    """旧指标级口径：PC + TT 用指标族前缀匹配；其余用精确 f"{service}_{mapped}\""""
    if method == 'pc' and ds == 'RE1-TT':
        family = PC_TT_FAMILY.get(fault, fault)
        if fault in ('delay', 'loss'):
            targets = ['%s_latency' % service]
        else:
            targets = [c for c in nodes if c.startswith('%s_%s' % (service, family))]
        hits = [nodes.index(t) + 1 for t in targets if t in nodes]
        return min(hits) if hits else None
    target = '%s_%s' % (service, FAULT_TO_METRIC.get(fault, fault))
    return (nodes.index(target) + 1) if target in nodes else None


# ============================================================
# 汇总
# ============================================================
def summarize(rows):
    """rows: [(ds, fault, svc_rank, mtr_rank, n)] -> 指标 dict"""
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def hit(rank, k):
        return 1.0 if (rank is not None and rank <= k) else 0.0

    return {
        'n': len(rows),
        'Svc-AC@1': mean([hit(s, 1) for _, _, s, _, _ in rows]),
        'Svc-AC@3': mean([hit(s, 3) for _, _, s, _, _ in rows]),
        'Svc-AC@5': mean([hit(s, 5) for _, _, s, _, _ in rows]),
        'Mtr-AC@1': mean([hit(m, 1) for _, _, _, m, _ in rows]),
        'Mtr-AC@3': mean([hit(m, 3) for _, _, _, m, _ in rows]),
        'Mtr-AC@5': mean([hit(m, 5) for _, _, _, m, _ in rows]),
    }


def collect(nodes_map, mode, method):
    rows = []
    for (ds, svc, fault, case), nodes in nodes_map.items():
        if mode == 'official':
            r = compute_rank_accuracy(nodes, svc, fault)
            s = r['Svc_Rank']
            s = None if s > len(nodes) else s
            m = r['Mtr_Rank']
            m = None if m > len(nodes) else m
        else:
            s = old_service_rank(nodes, svc)
            m = old_metric_rank(nodes, svc, fault, method=method, ds=ds)
        rows.append((ds, fault, s, m, len(nodes)))
    return rows


def table(title, rows):
    print('=' * 96)
    print(title)
    print('%-9s %-6s %5s %8s %8s %8s  | %8s %8s %8s' %
          ('dataset', 'fault', 'n', 'Svc@1', 'Svc@3', 'Svc@5', 'Mtr@1', 'Mtr@3', 'Mtr@5'))
    print('-' * 96)
    allrows = []
    for ds in DS_ORDER:
        sub = [r for r in rows if r[0] == ds]
        if not sub:
            continue
        for fault in FAULT_ORDER:
            fs = [r for r in sub if r[1] == fault]
            if not fs:
                continue
            a = summarize(fs)
            print('%-9s %-6s %5d %8.3f %8.3f %8.3f  | %8.3f %8.3f %8.3f' %
                  (ds, fault, a['n'], a['Svc-AC@1'], a['Svc-AC@3'], a['Svc-AC@5'],
                   a['Mtr-AC@1'], a['Mtr-AC@3'], a['Mtr-AC@5']))
        a = summarize(sub)
        print('%-9s %-6s %5d %8.3f %8.3f %8.3f  | %8.3f %8.3f %8.3f' %
              (ds, 'ALL', a['n'], a['Svc-AC@1'], a['Svc-AC@3'], a['Svc-AC@5'],
               a['Mtr-AC@1'], a['Mtr-AC@3'], a['Mtr-AC@5']))
        allrows += sub
    a = summarize(allrows)
    print('%-9s %-6s %5d %8.3f %8.3f %8.3f  | %8.3f %8.3f %8.3f' %
          ('ALL', 'ALL', a['n'], a['Svc-AC@1'], a['Svc-AC@3'], a['Svc-AC@5'],
           a['Mtr-AC@1'], a['Mtr-AC@3'], a['Mtr-AC@5']))
    return ({ds: summarize([r for r in allrows if r[0] == ds]) for ds in DS_ORDER},
            summarize(allrows))


def runner_metrics(cfg_dir, algo):
    """读各 runner 自带的 <algo>_metrics.csv（行式 metric,value）"""
    p = os.path.join(cfg_dir, '%s_metrics.csv' % algo)
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        try:
            out[r['metric']] = float(r['value'])
        except Exception:
            pass
    return out


# ============================================================
# 主流程
# ============================================================
LOADERS = {'Granger': (load_granger, None),      # (loader, 各配置目录函数)
           'PC': (load_pc, lambda ds: PC[ds]),
           'PCMCI': (load_pcmci, lambda ds: PCMCI[ds])}

summary_rows = []
for method, (loader, cfg_of) in LOADERS.items():
    nodes_map = loader()
    if not nodes_map:
        print('!! %s 未读到案例排名，跳过' % method)
        continue
    print('\n### %s：读到 %d 例' % (method, len(nodes_map)))
    method_l = method.lower()

    old_rows = collect(nodes_map, 'old', method_l)
    off_rows = collect(nodes_map, 'official', method_l)

    table('%s — 对照：修改前的旧口径（仅用于量化口径差异）' % method, old_rows)
    print()
    t_off, _ = table('%s — 官方口径' % method, off_rows)

    for ds in DS_ORDER:
        if ds not in t_off:
            continue
        n = t_off[ds]
        summary_rows.append({
            'method': method, 'dataset': ds, 'n': n['n'],
            'Svc@1': round(n['Svc-AC@1'], 4), 'Svc@3': round(n['Svc-AC@3'], 4),
            'Svc@5': round(n['Svc-AC@5'], 4),
            'Mtr@1': round(n['Mtr-AC@1'], 4), 'Mtr@3': round(n['Mtr-AC@3'], 4),
            'Mtr@5': round(n['Mtr-AC@5'], 4),
        })

    # === 自检：与各 runner 自带的 metrics 对照（应完全一致）===
    if cfg_of is not None:
        print('\n  [自检] %s：本脚本重算 vs 各 runner 自带 metrics' % method)
        for ds in DS_ORDER:
            rm = runner_metrics(cfg_of(ds), method_l)
            if not rm or ds not in t_off:
                continue
            mine = t_off[ds]
            diffs = []
            for k_src, k_dst in [('Svc-AC@1', 'Svc-AC@1'), ('Svc-AC@3', 'Svc-AC@3'),
                                 ('Svc-AC@5', 'Svc-AC@5'), ('Mtr-AC@1', 'Mtr-AC@1'),
                                 ('Mtr-AC@5', 'Mtr-AC@5')]:
                if k_src in rm:
                    d = abs(rm[k_src] - mine[k_dst])
                    if d > 1e-6:
                        diffs.append('%s: runner=%.4f vs 重算=%.4f' % (k_src, rm[k_src], mine[k_dst]))
            print('    %-9s %s' % (ds, '✅ 一致' if not diffs else '⚠️ ' + '; '.join(diffs)))

out_csv = os.path.join(HERE, '官方口径重算_三方法对比.csv')
with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)
print('\n已写出: %s' % out_csv)

# ============================================================
# 自检：Granger 的官方口径应与既有 eval.csv 一致（口径实现的回归检查）
# ============================================================
print('\n' + '=' * 96)
print('自检：Granger 官方口径 vs 既有 eval.csv（rebuild_eval_official.py 产出）')
g = load_granger()
rows = collect(g, 'official', 'granger')
for ds in DS_ORDER:
    a = summarize([r for r in rows if r[0] == ds])
    print('  重算   %-9s Svc-AC@1=%.4f @3=%.4f @5=%.4f  Mtr-AC@5=%.3f' %
          (ds, a['Svc-AC@1'], a['Svc-AC@3'], a['Svc-AC@5'], a['Mtr-AC@5']))
print('  既有   OB 0.088/0.256/0.392 | SS 0.224/0.608/0.760 | TT 0.120/0.368/0.464')
