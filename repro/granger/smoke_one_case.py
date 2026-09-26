"""
单案例冒烟测试：验证 Granger 补丁（try/except）能否让此前失败的 MEM / CPU 案例跑通。

背景：官方 granger.py 的 grangercausalitytests 是裸调用，遇到常量/近常量指标对
（InfeasibleTestError: VAR has a perfect data）会中断整个案例。
我们改为跳过该对并视为无因果边。本脚本只跑 1 个案例来确认补丁生效。

用法（在 ~/RCAEval 目录下执行，数据路径是相对路径）:
  /home/lumzrio/RCAEval/env/bin/python /mnt/d/北交威/大创/结果/Granger/smoke_one_case.py

也可指定自己的案例:
  ... smoke_one_case.py "data/RE1/RE1-OB/adservice_mem/1/data.csv"
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_granger as rg   # noqa: E402  (它内部会自动把 RCAEval 加入 sys.path)

DS_KEY = {'RE1-OB': 're1-ob', 'RE1-SS': 're1-ss', 'RE1-TT': 're1-tt'}

# 默认测试案例：MEM 是此前 100% 失败的故障类型，CPU 失败率也很高
DEFAULT_CASES = [
    ('RE1-OB', 'adservice_mem', 1),
    ('RE1-OB', 'cartservice_cpu', 1),
]


def resolve(rel_path):
    """相对路径按 ~/RCAEval 解析（WSL 下执行时 cwd 必须是 ~/RCAEval）。"""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    for c in (rel_path,
              os.path.join(os.path.expanduser('~/RCAEval'), rel_path),
              os.path.join('/home/lumzrio/RCAEval', rel_path)):
        if os.path.exists(c):
            return c
    return None


def run_one(ds_name, svc_fault, case_id, length=20):
    root = rg.DATASET_MAP[DS_KEY[ds_name]]
    rel = os.path.join(root, svc_fault, str(case_id), 'data.csv')
    data_path = resolve(rel)
    if data_path is None:
        print(f'[SKIP] 找不到数据: {rel}')
        return None

    service, fault = svc_fault.split('_', 1)
    print(f'--- {ds_name} / {svc_fault} / {case_id} ---')
    print(f'    数据: {data_path}')

    t0 = time.time()
    try:
        data, inject_time, svc, ft, sli = rg.load_and_preprocess(
            data_path, window_length_min=length, verbose=False)
        print(f'    预处理后: shape={data.shape}  inject_time={inject_time}  sli={sli}')

        result = rg.run_granger_pagerank(
            data, inject_time=inject_time, dataset=ds_name, sli=sli, verbose=False)
    except Exception:
        print('    ❌ 仍然失败（补丁未生效或另有异常）：')
        traceback.print_exc()
        return False

    elapsed = time.time() - t0
    ranks = result['ranks']
    try:
        truth_rank = next(i + 1 for i, n in enumerate(ranks)
                          if n.split('_')[0] == service)
    except StopIteration:
        truth_rank = 'not_found'

    print(f'    ✅ 跑通  耗时 {elapsed:.1f}s')
    print(f'    图规模: n_nodes={result["n_nodes"]}  n_edges={result["n_edges"]}')
    print(f'    真值服务名次: {truth_rank}')
    print(f'    Top-5: {ranks[:5]}')
    print()
    return True


def main():
    if len(sys.argv) > 1:
        p = resolve(sys.argv[1])
        if p is None:
            print(f'找不到数据文件: {sys.argv[1]}')
            sys.exit(1)
        # 从路径反推 dataset / service_fault / case_id
        # 目录结构有两种可能：
        #   {root}/{svc_fault}/{case_id}/data.csv   ← RE1/RE2 实际结构
        #   {root}/{svc_fault}/{case_id}.csv
        parts = [x for x in p.replace('\\', '/').split('/') if x]
        if parts and parts[-1].lower().endswith('.csv'):
            parts = parts[:-1]          # 丢掉文件名（data.csv / 1.csv）
        if not parts or not parts[-1].isdigit():
            print(f'无法从路径反推案例编号（找不到纯数字的案例目录）: {p}')
            sys.exit(1)
        case_id = int(parts[-1])
        svc_fault = parts[-2]
        ds_name = next((k for k, v in DS_KEY.items() if rg.DATASET_MAP[v] in p), 'RE1-OB')
        print(f'反推: dataset={ds_name}  svc_fault={svc_fault}  case_id={case_id}')
        ok = run_one(ds_name, svc_fault, case_id)
        sys.exit(0 if ok else 2)

    print('=' * 70)
    print('Granger 补丁冒烟测试（单案例）')
    print('=' * 70)
    print(f'解释器: {sys.executable}')
    print(f'工作目录: {os.getcwd()}')
    print()
    results = [run_one(*c) for c in DEFAULT_CASES]
    ok = [r for r in results if r]
    print('=' * 70)
    if len(ok) == len([c for c in results if c is not None]):
        print('✅ 冒烟通过 —— 此前失败的 MEM / CPU 案例现在可以跑通')
        print('   下一步：备份现有结果后，后台重跑 OB（--resume 只补缺失的 48 例）')
    else:
        print('❌ 冒烟未通过 —— 先不要放开跑，把上面的报错发给开发者')
    print('=' * 70)


if __name__ == '__main__':
    main()
