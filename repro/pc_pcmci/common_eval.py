# -*- coding: utf-8 -*-
"""
common_eval.py — 三方法共用的【官方口径】评估函数（单一实现，杜绝多版本漂移）

背景：此前 PC / PCMCI 的 5 个 runner 各自复制了一份 compute_rank_accuracy，
      导致服务级口径、指标级映射三系统互不一致。本模块统一为官方定义。

官方依据（RCAEval 仓库，逐行核对）：
  main.py:380-382   服务级：
      s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
      s_ranks = [old[0]] + [old[i] for i in range(1, len(old)) if old[i] not in old[:i]]
      → 先做 '-db' 归一，再对服务实体序列【去重】
  main.py:394-397   指标级：
      f_ranks = [Node(x.split("_")[0], x.split("_")[1] if "_" in x else "unknown") for x in ranks]
      → 节点名按第一个下划线切成 (服务, 指标)，真值为 (service, fault)
      → 即【精确相等】才算命中；官方**不做** '-db' 归一、**不去重**
  benchmark/evaluation.py   AC@k = answer 是否出现在 ranks[:k]；Avg@k = mean(AC@1..AC@k)

约定：
  * 未命中时 rank 记为 len(nodes) + 1（哨兵值，沿用旧实现，便于与原表对照）
  * 指标映射表与 RE1-OB 口径对齐（disk→diskio），全套方法共用一份
"""

# ============================================================
# 故障类型 → 指标名（官方 main.py:413-432 的"映射版"口径，与 RE1-OB 一致）
# ============================================================
FAULT_TO_METRIC = {
    "cpu":   "cpu",
    "mem":   "mem",
    "delay": "latency",
    "loss":  "latency",
    "disk":  "diskio",
}

DEFAULT_TOP_K = (1, 3, 5)


# ============================================================
# 排名函数（官方口径）
# ============================================================
def service_rank(nodes, service):
    """
    服务级名次（官方 main.py:380 口径）：
      节点实体 = 节点名按第一个 '_' 前的部分，再去掉 '-db'
      对实体序列【去重】（保留首次出现顺序）
      返回真值服务在去重后序列中的 1-based 名次；找不到返回 None
    """
    if not nodes:
        return None
    entities = [n.split("_")[0].replace("-db", "") for n in nodes]
    dedup = [entities[0]] + [e for i, e in enumerate(entities)
                             if i and e not in entities[:i]]
    try:
        return dedup.index(service) + 1
    except ValueError:
        return None


def metric_rank(nodes, service, fault):
    """
    指标级名次（官方 main.py:394 口径）：
      真值节点名 = f"{service}_{FAULT_TO_METRIC[fault]}"
      要求节点名与真值【精确相等】；找不到返回 None

    ⚠️ 注意：RE1-SS / RE1-TT 的节点多为 Prometheus 长名
       （如 carts_container-cpu-usage-seconds-total），
       按官方规则永远无法等于 carts_cpu → 这两个数据集上该指标不可算，
       论文中应标注 N/A（属官方命名约定的既有局限，非本项目引入）。
    """
    target = "%s_%s" % (service, FAULT_TO_METRIC.get(fault, fault))
    for i, n in enumerate(nodes):
        if n == target:
            return i + 1
    return None


# ============================================================
# 主评估函数
# ============================================================
def compute_rank_accuracy(nodes, service, fault, top_k_list=DEFAULT_TOP_K):
    """
    计算单案例的 Svc-AC@k / Mtr-AC@k / Svc_Rank / Mtr_Rank（官方口径）

    参数:
        nodes: list[str]，按 PageRank 降序排列的节点名（完整排名，不是 top-k）
        service: 真值服务名（如 "adservice" / "ts-auth-service"）
        fault:   故障类型（cpu / mem / delay / loss / disk）
        top_k_list: 需要计算的 k

    返回:
        dict，含 Svc-AC@{k} / Mtr-AC@{k} / Svc_Rank / Mtr_Rank
    """
    nodes = list(nodes)
    n = len(nodes)
    sentinel = n + 1

    s_rank = service_rank(nodes, service)
    m_rank = metric_rank(nodes, service, fault)

    metrics = {}
    for k in top_k_list:
        metrics[f"Svc-AC@{k}"] = 1.0 if (s_rank is not None and s_rank <= k) else 0.0
    for k in top_k_list:
        metrics[f"Mtr-AC@{k}"] = 1.0 if (m_rank is not None and m_rank <= k) else 0.0

    metrics["Svc_Rank"] = s_rank if s_rank is not None else sentinel
    metrics["Mtr_Rank"] = m_rank if m_rank is not None else sentinel
    metrics["n_nodes"] = n
    return metrics


# ============================================================
# 自检（直接运行本文件时执行）
# ============================================================
if __name__ == "__main__":
    # 官方 main.py:380 的行为样例：'carts-db_x' 与 'carts_y' 视为同一服务
    demo = ["front-end_mem", "carts-db_cpu-x", "user_latency",
            "carts_container-cpu-user", "carts_latency"]
    print("节点顺序:", demo)
    print("service_rank(carts) =", service_rank(demo, "carts"),
          "  ← 去重后 carts 在 'front-end','carts','user' 里排第 2")
    print("metric_rank(carts, delay) =", metric_rank(demo, "carts", "delay"),
          "  ← carts_latency 精确命中")
    print("metric_rank(carts, cpu)   =", metric_rank(demo, "carts", "cpu"),
          "  ← 无 carts_cpu，返回 None（长名不匹配）")
    print(compute_rank_accuracy(demo, "carts", "delay"))
