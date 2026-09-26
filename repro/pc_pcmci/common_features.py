# -*- coding: utf-8 -*-
"""
common_features.py — 三方法共用的【统一变量筛选】（单一实现，消除三系统各一套）

背景：此前 PC / PCMCI 各数据集用了四套不同的筛选规则，导致图规模不可比：
    RE1-OB  : 完全不筛选（只删常量列）             → 44~47 个变量
    RE1-SS  : 按 anomaly score 排序 + 去相关 → 40   （pc_module_enhanced._preselect_features）
    RE1-TT  : 方差 Top-50 + 去相关（且豁免真值列）   → 9~31 个变量
    PCMCI SS: 方差 Top-50 + 保留 latency            → 50 个变量
本模块统一为一套确定性规则。

统一规则（按顺序）：
    1. 删除非特征列（time / time.1 等）
    2. 删除节点级指标（列名第一段以 IP 开头，如 192-168-15-8-9100_node-*）
    3. 删除常量列（std <= eps）
    4. 【强制保留】所有名称含 preserve_patterns（默认 "_latency"）的列
       —— 这是【通用规则】：与具体案例无关、与真值无关，对所有案例一视同仁
    5. 其余列按【方差降序】取前 (max_features - 已保留数) 列
    6. 去冗余：对（第 4 步保留列 + 第 5 步补足列）整体做一次贪心扫描，
       剔除与已保留列 |corr| > corr_threshold（默认 0.95）的列
       —— 保留列排在最前，故延迟列优先胜出
       —— 三系统使用同一阈值、同一顺序
       （2026-09-26：曾放宽到 0.999 又回退，理由是 PCMCI 成本与近共线风险，
         详见下方 DEFAULT_CORR_THRESHOLD 注释）

⚠️ 关于"为什么必须缩规模"（论文中需说明）：
    官方 main.py:303 传 dk_select_useful=False，即**使用全部变量**。
    但实测 PC 在 RE1-SS 的 227~231 个变量上：单例 990 s、125 例**全部输出全零图**；
    PCMCI 是 PC 的超集，只会更严重。⇒ 缩规模是**被迫的偏离**，必须显式报告。
    本模块的作用是让"被迫的缩规模"在三个系统、两个方法之间**保持同一套规则**。

⚠️ 关于真值泄漏：
    本模块**严禁**接收任何依赖案例真值的保留列表（如"当前故障服务的 latency 列"）。
    只接受与案例无关的通用模式。参见 2026-09-25 的 P0-1 修复。
"""

DEFAULT_MAX_FEATURES = 50
# 【2026-09-26 决议】保持 0.95 —— 曾一度改为 0.999，现回退
#   背景：0.95 会把 RE1-SS 的内存类指标（互相 ~0.99 相关）削掉大半，
#         SS 最终约 17 列，而 OB 44 / TT 40，规模偏小。
#   为什么最终**不改**（新查到的两个硬约束）：
#     1) PCMCI 成本随变量规模超线性增长：OB(44 变量) 实测 162s/例、125 例 ≈ 5.6h。
#        SS 变量数若由 17 升到 ~50，单数据集预计从 ~0.5h 涨到 ~7h；
#     2) 阈值放到 0.999 会保留近共线列，有重现「SS 相关矩阵奇异 → PC 输出全零图」
#        的风险 —— 当初加"去冗余"正是为了避开这个问题；
#     3) 跨方法比较的规模差异本就由 Granger(275/761) 主导，SS 17→50 改善有限。
#   ⇒ 复现阶段（阶段二）统一按 0.95 收尾；
#     「变量集敏感性（0.95 vs 0.999）」作为消融变体，留到阶段三/鲁棒性实验一起跑。
#   注：本默认值只影响**未显式传参**的 runner
#       （`run_pc_enhanced_ss.py` 自行钉了 MAX_FEATURES=40 / CORR_THRESHOLD=0.95，
#         其 0.95 与本值一致；40 的上限比 50 更紧，但去冗余先于上限生效，影响可忽略）。
DEFAULT_CORR_THRESHOLD = 0.95
DEFAULT_PRESERVE_PATTERNS = ("_latency",)
DEFAULT_STD_EPS = 1e-10


def _is_node_level(col: str) -> bool:
    """列名第一段是否为 IP（宿主机 node-exporter 指标，如 192-168-15-8-9100_node-cpu-*）"""
    head = col.split("_")[0]
    return head.startswith("192-")


def select_features(
    data,
    max_features: int = DEFAULT_MAX_FEATURES,
    drop_node_metrics: bool = True,
    preserve_patterns=DEFAULT_PRESERVE_PATTERNS,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    std_eps: float = DEFAULT_STD_EPS,
    verbose: bool = False,
):
    """
    统一变量筛选（返回筛选后的 DataFrame；列顺序 = 方差降序 + 强制保留列在前）

    参数:
        data: pd.DataFrame
        max_features: 目标变量数上限
        drop_node_metrics: 是否丢弃节点级指标（192-* 前缀）
        preserve_patterns: 强制保留的列名子串（与案例无关的通用规则）
        corr_threshold: 去冗余的相关系数阈值（|corr| 超过则剔除后者）
        std_eps: 常量列判据
        verbose: 是否打印筛选过程
    """
    n_before = data.shape[1]

    # --- 1) 非特征列（time 等）---
    drop = [c for c in data.columns
            if c.lower() in ("time", "time.1") or c.lower().startswith("time.")]
    if drop:
        data = data.drop(columns=drop, errors="ignore")

    # --- 2) 节点级指标 ---
    if drop_node_metrics:
        node_cols = [c for c in data.columns if _is_node_level(c)]
        if node_cols:
            data = data.drop(columns=node_cols)
            if verbose:
                print("  [FEAT_SEL] dropped %d node-level metrics" % len(node_cols))

    # --- 3) 常量列 ---
    std = data.std(numeric_only=True)
    const_cols = [c for c in data.columns
                  if c in std.index and (std[c] <= std_eps or std[c] != std[c])]
    if const_cols:
        data = data.drop(columns=const_cols)
        if verbose:
            print("  [FEAT_SEL] dropped %d constant cols" % len(const_cols))

    # --- 4) 强制保留列（通用规则，与真值无关）---
    keep = [c for c in data.columns
            if any(p in c for p in (preserve_patterns or ()))]
    keep = [c for c in data.columns if c in set(keep)]        # 保持原列序

    # --- 5) 其余列按方差降序补足 ---
    rest = [c for c in data.columns if c not in set(keep)]
    n_slots = max(max_features - len(keep), 0)
    if len(rest) > n_slots:
        var = data[rest].var().sort_values(ascending=False)
        rest = list(var.index[:n_slots])

    # --- 6) 去冗余（keep 列不再完全豁免）---
    # 【2026-09-26 修正】原实现只对 rest 去冗余、keep(=latency) 列完全豁免，
    # 若两列 latency 近乎共线则两列都会留下 → 相关矩阵奇异 → PC 退化。
    # 现改为对 keep + rest 整体做一遍（keep 在前，故延迟列优先胜出）。
    cols = list(keep) + list(rest)          # keep 优先，rest 已按方差降序
    n_dropped = 0
    if corr_threshold is not None and len(cols) > 1:
        corr = data[cols].corr().abs()
        kept, dropped = [], set()
        for c in cols:
            if c in dropped:
                continue
            kept.append(c)
            for c2 in cols:
                if c2 in dropped or c2 in kept:
                    continue
                try:
                    r = corr.loc[c, c2]
                except KeyError:
                    continue
                if r == r and r > corr_threshold:      # 非 NaN 且超阈值
                    dropped.add(c2)
        n_dropped = len(cols) - len(kept)
        cols = kept

    data = data[cols]

    if verbose:
        print("  [FEAT_SEL] %d -> %d  (强制保留 %d、方差补足 %d、剔除近重复 %d)"
              % (n_before, data.shape[1], len(keep), len(rest), n_dropped))
    return data
