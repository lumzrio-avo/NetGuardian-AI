# PC / PCMCI 代码审计与统一方案

> 审计对象：`NetGuardian-AI-Lynn4913-patch-1/`（组员提供的最新版）
> 审计日期：2026-09-25
> 审计方式：逐文件读源码，行号可复核

---

## 一、代码清单与定位

| 文件 | 行数 | 作用 | 对应数据集 |
|---|---|---|---|
| `run_rca_full_ob.py` | 709 | PC / PCMCI 完整运行（`--algo pc\|pcmci`） | **RE1-OB** |
| `pc_module.py` | 290 | PC 算法核心模块（正常窗口版） | OB / TT 共用 |
| `run_pc_enhanced_ss.py` | 531 | PC Enhanced 运行脚本 | **RE1-SS** |
| `pc_module_enhanced.py` | 331 | PC Enhanced 核心（预筛选 + 全窗口） | SS 专用 |
| `run_pc_tt.py` | 638 | PC 运行脚本 | **RE1-TT** |
| `pcmci_module.py` | 103 | PCMCI 算法核心 | 全部 |
| `run_pcmci.py` | 828 | PCMCI 运行脚本 | **RE1-SS** |
| `PCMCI_module_add_latency.py` | 581 | PCMCI「原 50 + latency」重跑 | SS / TT |

**核对结果**：`pcmci_module.py` 与官方 `RCAEval/graph_construction/pcmci.py` **逐字节一致**（✅ 算法层严格复现）。

---

## 二、完整配置对照表（问题总览）

### PC 侧

| 项 | OB | SS | TT |
|---|---|---|---|
| 脚本 | `run_rca_full_ob.py` | `run_pc_enhanced_ss.py` | `run_pc_tt.py` |
| 模块 | `pc_module.py` | `pc_module_enhanced.py` | `pc_module.py` |
| **窗口** | **正常窗口 480** | **全窗口 721** | **正常窗口 480** |
| 常量列判据所在窗口 | 正常窗口 | 全窗口 | 正常窗口 |
| **alpha** | 0.05 | **0.1** | 0.05 |
| **变量筛选** | 无（只删常量列） | **anomaly 排序 + 去相关 → 40** | **方差 Top-50 + 去相关** |
| **真值列强制保留** | 无 | 无 | **有（案例级）** |
| 变量数 | 44~47 | **恒 40** | 9~31 |
| **Mtr 匹配规则** | 精确 | 精确 | **指标族前缀** |
| Mtr 映射 `mem` | `mem` | `mem` | `container-memory` |
| Mtr 映射 `disk` | `diskio` | `diskio` | `container-fs` |

### PCMCI 侧

| 项 | OB | SS | SS/TT（add_latency） |
|---|---|---|---|
| 脚本 | `run_rca_full_ob.py --algo pcmci` | `run_pcmci.py` | `PCMCI_module_add_latency.py` |
| **窗口** | **全窗口** | **全窗口** | **全窗口** |
| `tau_max` / `alpha` | 3 / 0.05 | 3 / 0.05 | 3 / 0.05 |
| 变量筛选 | 无 → 44~47 | 方差 Top-50 + 保留全部 latency | 旧 50 + 所有服务 latency → 58 / 79 |
| **Mtr 匹配规则** | 精确 | 精确 | 精确 |
| Mtr 映射 `mem` | `mem` | **`memory`** | `mem` |
| Mtr 映射 `disk` | `diskio` | `diskio` | **`disk`** |

### 服务级排名（**五个 runner 一致**）

```python
svc_rank = next(i + 1 for i, n in enumerate(ranks) if n.split("_")[0] == truth_service)
```

即 **不去重、且不做 `.replace("-db", "")` 归一**。
官方口径（`main.py:380`）是「先 `-db` 归一，再去重」。
⇒ 与 Granger 的服务级指标**不可直接比较**（已量化，见第 3.4 节）。

---

## 三、问题清单（按严重度）

### 🔴 P0-1　`run_pc_tt.py` 把真值列强制保留进候选集 —— **真值泄漏**

```python
# run_pc_tt.py:447-455
# === 1.5 特征选择 (强制保留真相指标列) ===
truth_keep = get_truth_metrics(service, fault_type, data.columns.tolist())
data = select_features(..., keep_cols=truth_keep, ...)
```

```python
# run_pc_tt.py:258-271  get_truth_metrics
family = TT_FAULT_TO_FAMILY.get(fault_type, fault_type)
if fault_type in ("delay", "loss"):
    truth = [f"{service}_latency"]                              # 精确真值
else:
    truth = [c for c in metric_names if c.startswith(f"{service}_{family}")]
```

而且 `select_features` 的去相关步骤**也豁免真值列**（`run_pc_tt.py:237-239`，
注释原文：「真相列永远不被剔除」）。

**为什么这是问题**：评估时用的那个真值节点，是**我们人为塞进建图候选集的**，不是算法找出来的。
这会让 TT 的 `Mtr-AC` 系统性虚高 —— 尤其是「节点本来就不在图里 → 必然 not_found」的那类失败被直接抹掉了。

**旁证**：这正是 TT 的 `Mtr-AC@5 = 0.520` 与 SS 的 `0.096` 差一个数量级的直接原因。

**改法**：删掉 `keep_cols=truth_keep`，改为**通用规则**（例如「所有服务的 latency 列一律保留」
或「不做任何保留」），三系统同一套。真值列必须靠算法自己排上去。

---

### 🔴 P0-2　PC 的窗口随数据集变化（正常窗 / 全窗 + alpha 0.05 / 0.1）

| | OB | SS | TT |
|---|---|---|---|
| 窗口 | 正常窗口 480 | 全窗口 721 | 正常窗口 480 |
| alpha | 0.05 | 0.1 | 0.05 |

代码位置：
- 正常窗口：`pc_module.py:161,167`（`normal_end = int(t_fail * 0.8)`）
- 全窗口开关：`run_pc_enhanced_ss.py:60`（`USE_FULL_WINDOW = True`）
  → `pc_module_enhanced.py:252-262`（`if use_full_window: pc_data = data_selected`）

**为什么是问题**：同一种方法在三个数据集上跑了三种设定，横向比精度不成立。

---

### 🔴 P0-3　PC 与 PCMCI 的窗口不可比

`run_pcmci.py`（含 `PCMCI_module_add_latency.py`、OB 的 PCMCI 分支）**一律用全窗口**
（`load_and_preprocess` 里 `pd.concat([normal_df, anomal_df])`），
而 PC 的 OB / TT 用正常窗口。

⇒ 现在 **PC 和 PCMCI 这两条线连内部都不可比**。

---

### 🔴 P0-4　Mtr 映射表五套写法

| 脚本 | delay | loss | disk | cpu | mem |
|---|---|---|---|---|---|
| `run_rca_full_ob.py:67` | latency | latency | diskio | 原样 | 原样 |
| `run_pc_enhanced_ss.py:63` | latency | latency | diskio | 原样 | 原样 |
| `run_pc_tt.py:61` | latency | latency | **container-fs** | **container-cpu** | **container-memory** |
| `run_pcmci.py:71` | latency | latency | diskio | cpu | **memory** |
| `PCMCI_module_add_latency.py:101` | latency | latency | **disk** | cpu | **mem** |

**只有 `delay/loss → latency` 是一致的。** `disk` 有三种写法、`mem` 有三种写法。
⇒ 同一个"指标级准确率"，五个脚本算的不是同一个东西。

---

### 🟠 P1-1　TT 的 Mtr 用「指标族前缀匹配」，与其它数据集语义不同

`run_pc_tt.py:274-290` 对 cpu/mem/disk 用
`{service}_{family}` **前缀**匹配（命中任意 `container-cpu-*` 列即算对），
而 OB / SS 用的是**精确列名**匹配。

前缀匹配天然更宽松（一族好几列，只要有一列进 top-k 就算命中），
⇒ 即便没有 P0-1 的泄漏，TT 的 Mtr 也比别人"容易得分"。

---

### 🟠 P1-2　PC 的 `anomaly_scores` 是死代码

`pc_module.py:231-245`：`anomaly_dict` 算完归一化后**从未被使用**，
只打了一行日志 `已结合异常分数进行辅助评分`。
最终保存的排名就是 `_run_pagerank()` 的纯 PageRank 分数。

⇒ ① 排序实际是纯 PageRank（与官方式一致）；② 日志那句话误导。
（`pc_module_enhanced.py` 里的 `anomaly_scores` 是真的用了 —— 用在**特征预筛选**排序上。两处语义不同。）

---

### 🟠 P1-3　`PCMCI_module_add_latency.py` 依赖旧输出中间产物

`原 50 特征直接取自旧输出 node_names.json` —— 结果依赖前一轮运行的中间文件，
链路一旦丢失或重跑顺序变化就无法复现。建议把「原 50」的筛选规则固化进代码。

---

### 🟡 P2　健壮性 / 工程性

| 项 | 说明 |
|---|---|
| ✅ `pc_module.py` 新增 jitter 重试 | fisherz 相关矩阵奇异时加 `1e-6` 抖动重试 —— 好改动，解决了 PC 全零图 |
| ⚠️ 硬编码绝对路径 | `run_pcmci.py:61-62` 写死 `D:/WorkBuddy_PCMCI_SS/...`，换机器直接失效 |
| ⚠️ 五份 `compute_rank_accuracy` | 同名函数五处各写一遍，是上述所有口径分歧的温床 |
| ⚠️ `run_pc_tt.py` 去相关豁免真值列 | 见 P0-1，放大泄漏 |

---

## 四、解决方案：统一 module 的 5 件事

### A. 窗口 → 全窗口（推荐）

| 理由 |
|---|
| 官方 `main.py:213-214` 就是 `concat(normal, anomaly)` |
| Granger 已用全窗口；PCMCI 三个 runner 也全都是全窗口 → **统一后 PC 与另两条线才可比** |
| 样本从 480 → 721/961/1200，Fisher-z 检验统计功效更高 |
| 正常窗口的收益（避免"故障共因"假边）需要靠**排序阶段用异常分数**来兑现，而 PC 的异常分数是死代码（P1-2）→ 现在只有代价没有收益 |

> 若团队更认同"正常窗口是方法论上更正确的做法"，则必须**同时**把排序改成
> 「PageRank + 异常分数」的联合打分，三系统一致，并且要重跑 Granger / PCMCI 一起对齐。
> 这是方法论选择，**不能只改 PC**。

### B. alpha → 统一 0.05

`run_pc_enhanced_ss.py:57` 的 0.1 是 SS 专用的放宽，其它两系统是 0.05。统一回 0.05。

### C. 变量筛选 → 一套通用规则，且**不含任何案例级真值信息**

推荐规则（三系统 + 两方法一致）：

```
1. 丢弃 time / 非特征列
2. 丢弃节点级指标（列名以 IP 开头，如 192-168-*）
3. 丢弃常量列（std == 0，在【与建模窗口一致】的数据上判）
4. 通用保留规则：所有 {service}_latency 列（与真值无关，对所有案例一视同仁）
5. 剩余名额按方差 Top-N 填满至目标规模
6. 【删除】任何形如 keep_cols=truth_keep 的案例级真值保留
```

⚠️ 规模要按**方法能承受的上限**定，而不是统一成最大值：
- PC 在 SS 的 228 变量上实测 **990 s/例 + 125 例全零图** → 必须缩
- PCMCI 是 PC 的超集，只会更严重
- Granger 是成对检验，成本 ∝ n²，可承受全量

⇒ **「变量规模」本身就是必须写进报告的实验条件**（这一点已在
`结果/三方法对照清单.md` 第 3 节记录）。

### D. 评估口径 → 统一到官方

| 项 | 现状 | 改成 |
|---|---|---|
| 服务级排名 | 不去重、不 `-db` 归一 | `-db` 归一 + 去重（`main.py:380`） |
| 指标级匹配 | 精确 / 前缀 混用 | **一律精确**：节点名 == `{service}_{映射后指标}` |
| Mtr 映射表 | 五套 | **一份共享常量**（建议 `delay/loss→latency`、`disk→diskio`、`cpu→cpu`、`mem→mem`，与 OB 口径对齐） |
| 排名函数 | 五份各写一遍 | 抽成一个 `common_eval.py`，五个 runner 共用 |

### E. 工程性

- 路径改为命令行参数 / 相对路径，去掉 `D:/WorkBuddy_*` 硬编码
- 五份 `compute_rank_accuracy` 合并为一份
- `PCMCI_module_add_latency.py` 的「原 50」筛选规则固化进代码，不再读旧 `node_names.json`

---

## 五、改动代价（好消息：很便宜）

| 数据集 | PC 单例耗时 | 125 例 |
|---|---|---|
| RE1-OB | ≈1.9 s | ≈4 分钟 |
| RE1-SS | ≈4.8 s | ≈10 分钟 |
| RE1-TT | ≈0.34 s | ≈1 分钟 |

**PC 三系统全量重跑 ≈ 15 分钟。** PCMCI 同为秒级。
⇒ 统一口径的成本几乎为零，**唯一的成本是"想清楚要统一成什么"**。

---

## 六、优先级建议

| 优先级 | 事项 | 影响 |
|---|---|---|
| **P0** | 删掉 `run_pc_tt.py` 的案例级真值保留（P0-1） | 直接决定 TT 的 Mtr 数字是否可信 |
| **P0** | 统一窗口（P0-2、P0-3）+ alpha | 决定跨方法/跨系统能否横向比 |
| **P0** | 统一 Mtr 映射表（P0-4） | 决定"指标级准确率"是否是同一个量 |
| **P1** | 统一 Mtr 匹配规则为精确（P1-1） | 消除 TT 的系统性宽松 |
| **P1** | 评估函数抽公共模块 + 官方去重（P0-4 的服务级部分） | 与 Granger 可比 |
| **P2** | 清理死代码 / 硬编码路径 | 可复现性 |

> **一句话**：现有三份 PC 结果、三份 PCMCI 结果，**彼此之间没有一份是口径一致的**。
> 好消息是修复成本约 15 分钟；坏消息是必须一次改齐，否则又会引入新的不一致。


---

## 附录 A：「节点不全」的三层结构（2026-09-25 补充，实测）

指标级（Mtr-AC）要出数，需要**三关全过**：

| | 关一 | 关二 | 关三 |
|---|---|---|---|
| 含义 | 该指标**节点在不在图里** | 节点名**能不能和真值对上** | 它**排不排得进 top-k** |
| 决定因素 | 变量集（筛选规则） | 命名规则 + 评估匹配规则 | 算法能力 |

### 实测：375 例全部节点的指标族覆盖

（统计所有案例的全部节点名，按族归类）

| 方法 · 数据集 | cpu | mem/memory | latency | disk+network+fs | 卡在哪一关 |
|---|---|---|---|---|---|
| PC · OB | 26.8% | 26.4% | 21.7% | 0（+load 24.3%） | **三关全过** → Mtr 可算 |
| PCMCI · OB | 26.4% | 26.2% | 22.0% | 0（+load 24.0%） | **三关全过** |
| **PC · SS** | 32.4% | 42.2% | 5.3% | 17.6% | **关二**（族齐，但长名对不上短真值） |
| **PC · TT** | 3.4% | 89.3% | 2.3% | 5.1% | **关一 + 关二** |
| **PCMCI · SS** | **0%** | 96.0% | **0%** | 3.8% | **关一**（cpu / latency 完全不在图里） |
| **PCMCI · TT** | **0%** | 99.4% | **0%** | 0.6% | **关一** |

> 注：PC · SS 的族覆盖其实是**齐的**（cpu 32%、mem 42%、latency 5%、disk/network/fs 18%），
> 因为它按 **anomaly score** 选变量，故障时 cpu 会飙升所以被选中。
> 它指标级低是**卡在关二（命名）**，不是关一。

### 因此：「变量集统一」方案只能救一部分

第 4 节 C 项提的方案（保留所有 `{service}_latency` 列 + 方差 Top-N）：

- ✅ 能救 **latency 族**（覆盖 delay / loss 的真值）
- ❌ **救不了 cpu / disk** —— 方差 Top-N 里 memory 天生占满，cpu/disk 仍会被挤掉
- ⚠️ 对 **PC · SS 基本无收益**（它族本来就齐，卡在关二）

⇒ 那个方案的定位是「**让三系统在同一个规则下缩规模**」，**不是**「解决节点不全」。

### 要真正打通关一 + 关二，有两条路线

| | 路线 1（推荐） | 路线 2 |
|---|---|---|
| 做法 | 主指标用**服务级（Svc-AC@k）**；指标级在 OB 报、SS/TT 标 **N/A** | 变量集改为「**每服务每指标族取一个代表列**」+ 节点名改为 `{service}_{族}` |
| 指标级 | 不追求可比（如实记录为官方局限 + 输入约束） | **真正可算** |
| 代价 | 无（不需要重跑） | ① 三方法全部重跑 ② 变量集按真值指标族构造，须在论文声明该设计 ③ 图规模 275/761 → ~100/~380，所有数字改变 |
| 定位 | **复现阶段**（忠实复现 + 如实记录） | **阶段三（改进）**，可作创新点 |

**推荐路线 1**：复现阶段主线是「忠实复现官方、如实记录其局限」；
路线 2 属于重新设计实验，更适合放到阶段三。

⚠️ 注意：即便走路线 1，**服务级指标也是「口径一致、输入不同」**
（三方法变量规模差一个数量级 → 图不同 → PageRank 不同），横向比较时必须注明输入规模。


---

## 附录 B：🔴 P0 新发现 —— PCMCI 的图是「近乎完全图」（官方代码缺陷）

**这是本轮最高优先级的问题，影响全部 375 例 PCMCI 结果。**

### 实测：直接数邻接矩阵的非零元素

| 方法 | RE1-OB | RE1-SS | RE1-TT |
|---|---|---|---|
| **PCMCI** | **0.907** | **0.861** | **0.810** |
| PC（对照） | 0.042 | —— | 0.096 |

（PCMCI 旧指标 `Avg n_nodes = 45.536`、`Avg n_edges = 1843.568`；
45.5 × 44.5 ≈ 2025，1843 / 2025 = **0.91**）

⇒ PCMCI 的图不是稀疏因果图，而是「**只缺了少数边的完全图**」。

### 代码根因（`graph_construction/pcmci.py`，本项目副本逐字节一致）

```python
def _gather_tau(p_matrix: np.ndarray) -> np.ndarray:
    # 返回的是【p 值】，不是布尔值
    link_matrix.append([min(p_matrix[reason][result]) for result in range(num)])
    return np.array(link_matrix)

matrix = _gather_tau(report["p_matrix"])
matrix = np.around(matrix, 3)
graph.add_edges_from((nodes[cause], nodes[effect])
                     for cause, effect in zip(*np.where(matrix)))     # ← 问题所在
```

- `np.where(matrix)` 对浮点数组是「**非零即真**」⇒ 只要 min-p **≠ 0** 就加边
- 即 **p ≥ 0.0005 的指标对全部被加边** ⇒ 图 = **显著关系的补集**（方向正好相反）
- 源码注释写的是 `if matrix[i, j] **is True**` ⇒ 原意应是布尔矩阵，**漏了 `matrix < alpha` 比较**

### 官方同样如此使用（排除「本项目用法错误」）

| 位置 | 代码 |
|---|---|
| `legacy/graph_eval.py:760` | `adj = pcmci(pd.DataFrame(np_data))` → 直接 `adj2generalgraph(adj)` |
| `legacy/rq1.py:265` | 同上 |
| 官方 `main.py` | **无 `pcmci_pagerank`** ⇒ 官方论文从未评测 PCMCI 端到端 |

⇒ 官方把近完全图**直接当作因果图**使用，未做任何阈值处理。

### 影响

- PCMCI 的 PageRank 运行在 ~85% 密度的图上 ⇒ 分数近乎均匀 ⇒ **排名接近随机**
- 这解释了 PCMCI 长期偏低的表现（Svc-AC@1 ≈ 0.04，Mtr-AC ≈ 0）
- 结论：**问题不在「PCMCI 对某类故障不敏感」，也不在变量集，而在图构建这一步**

### 两种处理方式（待组内决定）

| | A. 保持官方原样 | B. 修正后纳入 |
|---|---|---|
| 做法 | 照官方实现运行 | 将 `np.where(matrix)` 改为 `np.where(matrix < alpha)` |
| 产出 | PCMCI 指标约等于随机 | 得到真正稀疏的 PCMCI 图 |
| 代价 | 0 | 需重跑 PCMCI（OB/SS/TT 合计约 15 分钟）＋标注为「修正版」 |
| 性质 | 忠实复现 | 偏离官方，需在论文方法一节披露 |

建议：**A + B 各出一版**（A 作为复现结果如实记录，B 作为修正版用于横向比较），
或在论文中把该缺陷作为一条独立的复现发现陈述。

---

## 附录 C：2026-09-26 进度快照

### 已完成（代码层改动，均在 `pc_pcmci_fixed/`）

| # | 事项 | 状态 |
|---|---|---|
| 1 | PC · TT 案例级真值列保留（真值泄漏） | ✅ 已删除，改为通用 `*_latency` 规则 |
| 2 | PC 窗口三系统不一致 | ✅ 统一为**全窗口**（对齐 `main.py:213` 与 Granger/PCMCI） |
| 3 | alpha 偏离官方 | ✅ PC = **0.05**、PCMCI = **0.20**（各自官方默认值） |
| 4 | PC / PCMCI 缺官方服务名去重 | ✅ 五个 runner 全部改接 `common_eval` |
| 5 | Mtr 映射表五套写法 | ✅ 统一为 `common_eval.FAULT_TO_METRIC` |
| 6 | PC · TT 指标级用「指标族前缀匹配」 | ✅ 改回官方精确匹配 |
| 7 | 变量筛选四套规则 | ✅ 统一为 `common_features.select_features` |
| 8 | 五份重复的 `compute_rank_accuracy` | ✅ 合并为一份 |
| 9 | 新增两个共享模块 | ✅ `common_eval.py` / `common_features.py` |

### 待办（按优先级）

| 优先级 | 事项 | 阻塞于 |
|---|---|---|
| **P0** | 确定 PCMCI 处理方案（附录 B：A / B / A+B） | 组内拍板 |
| **P0** | 定位冒烟中 3 个「无日志」配置（PC·OB / PCMCI·SS / PCMCI·TT） | 需要 `run_err.txt` |
| **P1** | 确认 PC·SS 变量数 17 是否偏小（去冗余阈值 0.95 可能过严） | 组内拍板 |
| **P1** | 全量重跑 PC + PCMCI（换新输出目录） | 上述 P0/P1 定案后 |
| **P1** | 重跑后按官方口径重算评估、更新三方法对比表 | 全量重跑完成 |
| **P2** | 抽公共评估代码进正式仓库、清理硬编码路径 | 可随时 |

### 不受影响（无需改动）

- `pcmci_module.py` 的算法部分与官方**逐字节一致**（sha256 `ac902c77d81faf39`）——
  附录 B 的问题是**官方上游的**，不是本项目引入的
- Granger 侧（375 例 + 375 邻接矩阵 + 官方口径评估）已完成并验证，不再改动

### 附录 C 补充（同日）—— 3 个「无日志」配置已定位并修复

| 配置 | 真实原因 | 状态 |
|---|---|---|
| PC · RE1-OB | `run_rca_full_ob.py` 缺 `--normal-only` 参数定义，`args.normal_only` 抛 `AttributeError` | ✅ 已补参数 |
| PCMCI · RE1-SS | **其实跑成功了**（19.5 节点 / 318.5 边），只是冒烟脚本读不到 `run_pcmci.py` 的日志格式 | ✅ 无问题 |
| PCMCI · RE1-TT | 同上（40 节点 / 1385 边） | ✅ 无问题 |

**同时修掉一个掩盖机制**：`run_rca_full_ob.py` 原实现在异常时生成「随机打乱的默认排名」并计入汇总，
使崩溃案例伪装成正常结果（`n_edges=0` 却带 `Svc-AC` 分数）——这正是「OB 输出空图」被误判两次的原因。
现改为：失败案例不进入评估，单独写 `failed_cases.csv` 并在汇总时显式告警。

### 附录 B 决策（2026-09-26）—— 复现阶段只做 A，B 推迟到阶段三

**定案：复现阶段（阶段二）按 A 处理 —— 保持官方实现原样，把缺陷作为一条复现发现如实记录。
修正方案 B（`np.where(matrix < alpha)`）留到阶段三（改进）作为候选创新点，另存为「修正版」。**

理由：
1. B 改的是**官方算法实现本身**，其产出属于「我们改进后的方法」，
   不能与 Granger / PC 的官方复现结果并列比较 —— 否则复现阶段的口径就不干净了；
2. A 的成本几乎为零：PCMCI 单例约 OB 2s / SS 5s / TT 1s，375 例合计十几分钟；
3. **「官方 PCMCI 端到端接近随机」本身就有复现价值** ——
   直接作为「复现过程中的发现」写入报告，比一个被修好的数字更有说服力；
4. 「修正高维因果发现的预处理/图构建缺陷」正好属于阶段三的主题，天然可作创新点。

**代码处理**：`pc_pcmci_fixed/pcmci_module.py` **功能不变**，仅在文件头与
`np.where(matrix)` 处增加说明性注释（防止后人误"修"）。已用 AST 比对确认代码逻辑
与官方 `graph_construction/pcmci.py` 完全等价（差异仅为注释与模块 docstring）。
