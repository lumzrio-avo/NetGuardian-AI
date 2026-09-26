# 基于因果推理的微服务根因分析 —— 复现实验代码

> **快照日期：2026-09-26**
> 对应工作阶段：阶段一（基建与复现）+ 阶段二（方法对比）
> 基准：RCAEval（RE1-OB / RE1-SS / RE1-TT），三个数据集 × 125 案例 = 375 例

---

## 这是什么

本项目复现并横向对比三种因果发现方法在微服务根因分析（RCA）上的表现：

| 方法 | 排序策略 | 说明 |
|---|---|---|
| **Granger** | PageRank | 成对 Granger 因果检验 → 邻接矩阵 → PageRank |
| **PCMCI** | PageRank | 条件独立性检验（tigramite） |
| **PC** | PageRank | 约束型因果发现（causal-learn） |

本仓库是**代码快照**，用于记录当前实现的版本状态。

---

## 目录结构

| 目录 | 内容 |
|---|---|
| `granger/` | Granger 方法的实验脚本：runner、分片并行、结果合并、官方口径评估重算、完整性校验 |
| `pc_pcmci/` | **PC / PCMCI 的最新版**（已统一到官方口径：窗口、alpha、变量筛选、评估规则） |
| `pc_pcmci_orig/` | 组员提供的、**修改前**的 PC / PCMCI 原件（用于对照） |
| `eval/` | 官方口径重算脚本（三方法统一评估 + 自检） |
| `legacy/` | 早期版本的 PCMCI runner（保留作历史参照） |
| `docs/` | 分析文档：三方法对照清单、最终对比表、代码审计报告、阶段计划对照 |

---

## 关键实现说明

### 1. Granger（`granger/run_granger.py`）

- 预处理严格对齐官方 `main.py`（全窗口 normal+anomaly）
- **唯一改动**：给 `grangercausalitytests` 加了**对级 try/except** ——
  官方实现为裸调用，遇到常量/近常量指标对会抛异常，且官方 `main.py` 的异常分支是 `raise e`，
  导致**任一案例出错整个评测任务终止**（补丁前覆盖率仅 OB 77/125、SS 108/125）。
  该补丁把不可计算的指标对记为"无因果边"并继续，是**让方法能跑完**，不改变已有结论
  （OB/SS 重跑排名 250/250 与旧结果一致，TT 重跑 125/125 一致）。
- TT 单例耗时约 3.24 小时（860 余个指标 → 成对检验约 70 万个），故提供 6 路分片脚本。

### 2. PC / PCMCI（`pc_pcmci/`）

相对 `pc_pcmci_orig/` 的修改（已统一到官方口径）：

| 改动 | 内容 |
|---|---|
| 窗口 | 统一为**全窗口**（normal+anomaly），对齐官方 `main.py:213` |
| alpha | PC = **0.05**、PCMCI = **0.20**（各自官方默认值，来源见代码注释） |
| 评估口径 | 抽出 `common_eval.py`：服务级 `-db` 归一 + 去重；指标级精确匹配 |
| 变量筛选 | 抽出 `common_features.py`：删常量列 → 保留 `*_latency` → 方差 Top-N |
| 真值泄漏 | 移除 TT 上的案例级真值列强制保留 |

> ⚠️ `pcmci_module.py` 的算法部分与官方 `graph_construction/pcmci.py` **逻辑等价**（AST 比对），
> **保留了官方的一个已知缺陷**：图构建用 `np.where(p值矩阵)`，会把"非显著"的边全部加入，
> 导致邻接矩阵密度达 0.81~0.91（近乎完全图）。这是**官方上游的问题**，
> 复现阶段保持原样并如实记录，修正留待后续改进阶段。

### 3. 数据不在本仓库

实验数据（375 例 × 3 方法的排名与邻接矩阵）体积约 **2.3 GB**，不适合放入 Git 仓库，另行备份。

---

## 环境

```bash
pip install -r pc_pcmci/requirements.txt
# 官方指定 causal-learn==0.1.3.3；另需 tigramite、networkx、statsmodels
```

## 运行

```bash
# Granger（单数据集）
python granger/run_granger.py --dataset RE1-OB --output <输出目录>

# PC / PCMCI
python pc_pcmci/run_rca_full_ob.py --algo pc     --data-root <RE1-OB路径> --output <输出目录>
python pc_pcmci/run_pc_enhanced_ss.py            --data-root <RE1-SS路径> --output <输出目录>
python pc_pcmci/run_pc_tt.py                     --data-root <RE1-TT路径> --output <输出目录>
python pc_pcmci/run_rca_full_ob.py --algo pcmci  --data-root <RE1-OB路径> --output <输出目录>
python pc_pcmci/run_pcmci.py                     --data-root <RE1-SS路径> --output <输出目录>

# 官方口径重算评估
python eval/recompute_official_eval.py
```

冒烟测试（含依赖自检）：见 `pc_pcmci/smoke_all.sh`。
静态检查（未定义名扫描）：`python pc_pcmci/check_names.py pc_pcmci/*.py`。

---

## 已发现并记录的复现问题（详见 `docs/`）

1. **PCMCI 图构建缺陷**：`np.where(p值矩阵)` 导致近完全图，端到端定位接近随机（官方上游）
2. **Granger 异常处理缺失**：官方裸调用 + `raise e`，任一案例失败即整个任务终止（官方上游）
3. **官方 latency 改名规则在 RE1-SS/TT 失效**：条件只认 OB 的列名形式（官方上游）
4. **官方指标级真值匹配规则在 SS/TT 失效**：`split("_")` 只对 OB 短名有效（官方上游）
5. **官方指标级真值口径两套并存**（`main.py:397` 与 `:413-432`）
6. **官方端到端无 PCMCI 入口**（PCMCI 在官方仅作零件被调用）
7. **高维不可行**：PC 在 SS 全量 228 变量上单例 990 秒且输出全零图，必须缩减变量集

> 综合结论 3+4：RCAEval 的端到端管线实际是**为 RE1-OB 的列名习惯定制**的。

---

## 参考

- RCAEval: <https://github.com/phamquiluan/RCAEval>
- 数据集：RE1-OB / RE1-SS / RE1-TT（Zenodo record 14590730）
