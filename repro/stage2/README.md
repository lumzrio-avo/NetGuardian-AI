# 阶段二 · 脚本归档（stage2）

> 归档日期：2026-10-08
> 说明：本目录是阶段二（三方法对比 + 多维鲁棒性 + 拓扑机制分析）新增的脚本与工具。
> 上游快照见 `../`（granger / pc_pcmci / eval / docs 等）。

## 一、数据与重跑工具

| 脚本 | 作用 |
| --- | --- |
| `make_downsampled.py` | 生成降采样数据（每 k 点取 1，`k=1/2/4/10`）。自动按 k 分目录；抽稀不改时间戳，窗口参数需同步除以 k |
| `run_layer2.sh` | 通用降采样重跑（支持 `RE1-OB/SS/TT`，按数据集分发到正确 runner）。用 `DSROOT`/`OUT` 环境变量指定数据与输出根 |
| `run_layer2_ob.sh` | RE1-OB 专用（`run_rca_full_ob.py` 无 `--workers`，此脚本串行调用） |
| `run_layer2_pcmci.sh` | PCMCI 的 SS/TT 降采样重跑（走 `run_pcmci_add_latency.py`，原生 `--workers` 并行；含 TT 基线目录名兼容探测） |
| `run_layer2_granger_shard.sh` | Granger 分片并行（单例太慢，必须分片）：`--shard i/N` 处理第 i 份 |
| `merge_granger_shards.py` | 合并 Granger 分片结果为完整 `eval.csv`（按官方口径重算）。**可在 Windows 侧直接运行** |
| `check_layer2_progress.sh` | 查看分片进度 + 实测速度（近 5 分钟完成 N 例 → 还需 X 小时）+ 内存占用 |

## 二、阶段二分析脚本

| 脚本 | 作用 |
| --- | --- |
| `阶段二_主结果表.py` | 三方法主结果表（方法 × 数据集 × 指标，含分故障） |
| `阶段二_2a_图特征.py` | 从 1125 张邻接矩阵提取图拓扑特征 → `图特征.csv` |
| `阶段二_2b_特征分析.py` | 拓扑特征 × 定位性能的层内相关分析（含 FDR） |
| `阶段二_2c_回归与汇总.py` | 多元回归（numpy 手写 OLS）+ 相关汇总表 |
| `阶段二_3a_图鲁棒性.py` | **Layer 1 图鲁棒性**：随机 / 定向删边扰动 + numpy 幂迭代 PageRank（与 networkx 逐位对拍） |
| `阶段二_3a2_后处理.py` | Layer 1 结果聚合（排除 PCMCI·OB 不可映射项 + 补连续指标） |

## 三、依赖与口径

- 因果发现三方法的 runner 见 `../granger/`、`../pc_pcmci/`
- 统一评估口径见 `../eval/recompute_official_eval.py`、`../pc_pcmci/common_eval.py`
- 阶段二完整结论见仓库内 `docs/`（或工作区 `结果/阶段二分析/`）
