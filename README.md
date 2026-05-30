# 智能网络健康诊断平台 v3.0

基于 **Granger 因果分析** + **网络故障知识库推理** 的轻量化智能诊断系统

## 项目结构

```
demo/
├── app_v2.py             # Streamlit 主界面（v3.0 优化版）
├── granger_analysis.py   # Granger 因果检验核心算法
├── knowledge_base.py     # 网络故障因果知识库
├── diagnosis_engine.py   # 智能诊断推理引擎
├── data_simulator.py     # 时序数据模拟器（批量生成CSV）
├── run_demo.py           # 一键终端演示脚本
└── requirements.txt      # 依赖清单
```

## 快速启动

### 安装依赖

```bash
pip install streamlit pandas
```

> 本项目**不依赖** statsmodels / scipy，所有算法均自主实现。

### 启动 Streamlit 界面

```bash
cd demo
streamlit run app_v2.py
```

### 终端演示（无需浏览器）

```bash
# 交互式菜单
python run_demo.py

# 直接运行 DNS 故障场景
python run_demo.py --scenario dns

# 运行全部场景
python run_demo.py --all
```

### 生成模拟数据集

```bash
python data_simulator.py              # 生成四类场景 CSV
python data_simulator.py --n 200      # 200 时间步
```

---

## 核心优化说明（相对 v2.0）

| 模块 | v2.0 | v3.0 优化 |
|------|------|-----------|
| 指标维度 | 2维（DNS + 丢包） | **4维**（DNS/Ping/丢包/TCP） |
| 健康评分 | 随机值 | **加权多指标算法** |
| 诊断逻辑 | 硬编码 if-else | **知识库模式匹配** |
| 传播分析 | 无 | **Granger 因果检验**（F检验 + p值） |
| 场景支持 | 2种 | **5种**（正常/DNS/拥塞/抖动/中断） |
| 依赖 | pandas | **零外部算法依赖** |

## 四维指标说明

| 指标 | 正常阈值 | 预警阈值 | 危险阈值 |
|------|---------|---------|---------|
| DNS RTT | < 60ms | 60-100ms | > 100ms |
| Ping RTT | < 40ms | 40-80ms | > 80ms |
| 丢包率 | < 0.5% | 0.5-2% | > 2% |
| TCP Connect | < 100ms | 100-300ms | > 300ms |

## Granger 因果分析原理

若时序 **X** 的历史值能显著提升对 **Y** 当前值的预测精度（OLS F检验），
则判定 **X Granger-causes Y**，即 X 在时间上"引导" Y。

典型传播链（DNS 故障场景）：
```
DNS RTT 升高 ──(0.87)──► TCP Connect 升高
```

## 知识库故障类型

1. DNS 服务器故障/劫持
2. 带宽拥塞
3. 物理链路抖动
4. 目标服务不可达
5. 复合故障（DNS + 丢包）
6. 正常波动（无故障）
7. 网络全面中断
