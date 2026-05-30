# 🛜 智能网络健康诊断平台

> 基于 Streamlit 与多源时序分析的家庭网络智能诊断 Demo

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)

## 📌 项目简介

本项目是一个面向家庭网络的智能健康诊断系统，旨在通过多源 DNS 探测与轻量级时序因果分析，解决传统网络诊断工具“只能测通断、不能查根因”的痛点。

系统能够实时监控 DNS 延迟、丢包率等关键指标，并通过 AI 推理引擎自动区分 **DNS 解析异常** 与 **带宽拥塞**，最终以自然语言形式输出可解释的诊断报告。

## ✨ 核心功能

- 🚦 **网络健康评分**：实时计算 0–100 的健康指数
- 📈 **多源时序监控**：同步采集 DNS 延迟与丢包率
- 🤖 **AI 根因分析**：基于规则引擎与统计推断，定位故障源头
- 💥 **模拟攻击演示**：一键触发 DNS 异常场景
- 🔧 **一键修复模拟**：演示系统自愈闭环

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
