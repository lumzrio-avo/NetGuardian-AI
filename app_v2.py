"""
app_v2.py  ——  智能网络健康诊断平台 v3.0（优化版）
================================================================
在原有 v2.0 基础上的核心优化：

  ① 四维指标监控（DNS RTT / Ping RTT / 丢包率 / TCP Connect Time）
  ② 集成真实 Granger 因果分析模块（granger_analysis.py）
  ③ 集成网络故障知识库推理（knowledge_base.py）
  ④ 集成诊断引擎（diagnosis_engine.py）
  ⑤ 新增"因果传播图"可视化（无需额外依赖）
  ⑥ 新增多场景演示按钮（DNS故障 / 拥塞 / 链路抖动 / 全面中断）
  ⑦ 健康评分改用加权多指标算法，而非随机值

运行：
    cd demo
    pip install streamlit pandas
    streamlit run app_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime, timedelta

from granger_analysis import GrangerAnalyzer
from knowledge_base import NetworkFaultKnowledgeBase
from diagnosis_engine import DiagnosisEngine, MetricSnapshot

# ================================================================
# 页面配置
# ================================================================
st.set_page_config(
    page_title="智能网络健康诊断平台 v3.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# 自定义 CSS
# ================================================================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E75B6 100%);
        border-radius: 10px; padding: 15px; color: white;
        text-align: center; margin-bottom: 10px;
    }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-label { font-size: 13px; opacity: 0.85; }
    .chain-box {
        background: #f0f4ff; border-left: 4px solid #2E75B6;
        padding: 10px 15px; border-radius: 5px; margin: 5px 0;
        font-size: 14px; color: #1B3A6B;
    }
    .score-critical { color: #d62728; font-size: 48px; font-weight: bold; }
    .score-warning  { color: #ff7f0e; font-size: 48px; font-weight: bold; }
    .score-normal   { color: #2ca02c; font-size: 48px; font-weight: bold; }
    .repair-p1 { color: #d62728; }
    .repair-p2 { color: #ff7f0e; }
    .repair-p3 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 初始化 Session State
# ================================================================
def _init_state():
    if "engine" not in st.session_state:
        st.session_state.engine = DiagnosisEngine(window=10)
    if "history" not in st.session_state:
        st.session_state.history = {
            "dns_rtt": [], "ping_rtt": [],
            "packet_loss": [], "tcp_connect": [],
        }
    if "scenario" not in st.session_state:
        st.session_state.scenario = "normal"
    if "report" not in st.session_state:
        st.session_state.report = None
    if "step" not in st.session_state:
        st.session_state.step = 0

_init_state()

# ================================================================
# 数据生成器（按场景）
# ================================================================
def generate_metrics(scenario: str, step: int) -> MetricSnapshot:
    """根据当前场景生成一组模拟指标"""
    r = random.gauss

    if scenario == "normal":
        dns   = max(10, r(45, 6))
        ping  = max(5,  r(30, 5))
        loss  = max(0,  r(0.1, 0.05))
        tcp   = max(20, r(80, 10))

    elif scenario == "dns_fault":
        # DNS 故障：DNS RTT 异常高，其余正常
        dns   = max(100, r(180, 20))
        ping  = max(5,   r(35, 5))
        loss  = max(0,   r(0.2, 0.1))
        tcp   = max(50,  r(200, 25))  # TCP 因 DNS 延迟而升高

    elif scenario == "congestion":
        # 带宽拥塞：Ping + 丢包 同时高
        dns   = max(20,  r(75, 10))
        ping  = max(80,  r(130, 20))
        loss  = max(2,   r(5, 1))
        tcp   = max(200, r(450, 50))

    elif scenario == "jitter":
        # 链路抖动：Ping 波动大
        phase = step % 20
        ping  = (30 + phase * 8) if phase < 10 else (110 - phase * 5)
        ping  = max(10, ping + r(0, 8))
        dns   = max(20, r(55, 8))
        loss  = max(0,  r(1.5, 0.5))
        tcp   = max(50, ping * 1.8 + r(0, 15))

    elif scenario == "outage":
        # 全面中断
        dns   = max(300, r(500, 50))
        ping  = max(200, r(350, 40))
        loss  = max(20,  r(60, 10))
        tcp   = max(800, r(1200, 100))

    else:
        dns, ping, loss, tcp = 45, 30, 0.1, 80

    return MetricSnapshot(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        dns_rtt=round(dns, 2),
        ping_rtt=round(ping, 2),
        packet_loss=round(max(0, loss), 3),
        tcp_connect=round(tcp, 2),
    )

# ================================================================
# 侧边栏
# ================================================================
st.sidebar.image("https://img.icons8.com/fluency/48/network.png", width=48)
st.sidebar.title("⚙️ 控制面板")

refresh_interval = st.sidebar.slider("刷新间隔(秒)", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 场景切换")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("✅ 正常运行"):
        st.session_state.scenario = "normal"
        st.session_state.engine = DiagnosisEngine(window=10)
        st.session_state.history = {k: [] for k in st.session_state.history}
    if st.button("🌐 带宽拥塞"):
        st.session_state.scenario = "congestion"
        st.session_state.engine = DiagnosisEngine(window=10)
        st.session_state.history = {k: [] for k in st.session_state.history}
with col_s2:
    if st.button("💥 DNS故障"):
        st.session_state.scenario = "dns_fault"
        st.session_state.engine = DiagnosisEngine(window=10)
        st.session_state.history = {k: [] for k in st.session_state.history}
    if st.button("📡 链路抖动"):
        st.session_state.scenario = "jitter"
        st.session_state.engine = DiagnosisEngine(window=10)
        st.session_state.history = {k: [] for k in st.session_state.history}

if st.sidebar.button("🚨 全面中断", use_container_width=True):
    st.session_state.scenario = "outage"
    st.session_state.engine = DiagnosisEngine(window=10)
    st.session_state.history = {k: [] for k in st.session_state.history}

st.sidebar.markdown("---")
st.sidebar.markdown(f"**当前场景:** `{st.session_state.scenario}`")

# ================================================================
# 生成新数据 & 诊断
# ================================================================
snapshot = generate_metrics(st.session_state.scenario, st.session_state.step)
st.session_state.step += 1

# 写入引擎
engine: DiagnosisEngine = st.session_state.engine
engine.ingest(snapshot)

# 更新历史（用于图表，保留最近 30 个点）
for k in st.session_state.history:
    st.session_state.history[k].append(getattr(snapshot, k))
    if len(st.session_state.history[k]) > 30:
        st.session_state.history[k].pop(0)

# 执行诊断（缓冲够 20 个点后）
report = engine.diagnose()
st.session_state.report = report

# ================================================================
# 标题
# ================================================================
st.markdown("## 🛜 智能网络健康诊断平台 v3.0")
st.caption(f"基于 Granger 因果分析 + 知识库推理 | 数据样本: {st.session_state.step} 帧 | "
           f"场景: **{st.session_state.scenario}**")
st.markdown("---")

# ================================================================
# 主体布局：左 1/3 | 右 2/3
# ================================================================
left, right = st.columns([1, 2])

# ----------------------------------------------------------------
# 左栏：评分 + 指标 + 诊断报告
# ----------------------------------------------------------------
with left:
    # 健康评分
    score = report.health_score
    if score < 50:
        score_cls = "score-critical"
        label_color = "🔴"
    elif score < 75:
        score_cls = "score-warning"
        label_color = "🟡"
    else:
        score_cls = "score-normal"
        label_color = "🟢"

    st.markdown(f"### 📊 网络健康评分")
    st.markdown(
        f'<div style="text-align:center">'
        f'<span class="{score_cls}">{score}</span>'
        f'<span style="font-size:24px;color:#888"> / 100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 四维指标卡片
    st.markdown("### 📈 四维关键指标")
    c1, c2 = st.columns(2)
    metrics = report.metrics_summary
    flags = report.anomaly_flags

    def _metric_md(label, val, unit, is_anom):
        color = "#d62728" if is_anom else "#2ca02c"
        icon = "⚠️" if is_anom else "✓"
        return (f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{color}">'
                f'{val:.1f} {unit}</div>'
                f'<div style="font-size:16px">{icon}</div></div>')

    with c1:
        st.markdown(_metric_md("DNS RTT",    metrics.get("dns_rtt", 0),     "ms", flags.get("dns_rtt")), unsafe_allow_html=True)
        st.markdown(_metric_md("丢包率",      metrics.get("packet_loss", 0), "%",  flags.get("packet_loss")), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_md("Ping RTT",   metrics.get("ping_rtt", 0),    "ms", flags.get("ping_rtt")), unsafe_allow_html=True)
        st.markdown(_metric_md("TCP Connect",metrics.get("tcp_connect", 0), "ms", flags.get("tcp_connect")), unsafe_allow_html=True)

    # 状态标签
    st.markdown("### 🚦 诊断结果")
    severity = report.severity
    fault_name = report.matched_fault or "等待数据..."
    if severity == "critical":
        st.error(f"🔴 {fault_name}")
    elif severity == "warning":
        st.warning(f"🟡 {fault_name}")
    else:
        st.success(f"🟢 {fault_name}")

    if report.confidence > 0:
        st.progress(report.confidence, text=f"置信度 {report.confidence:.1%}")

    # 根因
    st.markdown("### 🤖 AI 根因分析")
    st.info(f"**根因：** {report.root_cause}")

    # 修复建议
    if report.repair_actions:
        st.markdown("### 🔧 修复建议")
        for action in report.repair_actions[:3]:
            priority_icons = {1: "🔴 立即", 2: "🟡 尽快", 3: "🔵 观察"}
            tag = priority_icons.get(action.priority, "")
            st.markdown(
                f'<div class="chain-box">{tag}｜{action.action}'
                f'<span style="color:#888;float:right">{action.estimated_time}</span></div>',
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------
# 右栏：图表 + 传播链 + 数据表
# ----------------------------------------------------------------
with right:
    tab1, tab2, tab3 = st.tabs(["📉 实时监控图", "🔗 因果传播链", "📋 历史记录"])

    # --- Tab 1: 实时图表 ---
    with tab1:
        hist = st.session_state.history
        n = len(hist["dns_rtt"])
        if n > 1:
            df_all = pd.DataFrame({
                "DNS RTT (ms)":     hist["dns_rtt"],
                "Ping RTT (ms)":    hist["ping_rtt"],
                "TCP Connect (ms)": hist["tcp_connect"],
            })
            st.markdown("**延迟指标（DNS / Ping / TCP）**")
            st.line_chart(df_all, height=220)

            df_loss = pd.DataFrame({"丢包率 (%)": hist["packet_loss"]})
            st.markdown("**丢包率**")
            st.line_chart(df_loss, height=160, color=["#d62728"])
        else:
            st.info("等待数据积累中...")

    # --- Tab 2: 因果传播链 ---
    with tab2:
        st.markdown("#### Granger 因果分析传播链")
        if report.granger_chain:
            for i, (cause, effect, strength) in enumerate(report.granger_chain):
                bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                st.markdown(
                    f'<div class="chain-box">'
                    f'<b>{i+1}.</b> <span style="color:#1B3A6B">{cause}</span>'
                    f' ──<span style="color:#2E75B6;font-weight:bold">({strength:.2f})</span>──► '
                    f'<span style="color:#d62728">{effect}</span>'
                    f'<br><small style="color:#666">强度: [{bar}] {strength:.1%}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                f"🔄 正在积累数据（当前 {st.session_state.step}/20 帧），"
                "Granger 分析需要至少 20 个样本点。"
            )

        if report.propagation_chain:
            st.markdown("#### 知识库异常传播路径")
            for i, step_desc in enumerate(report.propagation_chain):
                arrow = "▼" if i < len(report.propagation_chain) - 1 else "⊕"
                st.markdown(
                    f'<div class="chain-box">{i+1}. {step_desc} {arrow}</div>',
                    unsafe_allow_html=True,
                )

        if report.significant_causality:
            st.markdown("#### 显著因果关系（p < 0.05）")
            sig_df = pd.DataFrame([{
                "因变量":  GrangerAnalyzer.METRIC_LABELS.get(r.cause, r.cause),
                "果变量":  GrangerAnalyzer.METRIC_LABELS.get(r.effect, r.effect),
                "滞后阶":  r.lag,
                "F统计量": r.f_statistic,
                "p值":     r.p_value,
                "强度":    r.strength,
            } for r in report.significant_causality])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

    # --- Tab 3: 历史记录 ---
    with tab3:
        hist = st.session_state.history
        n = len(hist["dns_rtt"])
        if n > 0:
            now = datetime.now()
            times = [(now - timedelta(seconds=(n - 1 - i) * refresh_interval)
                      ).strftime("%H:%M:%S") for i in range(n)]
            df_hist = pd.DataFrame({
                "时间":          times,
                "DNS RTT (ms)":  [f"{v:.1f}" for v in hist["dns_rtt"]],
                "Ping RTT (ms)": [f"{v:.1f}" for v in hist["ping_rtt"]],
                "丢包率 (%)":    [f"{v:.3f}" for v in hist["packet_loss"]],
                "TCP (ms)":      [f"{v:.1f}" for v in hist["tcp_connect"]],
            })
            st.dataframe(df_hist[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂无历史数据")

# ================================================================
# 底部信息
# ================================================================
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.caption("🤖 **AI Network Diagnosis System v3.0**")
with col_info2:
    st.caption("⚙️ 核心：Granger因果分析 + 知识库推理")
with col_info3:
    st.caption(f"🕒 下次刷新: {refresh_interval}s | 总样本: {st.session_state.step}")

# ================================================================
# 自动刷新
# ================================================================
time.sleep(refresh_interval)
st.rerun()
