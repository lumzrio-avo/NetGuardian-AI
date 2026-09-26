import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime, timedelta

# =============================
# 页面配置
# =============================
st.set_page_config(page_title="智能网络健康诊断平台", layout="wide")

st.title("🛜 智能网络健康诊断平台")

# =============================
# 侧边栏：系统控制
# =============================
st.sidebar.header("⚙️ 系统控制")
refresh_interval = st.sidebar.slider("数据刷新间隔(秒)", 1, 10, 2)
st.sidebar.write(f"页面每 **{refresh_interval}** 秒自动刷新")

# =============================
# 初始化 Session State（模拟数据）
# =============================
if "latency_data" not in st.session_state:
    st.session_state.latency_data = [random.randint(40, 70) for _ in range(10)]

if "packet_loss_data" not in st.session_state:
    st.session_state.packet_loss_data = [random.uniform(0, 1) for _ in range(10)]

if "health_score" not in st.session_state:
    st.session_state.health_score = 90

if "status" not in st.session_state:
    st.session_state.status = "正常"

# =============================
# 核心逻辑：动态数据生成 & 自动诊断
# =============================

# 1. 动态生成新的模拟数据点
new_latency = st.session_state.latency_data[-1] + random.randint(-8, 8)
new_latency = max(20, min(200, new_latency)) # 限制范围

new_packet_loss = max(0, min(10, st.session_state.packet_loss_data[-1] + random.uniform(-0.5, 0.5)))

# 2. 更新数据队列
st.session_state.latency_data.append(new_latency)
st.session_state.packet_loss_data.append(new_packet_loss)

st.session_state.latency_data.pop(0)
st.session_state.packet_loss_data.pop(0)

# 3. 自动诊断逻辑（根因推断 - 核心修改点）
avg_latency = sum(st.session_state.latency_data[-5:]) / 5
current_packet_loss = st.session_state.packet_loss_data[-1]

if avg_latency > 90 and current_packet_loss < 1.5:
    st.session_state.health_score = random.randint(30, 50)
    st.session_state.status = "严重异常 (DNS)"
elif avg_latency > 90 and current_packet_loss > 2:
    st.session_state.health_score = random.randint(30, 50)
    st.session_state.status = "严重异常 (拥塞)"
elif avg_latency > 70:
    st.session_state.health_score = random.randint(60, 75)
    st.session_state.status = "波动"
else:
    st.session_state.health_score = random.randint(85, 95)
    st.session_state.status = "正常"

# =============================
# 左右布局
# =============================
left, right = st.columns([1, 2])

# =============================
# 左边区域：指标与AI分析
# =============================
with left:
    st.subheader("📊 网络健康评分")
    st.metric(label="当前评分", value=f"{int(st.session_state.health_score)} / 100")

    st.subheader("📈 关键指标")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("DNS延迟", f"{new_latency:.1f} ms")
    with col2:
        st.metric("丢包率", f"{current_packet_loss:.2f}%")

    st.subheader("🚦 当前状态")
    if "严重异常" in st.session_state.status:
        st.error(f"🔴 {st.session_state.status}")
    elif st.session_state.status == "波动":
        st.warning(f"🟡 {st.session_state.status}")
    else:
        st.success(f"🟢 {st.session_state.status}")

    st.subheader("🤖 AI 根因分析")
    
    # 根据状态显示不同的AI推测（这是创新点的核心展示）
    if "DNS" in st.session_state.status:
        st.error("""
        **诊断报告：**
        检测到DNS解析延迟显著升高，但物理链路丢包率极低。
        👉 **推测根因：本地DNS服务器响应缓慢或存在DNS劫持风险。**
        """)
    elif "拥塞" in st.session_state.status:
        st.error("""
        **诊断报告：**
        检测到高延迟伴随高丢包。
        👉 **推测根因：家庭宽带带宽拥塞或光猫硬件故障。**
        """)
    else:
        st.success("""
        **诊断报告：**
        网络连接质量良好，未发现明显性能瓶颈。
        """)

    st.divider()

    # 手动干预按钮（用于演示极端情况）
    if st.button("💥 模拟DNS攻击"):
        st.session_state.latency_data = [random.randint(150, 250) for _ in range(10)]
        st.session_state.packet_loss_data = [random.uniform(0, 0.5) for _ in range(10)]
        st.rerun()

    if st.button("🔧 一键修复"):
        st.session_state.latency_data = [random.randint(40, 60) for _ in range(10)]
        st.session_state.packet_loss_data = [random.uniform(0, 0.5) for _ in range(10)]
        st.success("✅ 已尝试修复，网络状态正在恢复...")
        time.sleep(1.5) # 暂停1.5秒让用户看到修复成功的提示
        st.rerun()

# =============================
# 右边区域：图表与日志
# =============================
with right:
    st.subheader("📉 DNS延迟实时监控")
    
    df_chart = pd.DataFrame({
        "DNS延迟(ms)": st.session_state.latency_data,
        "警戒阈值": [100] * len(st.session_state.latency_data) # 100ms 红线
    })
    
    st.line_chart(df_chart, color=["#FF4B4B", "#808080"])

    st.divider()

    st.subheader("📋 最近监测记录")
    
    now = datetime.now()
    history_df = pd.DataFrame({
        "时间": [(now - timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(5)][::-1],
        "DNS延迟(ms)": [f"{x:.1f}" for x in st.session_state.latency_data[-5:]],
        "丢包率(%)": [f"{x:.2f}" for x in st.session_state.packet_loss_data[-5:]],
        "状态": [st.session_state.status] * 5
    })
    
    st.dataframe(history_df, use_container_width=True)

# =============================
# 底部状态栏 & 自动刷新
# =============================
st.divider()
st.caption(f"🤖 AI Network Diagnosis Demo System v2.0 | 下次刷新: {refresh_interval}s")

# 关键：实现自动刷新，让Demo“活”起来
time.sleep(refresh_interval)
st.rerun()