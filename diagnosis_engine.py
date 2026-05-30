"""
diagnosis_engine.py
====================
智能诊断引擎 —— 融合 Granger 因果分析 + 知识库推理

工作流程：
    1. 接收四维指标时序数据
    2. 调用 GrangerAnalyzer 分析异常传播关系
    3. 提取传播链 & 当前均值
    4. 调用 NetworkFaultKnowledgeBase 进行模式匹配
    5. 综合生成 DiagnosisReport（含健康评分、根因、修复建议）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from granger_analysis import GrangerAnalyzer, GrangerResult
from knowledge_base import NetworkFaultKnowledgeBase, FaultPattern, RepairAction


# ==============================================================
# 数据结构
# ==============================================================

@dataclass
class MetricSnapshot:
    """单时间步指标快照"""
    timestamp: str
    dns_rtt: float
    ping_rtt: float
    packet_loss: float
    tcp_connect: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "dns_rtt":     self.dns_rtt,
            "ping_rtt":    self.ping_rtt,
            "packet_loss": self.packet_loss,
            "tcp_connect": self.tcp_connect,
        }


@dataclass
class DiagnosisReport:
    """完整诊断报告"""
    timestamp: str
    health_score: int                            # 0-100
    severity: str                                # critical / warning / info
    matched_fault: Optional[str]                 # 匹配到的故障类型名称
    fault_id: Optional[str]                      # 故障 ID
    root_cause: str                              # 根因描述
    propagation_chain: List[str]                 # 传播链（知识库）
    granger_chain: List[Tuple[str, str, float]]  # Granger 分析传播链
    repair_actions: List[RepairAction]           # 修复建议
    metrics_summary: Dict[str, float]            # 当前均值指标
    anomaly_flags: Dict[str, bool]               # 各指标是否异常
    confidence: float                            # 诊断置信度
    significant_causality: List[GrangerResult]   # 显著因果关系列表


# ==============================================================
# 诊断引擎
# ==============================================================

class DiagnosisEngine:
    """
    智能网络健康诊断引擎

    设计目标：
        - 仅依赖四维关键指标完成推理（轻量化）
        - 将 Granger 因果分析结果注入知识库匹配，提升准确率
        - 输出人类可理解的结构化诊断报告
    """

    # 健康评分基准阈值
    THRESHOLDS = {
        "dns_rtt":     {"ok": 60,  "warn": 100, "crit": 200},
        "ping_rtt":    {"ok": 40,  "warn": 80,  "crit": 150},
        "packet_loss": {"ok": 0.5, "warn": 2.0, "crit": 5.0},
        "tcp_connect": {"ok": 100, "warn": 300, "crit": 800},
    }

    METRIC_WEIGHTS = {
        "dns_rtt":     0.25,
        "ping_rtt":    0.25,
        "packet_loss": 0.30,
        "tcp_connect": 0.20,
    }

    def __init__(self, window: int = 10):
        """
        参数：
            window: 用于 Granger 分析的滑动窗口长度
        """
        self.window = window
        self.analyzer = GrangerAnalyzer(max_lag=3, significance=0.05)
        self.kb = NetworkFaultKnowledgeBase()
        # 历史数据缓冲区
        self._history: Dict[str, List[float]] = {
            k: [] for k in ["dns_rtt", "ping_rtt", "packet_loss", "tcp_connect"]
        }

    # ------------------------------------------------------------------
    # 数据摄入
    # ------------------------------------------------------------------

    def ingest(self, snapshot: MetricSnapshot):
        """追加一个时间步的指标快照到缓冲区"""
        for k, v in snapshot.to_dict().items():
            self._history[k].append(v)
            # 保留最近 window*3 个点
            if len(self._history[k]) > self.window * 3:
                self._history[k].pop(0)

    def ingest_bulk(self, snapshots: List[MetricSnapshot]):
        for s in snapshots:
            self.ingest(s)

    # ------------------------------------------------------------------
    # 核心诊断
    # ------------------------------------------------------------------

    def diagnose(self) -> DiagnosisReport:
        """
        基于当前缓冲区数据执行诊断

        返回完整的 DiagnosisReport
        """
        # 1. 计算当前均值指标（最近 window 步）
        current = {
            k: (sum(v[-self.window:]) / len(v[-self.window:]))
            if v else 0.0
            for k, v in self._history.items()
        }

        # 2. 异常标记
        anomaly_flags = {
            k: (current[k] >= self.THRESHOLDS[k]["warn"])
            for k in current
        }

        # 3. Granger 因果分析（数据足够时）
        granger_results: List[GrangerResult] = []
        granger_chain: List[Tuple[str, str, float]] = []

        min_len = min(len(v) for v in self._history.values())
        if min_len >= 20:
            granger_results = self.analyzer.analyze(self._history)
            granger_chain = self.analyzer.extract_propagation_chain(granger_results)

        # 4. 知识库匹配
        matches = self.kb.match(current, granger_chain)
        top_pattern: Optional[FaultPattern] = None
        match_score = 0.0

        if matches:
            top_pattern, match_score = matches[0]

        # 5. 健康评分计算
        health_score = self._compute_health_score(current)

        # 6. 组装报告
        sig_causality = [r for r in granger_results if r.is_significant]

        if top_pattern:
            report = DiagnosisReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                health_score=health_score,
                severity=top_pattern.severity,
                matched_fault=top_pattern.name,
                fault_id=top_pattern.fault_id,
                root_cause=top_pattern.root_cause,
                propagation_chain=top_pattern.propagation_chain,
                granger_chain=granger_chain,
                repair_actions=sorted(
                    top_pattern.repair_actions, key=lambda a: a.priority
                ),
                metrics_summary={k: round(v, 2) for k, v in current.items()},
                anomaly_flags=anomaly_flags,
                confidence=round(match_score, 3),
                significant_causality=sig_causality,
            )
        else:
            report = DiagnosisReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                health_score=health_score,
                severity="info",
                matched_fault="数据不足，等待更多样本",
                fault_id=None,
                root_cause="缓冲区数据不足，无法完成因果推理，请等待更多监测数据。",
                propagation_chain=[],
                granger_chain=granger_chain,
                repair_actions=[],
                metrics_summary={k: round(v, 2) for k, v in current.items()},
                anomaly_flags=anomaly_flags,
                confidence=0.0,
                significant_causality=sig_causality,
            )

        return report

    # ------------------------------------------------------------------
    # 健康评分计算
    # ------------------------------------------------------------------

    def _compute_health_score(self, metrics: Dict[str, float]) -> int:
        """
        加权健康评分：每个指标线性映射到 [0, 100]，加权求和
        """
        total_score = 0.0
        for metric, weight in self.METRIC_WEIGHTS.items():
            val = metrics.get(metric, 0.0)
            ok  = self.THRESHOLDS[metric]["ok"]
            crit = self.THRESHOLDS[metric]["crit"]
            # 线性映射：val <= ok → 100分；val >= crit → 0分
            if val <= ok:
                sub_score = 100.0
            elif val >= crit:
                sub_score = 0.0
            else:
                sub_score = 100.0 * (crit - val) / (crit - ok)
            total_score += sub_score * weight

        return max(0, min(100, int(total_score)))

    # ------------------------------------------------------------------
    # 格式化输出
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: DiagnosisReport) -> str:
        """生成人类可读的纯文本诊断报告"""
        lines = [
            "=" * 60,
            f"  智能网络健康诊断报告",
            f"  {report.timestamp}",
            "=" * 60,
            f"  健康评分：{report.health_score} / 100",
            f"  严重程度：{report.severity.upper()}",
            f"  诊断置信度：{report.confidence:.1%}",
            "",
            "【当前指标均值】",
            f"  DNS RTT    : {report.metrics_summary.get('dns_rtt', 0):.1f} ms"
            f"  {'[!]' if report.anomaly_flags.get('dns_rtt') else '[OK]'}",
            f"  Ping RTT   : {report.metrics_summary.get('ping_rtt', 0):.1f} ms"
            f"  {'[!]' if report.anomaly_flags.get('ping_rtt') else '[OK]'}",
            f"  Packet Loss: {report.metrics_summary.get('packet_loss', 0):.2f} %"
            f"  {'[!]' if report.anomaly_flags.get('packet_loss') else '[OK]'}",
            f"  TCP Connect: {report.metrics_summary.get('tcp_connect', 0):.1f} ms"
            f"  {'[!]' if report.anomaly_flags.get('tcp_connect') else '[OK]'}",
            "",
            f"【匹配故障类型】  {report.matched_fault or '—'}",
            "",
            "【根因分析】",
            f"  {report.root_cause}",
            "",
        ]

        if report.granger_chain:
            lines.append("【Granger 因果传播链】")
            for c, e, s in report.granger_chain:
                lines.append(f"  {c} ──({s:.2f})──► {e}")
            lines.append("")

        if report.propagation_chain:
            lines.append("【异常传播路径（知识库）】")
            for i, step in enumerate(report.propagation_chain):
                lines.append(f"  {i+1}. {step}")
            lines.append("")

        if report.repair_actions:
            lines.append("【修复建议（按优先级）】")
            for a in report.repair_actions:
                tag = ["", "🔴立即", "🟡尽快", "🔵观察"][min(a.priority, 3)]
                lines.append(f"  {tag} {a.action}  [{a.estimated_time}]")
            lines.append("")

        if report.significant_causality:
            lines.append("【显著因果关系（Granger p<0.05）】")
            for r in report.significant_causality[:5]:
                cn = GrangerAnalyzer.METRIC_LABELS
                lines.append(
                    f"  {cn.get(r.cause, r.cause):8s} → {cn.get(r.effect, r.effect):8s}"
                    f"  lag={r.lag}  F={r.f_statistic:.2f}  p={r.p_value:.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================
# 快速测试
# ==============================================================
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  诊断引擎自测 — 模拟 DNS 故障场景")
    print("=" * 60)

    engine = DiagnosisEngine(window=10)

    # 模拟 40 个时间步：DNS 故障场景
    for i in range(40):
        dns   = 150 + random.gauss(0, 15) if i > 15 else 50 + random.gauss(0, 5)
        ping  = dns * 0.3 + random.gauss(0, 5)
        loss  = max(0, random.gauss(0.2, 0.1))  # 丢包率极低
        tcp   = ping * 1.5 + random.gauss(0, 10)

        engine.ingest(MetricSnapshot(
            timestamp=f"T+{i:02d}",
            dns_rtt=dns, ping_rtt=ping,
            packet_loss=loss, tcp_connect=tcp,
        ))

    report = engine.diagnose()
    print(DiagnosisEngine.format_report(report))
