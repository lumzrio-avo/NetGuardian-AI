"""
knowledge_base.py
==================
网络故障因果知识库模块

知识库结构：
    每条知识条目描述一种典型故障场景，包含：
    - 触发条件（指标阈值）
    - 异常传播链
    - 根因描述
    - 修复建议（优先级排序）
    - 置信度权重
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class RepairAction:
    """单条修复建议"""
    priority: int       # 1=立即处理, 2=尽快处理, 3=观察
    action: str         # 操作描述
    estimated_time: str # 预估处理时长


@dataclass
class FaultPattern:
    """故障模式条目"""
    fault_id: str
    name: str
    description: str
    # 触发条件：{指标名: (最小值, 最大值)} None 表示不限制
    conditions: Dict[str, Tuple[Optional[float], Optional[float]]]
    # 因果传播链（描述性）
    propagation_chain: List[str]
    root_cause: str
    repair_actions: List[RepairAction]
    severity: str       # "critical" / "warning" / "info"
    confidence: float   # 知识置信度 0-1


class NetworkFaultKnowledgeBase:
    """
    网络故障因果知识库

    包含 7 类典型家庭/办公网络故障场景：
      1. DNS 服务器故障
      2. 带宽拥塞
      3. 物理链路抖动
      4. 服务不可达（TCP 超时）
      5. DNS + 丢包复合故障
      6. 轻微延迟波动（非故障）
      7. 全面网络中断
    """

    def __init__(self):
        self._patterns: List[FaultPattern] = []
        self._build_knowledge()

    # ------------------------------------------------------------------
    # 知识库构建
    # ------------------------------------------------------------------

    def _build_knowledge(self):
        self._patterns = [

            # ① DNS 服务器故障 / DNS 劫持
            FaultPattern(
                fault_id="DNS_FAULT",
                name="DNS服务器故障/劫持",
                description="DNS解析时延显著升高，而链路丢包率和Ping时延保持正常，"
                            "表明底层物理链路健康，故障集中在DNS层面。",
                conditions={
                    "dns_rtt":     (100, None),   # DNS > 100ms
                    "ping_rtt":    (None, 80),     # Ping 仍在正常范围
                    "packet_loss": (None, 1.0),    # 丢包率 < 1%
                },
                propagation_chain=[
                    "DNS RTT 异常升高",
                    "域名解析耗时过长",
                    "TCP 连接建立被动延迟",
                    "用户感知：网页加载慢，但视频流相对正常",
                ],
                root_cause="本地/运营商 DNS 服务器响应过慢，或存在 DNS 劫持、污染风险。",
                repair_actions=[
                    RepairAction(1, "切换 DNS 服务器（推荐：阿里 223.5.5.5 或 腾讯 119.29.29.29）", "2分钟"),
                    RepairAction(1, "刷新本地 DNS 缓存：ipconfig /flushdns", "1分钟"),
                    RepairAction(2, "检查路由器 DNS 配置，避免使用运营商默认 DNS", "5分钟"),
                    RepairAction(3, "若怀疑劫持，可启用 DoH (DNS-over-HTTPS)", "10分钟"),
                ],
                severity="critical",
                confidence=0.91,
            ),

            # ② 带宽拥塞
            FaultPattern(
                fault_id="CONGESTION",
                name="带宽拥塞",
                description="Ping 时延与丢包率同步升高，符合典型的队列溢出特征，"
                            "TCP 连接时延随之增大。",
                conditions={
                    "ping_rtt":    (80, None),   # Ping > 80ms
                    "packet_loss": (2.0, None),  # 丢包 > 2%
                    "tcp_connect": (200, None),  # TCP > 200ms
                },
                propagation_chain=[
                    "上行/下行带宽饱和",
                    "路由器缓冲队列溢出 → 丢包率升高",
                    "TCP 拥塞控制触发 → Ping RTT 升高",
                    "TCP 连接建立超时",
                    "用户感知：视频卡顿、游戏高延迟、下载速度骤降",
                ],
                root_cause="家庭宽带带宽耗尽（大文件下载/上传、P2P 占用），或路由器性能瓶颈。",
                repair_actions=[
                    RepairAction(1, "定位并限制高带宽占用进程（使用任务管理器/资源监视器）", "3分钟"),
                    RepairAction(1, "路由器开启 QoS，优先保障视频会议流量", "5分钟"),
                    RepairAction(2, "重启路由器，清除内存缓存", "3分钟"),
                    RepairAction(3, "联系运营商升级带宽套餐", "1-3天"),
                ],
                severity="critical",
                confidence=0.88,
            ),

            # ③ 物理链路抖动（Wi-Fi 干扰 / 网线接触不良）
            FaultPattern(
                fault_id="LINK_JITTER",
                name="物理链路抖动",
                description="Ping RTT 出现周期性波动，丢包率轻度升高，DNS 相对正常。"
                            "典型的 Wi-Fi 信号干扰或物理接触不良表现。",
                conditions={
                    "ping_rtt":    (50, 200),     # Ping 50-200ms 波动
                    "packet_loss": (0.5, 5.0),    # 轻度丢包
                    "dns_rtt":     (None, 150),   # DNS 相对正常
                },
                propagation_chain=[
                    "物理层信号质量下降（Wi-Fi 干扰 / 网线接触不良）",
                    "数据帧重传率升高 → 丢包率周期性波动",
                    "Ping RTT 出现抖动（方差大）",
                    "用户感知：视频会议时断时续、游戏掉线",
                ],
                root_cause="Wi-Fi 频道干扰（相邻网络使用同一信道）、网线老化或接口松动、"
                           "路由器与终端距离过远。",
                repair_actions=[
                    RepairAction(1, "检查并更换 Wi-Fi 信道（推荐 5GHz 非重叠信道）", "5分钟"),
                    RepairAction(1, "检查网线两端水晶头，重新压接或更换网线", "10分钟"),
                    RepairAction(2, "将终端靠近路由器，或添加 Wi-Fi 中继器", "15分钟"),
                    RepairAction(3, "升级路由器固件，开启自动信道选择", "10分钟"),
                ],
                severity="warning",
                confidence=0.83,
            ),

            # ④ 服务端不可达（TCP 超时）
            FaultPattern(
                fault_id="TCP_TIMEOUT",
                name="目标服务不可达",
                description="TCP 连接时延极高或完全超时，但本地 Ping 和 DNS 均正常，"
                            "说明本地网络健康，问题在目标服务器侧或中间路由。",
                conditions={
                    "tcp_connect": (500, None),   # TCP > 500ms
                    "ping_rtt":    (None, 80),    # 本地 Ping 正常
                    "dns_rtt":     (None, 80),    # DNS 正常
                    "packet_loss": (None, 1.0),   # 丢包正常
                },
                propagation_chain=[
                    "本地网络正常（DNS、Ping 均在阈值内）",
                    "TCP SYN 包发出但无 SYN-ACK 响应",
                    "TCP 连接建立时延 > 500ms 或超时",
                    "用户感知：特定网站/应用无法访问，其他服务正常",
                ],
                root_cause="目标服务器故障、防火墙拦截、CDN 节点异常，或运营商 QoS 限速特定端口。",
                repair_actions=[
                    RepairAction(1, "通过 ping/tracert 确认是否为服务端问题", "2分钟"),
                    RepairAction(1, "尝试更换访问节点（切换 VPN 或代理）", "3分钟"),
                    RepairAction(2, "检查本机防火墙/安全软件是否拦截出站连接", "5分钟"),
                    RepairAction(3, "向目标服务提供商反馈故障", "—"),
                ],
                severity="warning",
                confidence=0.79,
            ),

            # ⑤ 复合故障：DNS + 高丢包
            FaultPattern(
                fault_id="COMPOUND_FAULT",
                name="复合故障（DNS异常+链路丢包）",
                description="DNS 时延和丢包率同时异常，表明故障层次复杂，"
                            "可能源于运营商网络故障或光猫/路由器硬件问题。",
                conditions={
                    "dns_rtt":     (120, None),
                    "packet_loss": (3.0, None),
                },
                propagation_chain=[
                    "运营商侧或光猫硬件异常",
                    "物理链路不稳定 → 丢包率升高",
                    "DNS 查询超时 → DNS RTT 升高",
                    "TCP 连接频繁中断",
                    "用户感知：几乎所有网络服务均受影响",
                ],
                root_cause="运营商网络故障、光猫/ONT 设备故障或宽带账号异常。",
                repair_actions=[
                    RepairAction(1, "重启光猫（断电 30 秒后重新上电）", "5分钟"),
                    RepairAction(1, "检查光猫指示灯，确认 PON 信号正常", "2分钟"),
                    RepairAction(1, "拨打运营商客服（10000/10010/10086）报修", "—"),
                    RepairAction(2, "用手机热点临时替代宽带，确认是否为运营商问题", "5分钟"),
                ],
                severity="critical",
                confidence=0.86,
            ),

            # ⑥ 轻微波动（正常范围内）
            FaultPattern(
                fault_id="NORMAL_FLUCTUATION",
                name="正常波动",
                description="所有指标均在正常范围内，存在轻微随机波动，无需处理。",
                conditions={
                    "dns_rtt":     (None, 80),
                    "ping_rtt":    (None, 60),
                    "packet_loss": (None, 0.5),
                    "tcp_connect": (None, 150),
                },
                propagation_chain=["各指标在正常阈值内波动，无异常传播"],
                root_cause="正常的网络随机抖动，无需干预。",
                repair_actions=[
                    RepairAction(3, "持续观察，若指标持续升高则进一步排查", "—"),
                ],
                severity="info",
                confidence=0.95,
            ),

            # ⑦ 全面中断
            FaultPattern(
                fault_id="FULL_OUTAGE",
                name="网络全面中断",
                description="四维指标全部异常，DNS 超时、Ping 极高、丢包率接近 100%、"
                            "TCP 完全无法建立连接。",
                conditions={
                    "dns_rtt":     (300, None),
                    "ping_rtt":    (200, None),
                    "packet_loss": (20, None),
                    "tcp_connect": (1000, None),
                },
                propagation_chain=[
                    "WAN 接口断开 / PPPoE 认证失败",
                    "所有出站流量中断",
                    "DNS 查询无响应 → 超时",
                    "Ping 极高或全部丢失",
                    "TCP 握手完全失败",
                ],
                root_cause="宽带断线（欠费、故障）、路由器 WAN 口物理断开、PPPoE 认证失败。",
                repair_actions=[
                    RepairAction(1, "检查路由器 WAN 口指示灯是否亮起", "1分钟"),
                    RepairAction(1, "重新拨号（路由器管理界面 → WAN 设置 → 重新连接）", "3分钟"),
                    RepairAction(1, "确认宽带账号是否欠费，登录运营商 App 查看", "2分钟"),
                    RepairAction(2, "重启路由器和光猫", "5分钟"),
                    RepairAction(2, "拨打运营商客服报障", "—"),
                ],
                severity="critical",
                confidence=0.93,
            ),
        ]

    # ------------------------------------------------------------------
    # 匹配接口
    # ------------------------------------------------------------------

    def match(
        self,
        metrics: Dict[str, float],
        granger_chain: Optional[List[Tuple[str, str, float]]] = None,
    ) -> List[Tuple[FaultPattern, float]]:
        """
        根据当前指标值匹配知识库条目

        参数：
            metrics:       {"dns_rtt": 120.0, "ping_rtt": 45.0, ...}
            granger_chain: Granger 分析得到的传播链（可选，用于提升匹配置信度）

        返回：
            [(FaultPattern, 匹配得分), ...]，按得分降序，仅返回得分 > 0 的条目
        """
        scored: List[Tuple[FaultPattern, float]] = []

        for pattern in self._patterns:
            score = self._compute_score(pattern, metrics)
            if score > 0:
                # 若 Granger 传播链与知识库描述吻合，加权提升
                if granger_chain:
                    boost = self._granger_boost(pattern, granger_chain)
                    score = min(1.0, score * (1 + boost * 0.2))
                scored.append((pattern, round(score * pattern.confidence, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_score(
        self, pattern: FaultPattern, metrics: Dict[str, float]
    ) -> float:
        """计算条件匹配得分（满足条件数 / 总条件数）"""
        total = len(pattern.conditions)
        if total == 0:
            return 0.0
        hit = 0
        for metric, (lo, hi) in pattern.conditions.items():
            val = metrics.get(metric)
            if val is None:
                continue
            lo_ok = (lo is None or val >= lo)
            hi_ok = (hi is None or val < hi)
            if lo_ok and hi_ok:
                hit += 1
        return hit / total

    @staticmethod
    def _granger_boost(
        pattern: FaultPattern,
        chain: List[Tuple[str, str, float]],
    ) -> float:
        """Granger 传播链与知识库描述的关键词匹配奖励"""
        chain_text = " ".join(f"{c}{e}" for c, e, _ in chain).lower()
        keywords = {
            "DNS_FAULT":        ["dns", "解析"],
            "CONGESTION":       ["ping", "丢包", "拥塞"],
            "LINK_JITTER":      ["ping", "抖动", "丢包"],
            "TCP_TIMEOUT":      ["tcp", "连接"],
            "COMPOUND_FAULT":   ["dns", "丢包"],
            "NORMAL_FLUCTUATION": [],
            "FULL_OUTAGE":      ["dns", "ping", "tcp"],
        }
        kws = keywords.get(pattern.fault_id, [])
        if not kws:
            return 0.0
        hits = sum(1 for kw in kws if kw in chain_text)
        return hits / len(kws)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_pattern(self, fault_id: str) -> Optional[FaultPattern]:
        for p in self._patterns:
            if p.fault_id == fault_id:
                return p
        return None

    def to_dict(self) -> List[dict]:
        """将知识库序列化为 JSON 可导出格式"""
        result = []
        for p in self._patterns:
            result.append({
                "fault_id": p.fault_id,
                "name": p.name,
                "severity": p.severity,
                "confidence": p.confidence,
                "conditions": {
                    k: {"min": v[0], "max": v[1]}
                    for k, v in p.conditions.items()
                },
                "propagation_chain": p.propagation_chain,
                "root_cause": p.root_cause,
                "repair_actions": [
                    {"priority": a.priority, "action": a.action, "time": a.estimated_time}
                    for a in p.repair_actions
                ],
            })
        return result

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"知识库已导出到: {path}")


# ==============================================================
# 快速测试
# ==============================================================
if __name__ == "__main__":
    kb = NetworkFaultKnowledgeBase()

    print("=" * 55)
    print("  知识库条目概览")
    print("=" * 55)
    for p in kb._patterns:
        print(f"  [{p.severity.upper():8s}] {p.fault_id:20s}  {p.name}")

    # 模拟 DNS 故障场景
    print("\n" + "=" * 55)
    print("  场景匹配测试：DNS 故障")
    print("=" * 55)
    test_metrics = {
        "dns_rtt": 180.0,
        "ping_rtt": 42.0,
        "packet_loss": 0.3,
        "tcp_connect": 230.0,
    }
    matches = kb.match(test_metrics)
    for pattern, score in matches[:3]:
        print(f"  [{score:.3f}] {pattern.name}")
        print(f"         根因：{pattern.root_cause[:60]}...")
        print()

    kb.export_json("knowledge_base_export.json")
