"""
data_simulator.py
==================
网络指标数据模拟器 & 批量数据生成工具

功能：
    1. 生成四类典型网络故障的模拟时序数据集（CSV）
    2. 用于离线验证 Granger 分析 & 知识库推理
    3. 可独立运行生成演示数据

用法：
    python data_simulator.py                    # 生成所有场景数据集
    python data_simulator.py --scenario dns     # 仅生成 DNS 故障数据
    python data_simulator.py --n 200            # 生成 200 个时间步
"""

import random
import math
import csv
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict


# ================================================================
# 噪声 & 信号工具
# ================================================================

def gaussian(mu: float, sigma: float, lo: float = 0, hi: float = float("inf")) -> float:
    return max(lo, min(hi, random.gauss(mu, sigma)))


def ar1(prev: float, phi: float, mu: float, sigma: float, lo=0, hi=1e6) -> float:
    """AR(1) 自回归过程：模拟时序相关性"""
    return max(lo, min(hi, phi * prev + (1 - phi) * mu + random.gauss(0, sigma)))


# ================================================================
# 场景模拟器
# ================================================================

class ScenarioSimulator:
    """各故障场景的四维指标时序生成器"""

    @staticmethod
    def normal(n: int) -> List[Dict]:
        """正常运行场景"""
        rows = []
        dns, ping, loss, tcp = 45.0, 30.0, 0.1, 80.0
        for i in range(n):
            dns  = ar1(dns,  0.7, 45.0, 4.0,  lo=10)
            ping = ar1(ping, 0.7, 30.0, 3.0,  lo=5)
            loss = ar1(loss, 0.6,  0.1, 0.05, lo=0,   hi=1.0)
            tcp  = ar1(tcp,  0.7, 80.0, 8.0,  lo=20)
            rows.append({"t": i, "dns_rtt": round(dns, 2), "ping_rtt": round(ping, 2),
                         "packet_loss": round(loss, 4), "tcp_connect": round(tcp, 2),
                         "label": "normal"})
        return rows

    @staticmethod
    def dns_fault(n: int) -> List[Dict]:
        """DNS 故障：DNS RTT 在 t=n//3 处突变"""
        rows = []
        dns, ping, loss, tcp = 45.0, 30.0, 0.1, 80.0
        fault_start = n // 3
        for i in range(n):
            if i >= fault_start:
                dns = ar1(dns, 0.7, 180.0, 18.0, lo=100)
                # TCP 跟随 DNS 延迟（滞后约 2 步）
                tcp = ar1(tcp, 0.7, 200.0 if i > fault_start + 2 else 80.0, 20.0, lo=20)
            else:
                dns = ar1(dns, 0.7, 45.0, 4.0, lo=10)
                tcp = ar1(tcp, 0.7, 80.0, 8.0, lo=20)
            ping = ar1(ping, 0.7, 32.0, 3.0, lo=5)      # Ping 不受影响
            loss = ar1(loss, 0.6,  0.15, 0.05, lo=0, hi=0.8)  # 丢包极低
            lbl  = "dns_fault" if i >= fault_start else "normal"
            rows.append({"t": i, "dns_rtt": round(dns, 2), "ping_rtt": round(ping, 2),
                         "packet_loss": round(loss, 4), "tcp_connect": round(tcp, 2),
                         "label": lbl})
        return rows

    @staticmethod
    def congestion(n: int) -> List[Dict]:
        """带宽拥塞：Ping + 丢包同时升高，Ping 领先丢包 1-2 步"""
        rows = []
        dns, ping, loss, tcp = 45.0, 30.0, 0.1, 80.0
        fault_start = n // 3
        for i in range(n):
            if i >= fault_start:
                ping = ar1(ping, 0.8, 130.0, 15.0, lo=80)
                # 丢包跟随 Ping（滞后 2 步）
                lag_ping = rows[i - 2]["ping_rtt"] if i >= fault_start + 2 else ping
                loss_target = max(2.0, (lag_ping - 50) * 0.04)
                loss = ar1(loss, 0.7, loss_target, 0.5, lo=0, hi=20)
                tcp  = ar1(tcp,  0.8, 400.0,  40.0, lo=100)
                dns  = ar1(dns,  0.7,  60.0,   8.0, lo=20)
            else:
                ping = ar1(ping, 0.7, 30.0, 3.0, lo=5)
                loss = ar1(loss, 0.6,  0.1, 0.05, lo=0, hi=1)
                tcp  = ar1(tcp,  0.7, 80.0, 8.0,  lo=20)
                dns  = ar1(dns,  0.7, 45.0, 4.0,  lo=10)
            lbl  = "congestion" if i >= fault_start else "normal"
            rows.append({"t": i, "dns_rtt": round(dns, 2), "ping_rtt": round(ping, 2),
                         "packet_loss": round(loss, 4), "tcp_connect": round(tcp, 2),
                         "label": lbl})
        return rows

    @staticmethod
    def link_jitter(n: int) -> List[Dict]:
        """链路抖动：Ping 周期性波动"""
        rows = []
        dns, ping, loss, tcp = 50.0, 35.0, 0.2, 90.0
        fault_start = n // 3
        for i in range(n):
            if i >= fault_start:
                # 周期 20 步的正弦抖动
                phase = (i - fault_start) % 20
                ping_base = 35 + 80 * math.sin(math.pi * phase / 20) ** 2
                ping = ar1(ping, 0.6, ping_base, 8.0, lo=10)
                loss = ar1(loss, 0.6, 1.8, 0.4, lo=0, hi=8)
                dns  = ar1(dns,  0.7, 52.0, 6.0, lo=20)
                tcp  = ar1(tcp,  0.6, ping * 1.8, 15.0, lo=30)
            else:
                ping = ar1(ping, 0.7, 35.0, 4.0, lo=5)
                loss = ar1(loss, 0.6,  0.2, 0.08, lo=0, hi=1)
                dns  = ar1(dns,  0.7, 50.0, 5.0, lo=10)
                tcp  = ar1(tcp,  0.7, 90.0, 9.0, lo=20)
            lbl  = "jitter" if i >= fault_start else "normal"
            rows.append({"t": i, "dns_rtt": round(dns, 2), "ping_rtt": round(ping, 2),
                         "packet_loss": round(loss, 4), "tcp_connect": round(tcp, 2),
                         "label": lbl})
        return rows


# ================================================================
# CSV 导出
# ================================================================

def save_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ 已生成: {path}  ({len(rows)} 行)")


# ================================================================
# 快速统计摘要
# ================================================================

def summarize(rows: List[Dict]):
    for col in ["dns_rtt", "ping_rtt", "packet_loss", "tcp_connect"]:
        vals = [r[col] for r in rows]
        print(f"  {col:12s}: mean={sum(vals)/len(vals):.2f}  "
              f"min={min(vals):.2f}  max={max(vals):.2f}")


# ================================================================
# 主程序
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="网络指标模拟数据生成器")
    parser.add_argument("--scenario", choices=["all", "normal", "dns", "congestion", "jitter"],
                        default="all", help="生成指定场景数据")
    parser.add_argument("--n", type=int, default=120, help="时间步数")
    parser.add_argument("--outdir", default="demo_data", help="输出目录")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  网络指标模拟数据生成器  (n={args.n})")
    print(f"{'='*55}")

    scenarios = {
        "normal":     ScenarioSimulator.normal,
        "dns":        ScenarioSimulator.dns_fault,
        "congestion": ScenarioSimulator.congestion,
        "jitter":     ScenarioSimulator.link_jitter,
    }

    targets = list(scenarios.keys()) if args.scenario == "all" else [args.scenario]

    for name in targets:
        print(f"\n[场景: {name}]")
        rows = scenarios[name](args.n)
        save_csv(rows, os.path.join(args.outdir, f"metrics_{name}.csv"))
        summarize(rows)

    print(f"\n{'='*55}")
    print(f"  完成！数据已保存至 ./{args.outdir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
