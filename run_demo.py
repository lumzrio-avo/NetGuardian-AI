"""
run_demo.py
============
一键演示脚本：离线运行完整的诊断流程
（不需要启动 Streamlit，直接在终端输出诊断报告）

用法：
    python run_demo.py                   # 交互式菜单
    python run_demo.py --scenario dns    # 直接运行 DNS 故障场景
    python run_demo.py --all             # 运行全部场景
"""

import sys
import os
import random
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from granger_analysis import GrangerAnalyzer
from knowledge_base import NetworkFaultKnowledgeBase
from diagnosis_engine import DiagnosisEngine, MetricSnapshot, DiagnosisEngine
from data_simulator import ScenarioSimulator


SCENARIOS = {
    "1": ("正常运行",   "normal"),
    "2": ("DNS故障",    "dns"),
    "3": ("带宽拥塞",   "congestion"),
    "4": ("链路抖动",   "jitter"),
}

SCENARIO_COLORS = {
    "normal":     "\033[92m",  # 绿
    "dns":        "\033[93m",  # 黄
    "congestion": "\033[91m",  # 红
    "jitter":     "\033[94m",  # 蓝
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def run_scenario(name: str, n_steps: int = 60, verbose: bool = True):
    """运行一个完整的诊断场景"""
    sim_fn = {
        "normal":     ScenarioSimulator.normal,
        "dns":        ScenarioSimulator.dns_fault,
        "congestion": ScenarioSimulator.congestion,
        "jitter":     ScenarioSimulator.link_jitter,
    }[name]

    rows = sim_fn(n_steps)
    engine = DiagnosisEngine(window=10)

    # 逐步喂入数据
    for row in rows:
        engine.ingest(MetricSnapshot(
            timestamp=f"T+{row['t']:03d}",
            dns_rtt=row["dns_rtt"],
            ping_rtt=row["ping_rtt"],
            packet_loss=row["packet_loss"],
            tcp_connect=row["tcp_connect"],
        ))

    # 最终诊断报告
    report = engine.diagnose()
    color  = SCENARIO_COLORS.get(name, "")

    print(f"\n{color}{BOLD}{'='*60}{RESET}")
    print(f"{color}{BOLD}  场景: {name.upper()}  ({n_steps} 步模拟数据){RESET}")
    print(f"{color}{BOLD}{'='*60}{RESET}")
    print(DiagnosisEngine.format_report(report))

    return report


def interactive_menu():
    """交互式菜单"""
    while True:
        print(f"\n{BOLD}{'='*50}{RESET}")
        print(f"{BOLD}  智能网络诊断引擎 — 演示菜单{RESET}")
        print(f"{'='*50}")
        for k, (label, _) in SCENARIOS.items():
            print(f"  [{k}] {label}")
        print("  [5] 运行全部场景")
        print("  [6] 导出知识库 JSON")
        print("  [0] 退出")
        print(f"{'='*50}")

        choice = input("请选择 > ").strip()

        if choice == "0":
            print("再见！")
            break
        elif choice in SCENARIOS:
            _, sc = SCENARIOS[choice]
            run_scenario(sc)
        elif choice == "5":
            for _, (label, sc) in SCENARIOS.items():
                run_scenario(sc)
        elif choice == "6":
            kb = NetworkFaultKnowledgeBase()
            path = os.path.join(os.path.dirname(__file__), "knowledge_base_export.json")
            kb.export_json(path)
        else:
            print("  无效选项，请重试")


def main():
    parser = argparse.ArgumentParser(description="智能网络诊断引擎演示")
    parser.add_argument("--scenario", choices=["normal", "dns", "congestion", "jitter"],
                        help="直接运行指定场景")
    parser.add_argument("--all", action="store_true", help="运行全部场景")
    parser.add_argument("--n", type=int, default=60, help="时间步数")
    args = parser.parse_args()

    if args.all:
        for _, (_, sc) in SCENARIOS.items():
            run_scenario(sc, args.n)
    elif args.scenario:
        run_scenario(args.scenario, args.n)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
