"""
granger_analysis.py
====================
基于 Granger 因果检验的网络异常传播链分析模块

核心思想：
    若时间序列 X 的历史值能够显著提升对 Y 当前值的预测精度，
    则称 X Granger-causes Y，即 X 在时间上"引导" Y。

四维指标：
    dns_rtt       - DNS 解析往返时延 (ms)
    ping_rtt      - ICMP Ping 往返时延 (ms)
    packet_loss   - 丢包率 (%)
    tcp_connect   - TCP 连接建立时延 (ms)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GrangerResult:
    """单条 Granger 因果检验结果"""
    cause: str          # 因变量名称
    effect: str         # 果变量名称
    lag: int            # 最优滞后阶数
    f_statistic: float  # F 统计量
    p_value: float      # p 值
    is_significant: bool  # 是否显著 (p < 0.05)
    strength: float     # 因果强度 (0-1)


class GrangerAnalyzer:
    """
    轻量级 Granger 因果分析器（不依赖 statsmodels）

    实现原理：
        1. 对两条时序分别建立 OLS 回归（受限模型 vs 非受限模型）
        2. 计算 F 统计量：F = ((RSS_r - RSS_u)/p) / (RSS_u/(T-2p-1))
        3. 用 F 分布近似 p 值（或基于阈值判断显著性）
    """

    METRIC_LABELS = {
        "dns_rtt":     "DNS解析时延",
        "ping_rtt":    "Ping时延",
        "packet_loss": "丢包率",
        "tcp_connect": "TCP连接时延",
    }

    def __init__(self, max_lag: int = 3, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance

    # ------------------------------------------------------------------
    # 内部 OLS 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
        """最小二乘法，返回残差平方和"""
        if X.shape[0] <= X.shape[1]:
            return float("inf")
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ coef
            return float(np.sum(residuals ** 2))
        except np.linalg.LinAlgError:
            return float("inf")

    def _build_lagged_matrix(
        self, series: np.ndarray, lag: int, start: int
    ) -> np.ndarray:
        """构建滞后特征矩阵（含截距项）"""
        n = len(series) - start
        cols = [np.ones(n)]
        for l in range(1, lag + 1):
            cols.append(series[start - l: len(series) - l])
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # F 统计量 → 近似 p 值
    # ------------------------------------------------------------------

    @staticmethod
    def _f_to_pvalue(f: float, df1: int, df2: int) -> float:
        """用 Beta 函数近似 F 分布 p 值（避免依赖 scipy）"""
        if f <= 0 or df1 <= 0 or df2 <= 0:
            return 1.0
        x = df2 / (df2 + df1 * f)
        x = max(0.0, min(1.0, x))
        # 不完全 Beta 函数近似（Continued Fraction）
        a, b = df2 / 2.0, df1 / 2.0
        try:
            log_beta = (
                GrangerAnalyzer._log_gamma(a + b)
                - GrangerAnalyzer._log_gamma(a)
                - GrangerAnalyzer._log_gamma(b)
            )
            log_x = np.log(x + 1e-300) * a + np.log(1 - x + 1e-300) * b
            incomplete = np.exp(log_beta + log_x) * GrangerAnalyzer._cf(x, a, b)
            return float(np.clip(incomplete, 0.0, 1.0))
        except Exception:
            # 粗略估计
            return float(np.exp(-0.5 * f))

    @staticmethod
    def _log_gamma(n: float) -> float:
        """Stirling 近似 ln(Gamma(n))"""
        if n <= 0:
            return 0.0
        if n < 0.5:
            return np.log(np.pi) - np.log(np.sin(np.pi * n)) - GrangerAnalyzer._log_gamma(1 - n)
        result = 0.0
        while n < 7:
            result -= np.log(n)
            n += 1
        result += (n - 0.5) * np.log(n) - n + 0.5 * np.log(2 * np.pi)
        result += 1 / (12 * n) - 1 / (360 * n ** 3)
        return result

    @staticmethod
    def _cf(x: float, a: float, b: float, iters: int = 50) -> float:
        """连分数展开计算不完全 Beta"""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        front = (x ** a) * ((1 - x) ** b) / (a * np.exp(
            GrangerAnalyzer._log_gamma(a) + GrangerAnalyzer._log_gamma(b)
            - GrangerAnalyzer._log_gamma(a + b)
        ))
        # Lentz 算法
        f, C, D = 1.0, 1.0, 0.0
        for m in range(iters):
            for sign in [1, -1]:
                if m == 0 and sign == 1:
                    d = 1.0
                elif sign == 1:
                    d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1 + d * D
                if abs(D) < 1e-30:
                    D = 1e-30
                D = 1 / D
                C = 1 + d / C
                if abs(C) < 1e-30:
                    C = 1e-30
                delta = C * D
                f *= delta
                if abs(delta - 1) < 1e-7:
                    break
        return front * f

    # ------------------------------------------------------------------
    # 单对检验
    # ------------------------------------------------------------------

    def test_pair(
        self, x: np.ndarray, y: np.ndarray, lag: int
    ) -> Tuple[float, float]:
        """
        检验 X Granger-causes Y（给定滞后阶 lag）
        返回 (f_statistic, p_value)
        """
        start = lag
        T = len(y) - start

        # 受限模型：只用 Y 的历史预测 Y
        X_r = self._build_lagged_matrix(y, lag, start)
        y_target = y[start:]
        rss_r = self._ols_rss(X_r, y_target)

        # 非受限模型：用 Y 和 X 的历史预测 Y
        cols_ur = [X_r]
        for l in range(1, lag + 1):
            cols_ur.append(x[start - l: len(x) - l].reshape(-1, 1))
        X_ur = np.hstack(cols_ur)
        rss_u = self._ols_rss(X_ur, y_target)

        if rss_u <= 0 or rss_r <= rss_u:
            return 0.0, 1.0

        f = ((rss_r - rss_u) / lag) / (rss_u / (T - 2 * lag - 1))
        p = self._f_to_pvalue(f, lag, T - 2 * lag - 1)
        return float(f), float(p)

    # ------------------------------------------------------------------
    # 多指标全对检验
    # ------------------------------------------------------------------

    def analyze(self, data: Dict[str, List[float]]) -> List[GrangerResult]:
        """
        对所有指标对 (i -> j) 执行 Granger 检验
        自动选择最优滞后阶数（AIC 最小）

        参数：
            data: {"dns_rtt": [...], "ping_rtt": [...], ...}

        返回：
            GrangerResult 列表，按因果强度降序排列
        """
        keys = list(data.keys())
        arrays = {k: np.array(v, dtype=float) for k, v in data.items()}
        results: List[GrangerResult] = []

        for cause in keys:
            for effect in keys:
                if cause == effect:
                    continue
                best_f, best_p, best_lag = 0.0, 1.0, 1
                for lag in range(1, self.max_lag + 1):
                    if len(arrays[cause]) < 2 * lag + 5:
                        continue
                    f, p = self.test_pair(arrays[cause], arrays[effect], lag)
                    if f > best_f:
                        best_f, best_p, best_lag = f, p, lag

                # 因果强度归一化到 [0, 1]
                strength = float(np.clip(1 - best_p, 0, 1))
                results.append(GrangerResult(
                    cause=cause,
                    effect=effect,
                    lag=best_lag,
                    f_statistic=round(best_f, 4),
                    p_value=round(best_p, 4),
                    is_significant=best_p < self.significance,
                    strength=round(strength, 3),
                ))

        results.sort(key=lambda r: r.strength, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 传播链提取
    # ------------------------------------------------------------------

    def extract_propagation_chain(
        self, results: List[GrangerResult]
    ) -> List[Tuple[str, str, float]]:
        """
        从显著因果关系中提取最长异常传播链
        返回 [(cause_label, effect_label, strength), ...]
        """
        sig = [r for r in results if r.is_significant]
        if not sig:
            return []

        edges: Dict[str, List[Tuple[str, float]]] = {}
        for r in sig:
            edges.setdefault(r.cause, []).append((r.effect, r.strength))

        # 寻找出度最大的起始节点
        start = max(edges, key=lambda k: len(edges[k]))
        chain, visited = [], {start}

        node = start
        for _ in range(10):
            if node not in edges:
                break
            neighbors = [(n, s) for n, s in edges[node] if n not in visited]
            if not neighbors:
                break
            nxt, strength = max(neighbors, key=lambda x: x[1])
            chain.append((
                self.METRIC_LABELS.get(node, node),
                self.METRIC_LABELS.get(nxt, nxt),
                strength,
            ))
            visited.add(nxt)
            node = nxt

        return chain

    # ------------------------------------------------------------------
    # 滑动窗口异常检测（用于实时流）
    # ------------------------------------------------------------------

    @staticmethod
    def detect_anomaly(series: List[float], window: int = 5, sigma: float = 2.0) -> bool:
        """
        Z-score 基础异常检测：最新值是否超出历史均值 ± sigma*std
        """
        if len(series) < window + 1:
            return False
        hist = np.array(series[-(window + 1):-1])
        mu, std = hist.mean(), hist.std()
        if std < 1e-6:
            return False
        z = abs(series[-1] - mu) / std
        return z > sigma


# ==============================================================
# 快速测试（python granger_analysis.py）
# ==============================================================
if __name__ == "__main__":
    import random as rnd
    print("=" * 55)
    print("  Granger 因果分析模块 — 自测")
    print("=" * 55)

    n = 60
    dns = [50 + rnd.gauss(0, 5) for _ in range(n)]
    # ping 跟随 dns（滞后 1 步）
    ping = [dns[max(0, i - 1)] * 0.8 + rnd.gauss(0, 3) for i in range(n)]
    loss = [max(0, ping[i] * 0.01 + rnd.gauss(0, 0.2)) for i in range(n)]
    tcp  = [loss[i] * 5 + ping[i] * 0.5 + rnd.gauss(0, 4) for i in range(n)]

    analyzer = GrangerAnalyzer(max_lag=3)
    results = analyzer.analyze({
        "dns_rtt": dns, "ping_rtt": ping,
        "packet_loss": loss, "tcp_connect": tcp,
    })

    print("\n[显著因果关系]")
    for r in results:
        if r.is_significant:
            print(f"  {r.cause:12s} → {r.effect:12s}  "
                  f"lag={r.lag}  F={r.f_statistic:.2f}  p={r.p_value:.4f}  "
                  f"strength={r.strength:.3f}")

    chain = analyzer.extract_propagation_chain(results)
    print("\n[传播链]")
    for c, e, s in chain:
        print(f"  {c} ──({s:.2f})──► {e}")
    print("=" * 55)
