# -*- coding: utf-8 -*-
"""pcmci_module.py — PCMCI 图构建（代码逻辑与官方 RCAEval `graph_construction/pcmci.py` 完全一致；
本文件相对官方**仅增加了说明性注释与模块 docstring**，AST 已校验等价）

⚠️ 重要：本文件**故意保留**官方实现的一个缺陷，以保证「忠实复现」。

    _gather_tau() 返回的是 **p 值矩阵**（不是布尔矩阵），而下游用
        np.where(matrix)
    取"非零"位置加边 —— 浮点数组的"非零即真"意味着
        p ≥ 0.0005 的指标对**全部**被加边
    即：得到的图 = **显著关系的补集**（方向正好相反）。

    实测后果：邻接矩阵密度 RE1-OB 0.907 / RE1-SS 0.861 / RE1-TT 0.810
    （对照 PC 仅 0.042 / — / 0.096），PageRank 在近完全图上近乎均匀，
    端到端指标接近随机（Svc-AC@1 ≈ 0.04）。

    官方同样如此使用（`legacy/graph_eval.py:760`、`legacy/rq1.py:265` 直接
    `adj = pcmci(...)` 后当图用；官方 main.py 也没有 pcmci_pagerank 入口），
    因此这是**官方上游的缺陷**，不是本项目引入的。

    ⇒ 复现阶段（阶段二）：保持原样运行，并把该缺陷作为一条独立的复现发现记录。
    ⇒ 修正方案 `np.where(matrix < alpha)` 属**改进**，留到阶段三，另存为「修正版」，
       不得与 Granger / PC 的官方复现结果并列。
"""
import logging

import networkx as nx
import numpy as np
from tigramite import data_processing
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


class _ParCorr(ParCorr):
    """Wrap ParCorr to handle constant"""

    _logger_name = f"{ParCorr.__module__}.{ParCorr.__name__}"

    def _get_single_residuals(
        self,
        array: np.ndarray,
        target_var: int,
        standardize: bool = True,
        return_means: bool = False,
    ) -> np.ndarray:
        y: np.ndarray = array[target_var, :]
        z: np.ndarray = np.copy(array[2:, :])

        # Standardize
        if standardize:
            y -= y.mean()
            std: np.ndarray = y.std()
            if std > 0:
                y /= std

            z -= z.mean(axis=1).reshape(-1, 1)
            std = z.std(axis=1)
            # Skip constant variables
            indexes: np.ndarray = np.where(std)[0]
            z = z[indexes, :] / std[indexes].reshape(-1, 1)
            if np.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, possibly constant array!")

        if z.shape[0] > 0:
            z = z.T
            try:
                beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
                mean = np.dot(z, beta_hat)
                resid = y - mean
            except np.linalg.LinAlgError as err:
                logging.getLogger(self._logger_name).warning(err, exc_info=True)
                resid = y
                mean = None
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid


def _gather_tau(p_matrix: np.ndarray) -> np.ndarray:
    # reason, result, tau
    num, result_num, _ = p_matrix.shape
    assert num == result_num

    link_matrix = []
    for reason in range(num):
        link_matrix.append([min(p_matrix[reason][result]) for result in range(num)])
    return np.array(link_matrix)


def pcmci(data, tau_max=3, alpha=0.2):
    nodes = data.columns.to_list()

    dataframe = data_processing.DataFrame(data.to_numpy())

    m = PCMCI(dataframe=dataframe, cond_ind_test=_ParCorr(significance="analytic"), verbosity=0)
    report = m.run_pcmci(
        tau_max=tau_max,
        pc_alpha=alpha,
        max_conds_dim=None,
    )

    matrix = _gather_tau(report["p_matrix"])

    """
    matrix: if matrix[i, j] is True, i may be one of the causes of j
    nodes: mapping from the matrix indexes to Nodes
    """
    # hmm..
    matrix = np.around(matrix, 3)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    # graph.add_node(data.sli)
    # ⚠️ 【官方实现原样保留，勿改】matrix 是 p 值矩阵，np.where 对浮点是「非零即真」，
    #    因此这里把 p ≥ 0.0005 的指标对全部加边 → 图 = 显著关系的补集（近完全图）。
    #    详见文件头部说明；修正为 `np.where(matrix < alpha)` 属阶段三的改进项。
    graph.add_edges_from((nodes[cause], nodes[effect]) for cause, effect in zip(*np.where(matrix)))

    adj = nx.adjacency_matrix(graph).todense()
    # adj = np.zeros((data.shape[1], data.shape[1]))
    # for n, neighbors in graph.adjacency():
    #     for i in neighbors.keys():
    #         adj[i, n] = 1

    return adj
