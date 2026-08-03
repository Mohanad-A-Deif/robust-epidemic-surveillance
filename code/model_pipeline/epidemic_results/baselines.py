"""External and internal comparison baselines."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import ndimage, optimize

from .proposed_model import InferenceResult

EPS = 1e-12


def _received(messages: pd.DataFrame) -> pd.DataFrame:
    frame = messages.copy()
    if "received" in frame:
        if frame["received"].dtype != bool:
            frame["received"] = frame["received"].astype(str).str.lower().isin(["true", "1"])
        frame = frame[frame["received"]]
    for col in ["arrival_index", "generation_index", "observed_value", "node_id", "delay"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["node_id", "observed_value"]).copy()


def interpolate_matrix(grid: np.ndarray) -> np.ndarray:
    result = np.asarray(grid, dtype=float).copy()
    n, t = result.shape
    index = np.arange(t)
    for i in range(n):
        valid = np.isfinite(result[i])
        if not valid.any():
            result[i] = 0.0
        elif valid.sum() == 1:
            result[i] = result[i, valid][0]
        else:
            result[i] = np.interp(index, index[valid], result[i, valid])
    return np.maximum(result, 0.0)


def aggregate_on_index(messages: pd.DataFrame, n_nodes: int, n_time: int, index_column: str) -> np.ndarray:
    grid = np.full((n_nodes, n_time), np.nan)
    frame = _received(messages)
    frame[index_column] = pd.to_numeric(frame[index_column], errors="coerce")
    frame = frame.dropna(subset=[index_column])
    grouped = frame.groupby(["node_id", index_column], as_index=False)["observed_value"].mean()
    for row in grouped.itertuples(index=False):
        node = int(row.node_id)
        time_index = int(getattr(row, index_column))
        if 0 <= node < n_nodes and 0 <= time_index < n_time:
            grid[node, time_index] = float(row.observed_value)
    return grid


def estimate_graph_from_states(x: np.ndarray, beta: float = 0.12, gamma: float = 0.08, row_sum_cap: float = 2.5, ridge: float = 1e-4, candidate_mask: np.ndarray | None = None) -> np.ndarray:
    """Estimate a nonnegative directed graph from a fixed state trajectory."""
    x = np.asarray(x, dtype=float)
    n, t_count = x.shape
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = [j for j in range(n) if j != i and (candidate_mask is None or bool(candidate_mask[i, j]))]
        if not candidates or t_count < 3:
            continue
        features = np.column_stack([beta * (x[j, :-1] - x[i, :-1]) for j in candidates])
        target = x[i, 1:] - (1.0 - gamma) * x[i, :-1]
        features_aug = np.vstack([features, np.sqrt(ridge) * np.eye(len(candidates))])
        target_aug = np.concatenate([target, np.zeros(len(candidates))])
        coefficients, _ = optimize.nnls(features_aug, target_aug)
        total = coefficients.sum()
        if total > row_sum_cap:
            coefficients *= row_sum_cap / total
        w[i, candidates] = coefficients
    np.fill_diagonal(w, 0.0)
    return w


def _result(x_hat: np.ndarray, w_hat: np.ndarray, runtime: float, memory: float, diagnostics: dict[str, Any] | None = None) -> InferenceResult:
    return InferenceResult(
        x_hat=x_hat,
        w_hat=w_hat,
        posterior=[],
        feasible_delays=[],
        predicted_delays=np.array([], dtype=float),
        objective_history=[],
        normalized_objective_history=[],
        x_change_history=[],
        w_change_history=[],
        runtime_seconds=float(runtime),
        peak_memory_mb=float(memory),
        iterations=1,
        converged=True,
        diagnostics=diagnostics or {},
    )


@dataclass
class BaselineConfig:
    beta: float = 0.12
    gamma: float = 0.08
    row_sum_cap: float = 2.5
    candidate_mask: np.ndarray | None = None


class ArrivalAlignedInterpolation:
    name = "Arrival interpolation"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        x = interpolate_matrix(aggregate_on_index(messages, n_nodes, n_time, "arrival_index"))
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20)


class OracleTimestampInterpolation:
    name = "Known-delay oracle"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        x = interpolate_matrix(aggregate_on_index(messages, n_nodes, n_time, "generation_index"))
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20)


class DelayBackprojectionNowcast:
    """Distribute each arrival backward using the assumed delay PMF."""
    name = "Delay backprojection"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, max_delay: int, delay_pmf_by_slice: dict[str, np.ndarray] | None = None, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        frame = _received(messages)
        numerator = np.zeros((n_nodes, n_time), dtype=float)
        denominator = np.zeros((n_nodes, n_time), dtype=float)
        for row in frame.itertuples(index=False):
            node, arrival, value = int(row.node_id), int(row.arrival_index), float(row.observed_value)
            support = np.arange(0, min(max_delay, arrival) + 1)
            slice_id = str(getattr(row, "slice_id", "default"))
            if delay_pmf_by_slice and slice_id in delay_pmf_by_slice:
                pmf = np.asarray(delay_pmf_by_slice[slice_id], dtype=float)
            elif delay_pmf_by_slice and "default" in delay_pmf_by_slice:
                pmf = np.asarray(delay_pmf_by_slice["default"], dtype=float)
            else:
                pmf = np.ones(max_delay + 1, dtype=float)
            if len(pmf) < max_delay + 1:
                pmf = np.pad(pmf, (0, max_delay + 1 - len(pmf)))
            weights = np.maximum(pmf[support], 0.0)
            weights /= max(weights.sum(), EPS)
            indices = arrival - support
            numerator[node, indices] += weights * value
            denominator[node, indices] += weights
        grid = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > EPS)
        x = interpolate_matrix(grid)
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20)


class RobustMedianSmoother:
    name = "Robust median smoother"

    def __init__(self, window: int = 5, config: BaselineConfig | None = None):
        self.window = int(window) if int(window) % 2 == 1 else int(window) + 1
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        x = interpolate_matrix(aggregate_on_index(messages, n_nodes, n_time, "arrival_index"))
        x = ndimage.median_filter(x, size=(1, self.window), mode="nearest")
        x = np.maximum(x, 0.0)
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20)


class LowRankMatrixCompletion:
    """Iterative singular-value thresholding on the arrival-aligned matrix."""
    name = "Robust low-rank completion"

    def __init__(self, rank: int = 3, max_iter: int = 100, tol: float = 1e-5, config: BaselineConfig | None = None):
        self.rank = rank; self.max_iter = max_iter; self.tol = tol
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        observed = aggregate_on_index(messages, n_nodes, n_time, "arrival_index")
        mask = np.isfinite(observed)
        x = interpolate_matrix(observed)
        for iteration in range(self.max_iter):
            previous = x.copy()
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            r = min(self.rank, len(s))
            reconstructed = (u[:, :r] * s[:r]) @ vt[:r]
            x = reconstructed
            x[mask] = observed[mask]
            x = np.maximum(x, 0.0)
            if np.linalg.norm(x - previous) / max(np.linalg.norm(previous), 1.0) < self.tol:
                break
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20, {"iterations": iteration + 1})


class GraphTemporalSmoother:
    name = "Graph-temporal reconstruction"

    def __init__(self, adjacency: np.ndarray, temporal_weight: float = 0.25, graph_weight: float = 0.25, iterations: int = 40, config: BaselineConfig | None = None):
        self.adjacency = np.asarray(adjacency, dtype=float)
        self.temporal_weight = temporal_weight
        self.graph_weight = graph_weight
        self.iterations = iterations
        self.config = config or BaselineConfig(candidate_mask=self.adjacency > 0)

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        observed = aggregate_on_index(messages, n_nodes, n_time, "arrival_index")
        mask = np.isfinite(observed)
        x = interpolate_matrix(observed)
        adjacency = 0.5 * (self.adjacency + self.adjacency.T)
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency
        for _ in range(self.iterations):
            temporal = x.copy()
            temporal[:, 1:-1] = 0.5 * (x[:, :-2] + x[:, 2:])
            graph_smoothed = x - self.graph_weight * (laplacian @ x)
            proposal = (1 - self.temporal_weight) * graph_smoothed + self.temporal_weight * temporal
            proposal[mask] = 0.7 * observed[mask] + 0.3 * proposal[mask]
            x = np.maximum(proposal, 0.0)
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20)


class KalmanRTSSmoother:
    """Independent local-level Kalman filter followed by RTS smoothing.

    This is an internal standard implementation, not a reproduction of a
    specific published epidemic-nowcasting package.
    """
    name = "Kalman/RTS smoother (internal)"

    def __init__(self, process_variance: float = 0.03, observation_variance: float = 0.15, config: BaselineConfig | None = None):
        self.process_variance = float(process_variance)
        self.observation_variance = float(observation_variance)
        self.config = config or BaselineConfig()

    @staticmethod
    def _smooth_series(observations: np.ndarray, q: float, r: float) -> np.ndarray:
        y = np.asarray(observations, dtype=float)
        t_count = len(y)
        m_f = np.zeros(t_count)
        p_f = np.zeros(t_count)
        finite = np.flatnonzero(np.isfinite(y))
        m = float(y[finite[0]]) if len(finite) else 0.0
        p = 1.0
        for t in range(t_count):
            m_pred, p_pred = m, p + q
            if np.isfinite(y[t]):
                k = p_pred / max(p_pred + r, EPS)
                m = m_pred + k * (y[t] - m_pred)
                p = (1.0 - k) * p_pred
            else:
                m, p = m_pred, p_pred
            m_f[t], p_f[t] = m, max(p, EPS)
        m_s = m_f.copy()
        p_s = p_f.copy()
        for t in range(t_count - 2, -1, -1):
            p_pred_next = p_f[t] + q
            gain = p_f[t] / max(p_pred_next, EPS)
            m_s[t] = m_f[t] + gain * (m_s[t + 1] - m_f[t])
            p_s[t] = max(p_f[t] + gain * gain * (p_s[t + 1] - p_pred_next), EPS)
        return np.maximum(m_s, 0.0)

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        observed = aggregate_on_index(messages, n_nodes, n_time, "arrival_index")
        x = np.vstack([
            self._smooth_series(observed[i], self.process_variance, self.observation_variance)
            for i in range(n_nodes)
        ])
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20,
                       {"implementation": "internal local-level Kalman filter plus RTS smoother"})


class DelayAwareAugmentedStateSpace:
    """Internal delay-aware state-space approximation.

    Reports are first probabilistically backprojected over feasible generation
    times, then an RTS smoother is applied to the resulting weighted series.
    This baseline is intentionally labelled as an internal approximation.
    """
    name = "Delay-aware state-space (internal)"

    def __init__(self, process_variance: float = 0.03, observation_variance: float = 0.15, config: BaselineConfig | None = None):
        self.process_variance = float(process_variance)
        self.observation_variance = float(observation_variance)
        self.config = config or BaselineConfig()

    def fit(self, messages: pd.DataFrame, n_nodes: int, n_time: int, max_delay: int,
            delay_pmf_by_slice: dict[str, np.ndarray] | None = None, **_: Any) -> InferenceResult:
        process = psutil.Process(); m0 = process.memory_info().rss; start = time.perf_counter()
        frame = _received(messages)
        numerator = np.zeros((n_nodes, n_time), dtype=float)
        denominator = np.zeros((n_nodes, n_time), dtype=float)
        for row in frame.itertuples(index=False):
            node, arrival, value = int(row.node_id), int(row.arrival_index), float(row.observed_value)
            support = np.arange(0, min(max_delay, arrival) + 1)
            slice_id = str(getattr(row, "slice_id", "default"))
            if delay_pmf_by_slice and slice_id in delay_pmf_by_slice:
                pmf = np.asarray(delay_pmf_by_slice[slice_id], dtype=float)
            elif delay_pmf_by_slice and "default" in delay_pmf_by_slice:
                pmf = np.asarray(delay_pmf_by_slice["default"], dtype=float)
            else:
                pmf = np.ones(max_delay + 1, dtype=float)
            if len(pmf) < max_delay + 1:
                pmf = np.pad(pmf, (0, max_delay + 1 - len(pmf)))
            weights = np.maximum(pmf[support], 0.0)
            weights /= max(float(weights.sum()), EPS)
            indices = arrival - support
            np.add.at(numerator[node], indices, weights * value)
            np.add.at(denominator[node], indices, weights)
        backprojected = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > EPS)
        x = np.vstack([
            KalmanRTSSmoother._smooth_series(backprojected[i], self.process_variance, self.observation_variance)
            for i in range(n_nodes)
        ])
        w = estimate_graph_from_states(x, self.config.beta, self.config.gamma, self.config.row_sum_cap, candidate_mask=self.config.candidate_mask)
        return _result(x, w, time.perf_counter() - start, (process.memory_info().rss - m0) / 2**20,
                       {"implementation": "internal delay backprojection plus local-level RTS smoothing"})
