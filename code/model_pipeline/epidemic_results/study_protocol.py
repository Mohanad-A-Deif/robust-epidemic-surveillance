"""Leakage-safe evaluation helpers for retrospective and causal analyses."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .baselines import (
    ArrivalAlignedInterpolation,
    BaselineConfig,
    DelayAwareAugmentedStateSpace,
    DelayBackprojectionNowcast,
    GraphTemporalSmoother,
    KalmanRTSSmoother,
    LowRankMatrixCompletion,
    OracleTimestampInterpolation,
    RobustMedianSmoother,
)
from .proposed_model import DelayAwareRobustGraphInference, InferenceConfig, InferenceResult


def split_indices(splits: pd.Series, dates: pd.Index) -> dict[str, np.ndarray]:
    labels = splits.reindex(dates).astype(str).to_numpy()
    result = {name: np.flatnonzero(labels == name) for name in ("train", "validation", "test")}
    if any(len(result[name]) == 0 for name in result):
        raise ValueError(f"All chronological splits must be non-empty: { {k: len(v) for k, v in result.items()} }")
    if not (result["train"].max() < result["validation"].min() <= result["validation"].max() < result["test"].min()):
        raise ValueError("Train/validation/test splits must be chronological and non-overlapping.")
    return result


def received_messages_available_by(messages: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    frame = messages.copy()
    if "received" in frame:
        if frame["received"].dtype != bool:
            frame["received"] = frame["received"].astype(str).str.lower().isin(["true", "1"])
        frame = frame[frame["received"]]
    arrival = pd.to_numeric(frame["arrival_index"], errors="coerce")
    frame = frame[arrival.notna() & (arrival <= int(cutoff))].copy()
    frame["arrival_index"] = pd.to_numeric(frame["arrival_index"], errors="raise").astype(int)
    return frame.reset_index(drop=True)


def config_to_jsonable(config: InferenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("geographic_reference", "candidate_mask", "fixed_graph"):
        value = payload.get(key)
        payload[key] = np.asarray(value).tolist() if value is not None else None
    return payload


def config_from_jsonable(payload: dict[str, Any]) -> InferenceConfig:
    data = dict(payload)
    for key in ("geographic_reference", "candidate_mask", "fixed_graph"):
        if data.get(key) is not None:
            dtype = bool if key == "candidate_mask" else float
            data[key] = np.asarray(data[key], dtype=dtype)
    return InferenceConfig(**data)


def normalized_geographic_graph(adjacency: np.ndarray | None, row_sum: float = 0.5) -> np.ndarray | None:
    if adjacency is None:
        return None
    graph = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(graph, 0.0)
    sums = graph.sum(axis=1, keepdims=True)
    graph = np.divide(row_sum * graph, np.maximum(sums, 1.0), where=np.ones_like(graph, dtype=bool))
    return graph


def build_method(
    method_name: str,
    selected_config: InferenceConfig,
    adjacency: np.ndarray | None,
    w_true: np.ndarray | None = None,
    quick: bool = False,
) -> Any:
    base = selected_config
    baseline_cfg = BaselineConfig(
        beta=base.beta,
        gamma=base.gamma,
        row_sum_cap=base.row_sum_cap,
        candidate_mask=None,
    )
    fixed_geo = normalized_geographic_graph(adjacency)
    if method_name == "Proposed revised":
        return DelayAwareRobustGraphInference(base)
    if method_name == "Original proposed ablation":
        hard_mask = None
        if adjacency is not None:
            hard_mask = (np.asarray(adjacency) + np.asarray(adjacency).T) > 0
            np.fill_diagonal(hard_mask, False)
        original = replace(
            base,
            initialization="arrival_interpolation",
            posterior_temperature=1.0,
            delay_transition_strength=0.0,
            warmup_outer_iter=0,
            graph_stage_outer_iter=0,
            lambda_row=0.0,
            geo_prior_weight=0.0,
            geographic_reference=None,
            candidate_mask=hard_mask,
        )
        return DelayAwareRobustGraphInference(original)
    if method_name == "Known-delay proposed oracle":
        return DelayAwareRobustGraphInference(replace(base, oracle_delay=True))
    if method_name == "No-delay ablation":
        return DelayAwareRobustGraphInference(replace(base, no_delay=True))
    if method_name == "No-robustness ablation":
        return DelayAwareRobustGraphInference(replace(base, huber_kappa=1e6))
    if method_name == "No-graph ablation":
        zeros = np.zeros_like(adjacency, dtype=float) if adjacency is not None else None
        if zeros is None:
            raise ValueError("No-graph ablation requires the graph dimension through adjacency.")
        return DelayAwareRobustGraphInference(replace(base, fixed_graph=zeros, update_graph=False, lambda_g=0.0, lambda_w=0.0, geo_prior_weight=0.0))
    if method_name == "Fixed geographic graph":
        if fixed_geo is None:
            raise ValueError("Fixed geographic graph requires adjacency.")
        return DelayAwareRobustGraphInference(replace(base, fixed_graph=fixed_geo, update_graph=False, geographic_reference=None, geo_prior_weight=0.0))
    if method_name == "Fixed true-graph oracle":
        if w_true is None:
            raise ValueError("Fixed true-graph oracle requires W_true.")
        return DelayAwareRobustGraphInference(replace(base, fixed_graph=np.asarray(w_true, dtype=float), update_graph=False, geographic_reference=None, geo_prior_weight=0.0))
    if method_name == "Arrival interpolation":
        return ArrivalAlignedInterpolation(baseline_cfg)
    if method_name == "Oracle timestamp interpolation":
        return OracleTimestampInterpolation(baseline_cfg)
    if method_name == "Delay backprojection":
        return DelayBackprojectionNowcast(baseline_cfg)
    if method_name == "Kalman/RTS smoother (internal)":
        return KalmanRTSSmoother(config=baseline_cfg)
    if method_name == "Delay-aware state-space (internal)":
        return DelayAwareAugmentedStateSpace(config=baseline_cfg)
    if method_name == "Robust median smoother":
        return RobustMedianSmoother(window=5, config=baseline_cfg)
    if method_name == "Robust low-rank completion":
        return LowRankMatrixCompletion(rank=3, max_iter=60 if not quick else 15, config=baseline_cfg)
    if method_name == "Graph-temporal reconstruction":
        if adjacency is None:
            raise ValueError("Graph-temporal reconstruction requires adjacency.")
        return GraphTemporalSmoother(adjacency, iterations=40 if not quick else 10, config=baseline_cfg)
    raise KeyError(f"Unknown method: {method_name}")


def causal_rolling_nowcast(
    estimator_factory: Callable[[int, np.ndarray | None, np.ndarray | None], Any],
    messages: pd.DataFrame,
    n_nodes: int,
    test_indices: np.ndarray,
    max_delay: int,
    delay_pmf_by_slice: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[InferenceResult]]:
    """Estimate x_t using only reports whose arrival index is at most t."""
    estimates = np.full((n_nodes, len(test_indices)), np.nan, dtype=float)
    results: list[InferenceResult] = []
    previous_x: np.ndarray | None = None
    previous_w: np.ndarray | None = None
    for position, t in enumerate(test_indices):
        available = received_messages_available_by(messages, int(t))
        if previous_x is not None:
            padded = np.empty((n_nodes, int(t) + 1), dtype=float)
            padded[:, :-1] = previous_x
            padded[:, -1] = previous_x[:, -1]
            initial_x = padded
        else:
            initial_x = None
        estimator = estimator_factory(position, initial_x, previous_w)
        kwargs: dict[str, Any] = {
            "messages": available,
            "n_nodes": n_nodes,
            "n_time": int(t) + 1,
            "max_delay": max_delay,
            "delay_pmf_by_slice": delay_pmf_by_slice,
        }
        if isinstance(estimator, DelayAwareRobustGraphInference):
            kwargs["initial_x"] = initial_x
            kwargs["initial_w"] = previous_w
        result = estimator.fit(**kwargs)
        estimates[:, position] = result.x_hat[:, -1]
        previous_x = result.x_hat
        previous_w = result.w_hat
        results.append(result)
    return estimates, results
