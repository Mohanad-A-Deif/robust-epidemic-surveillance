"""Delay-aware robust state and directed-graph inference.

The directed matrix ``W`` is used in the epidemic dynamics, while graph
smoothness is defined through the positive-semidefinite Laplacian of the
symmetrized association matrix ``(W + W.T) / 2``.  Geographic adjacency can be
used as a *soft* sparsity prior; it is never required as a hard support mask.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import psutil

EPS = 1e-12


def huber_loss(residual: np.ndarray, kappa: float) -> np.ndarray:
    r = np.asarray(residual, dtype=float)
    absolute = np.abs(r)
    return np.where(absolute <= kappa, 0.5 * r**2, kappa * absolute - 0.5 * kappa**2)


def huber_score(residual: np.ndarray, kappa: float) -> np.ndarray:
    return np.clip(np.asarray(residual, dtype=float), -kappa, kappa)


def capped_simplex_projection(vector: np.ndarray, cap: float) -> np.ndarray:
    """Project a vector onto ``{x >= 0, sum(x) <= cap}``."""
    v = np.maximum(np.asarray(vector, dtype=float), 0.0)
    if not np.isfinite(cap) or cap <= 0:
        return np.zeros_like(v) if cap <= 0 else v
    if v.sum() <= cap:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, len(u) + 1) > (cssv - cap)
    if not np.any(rho_candidates):
        return np.zeros_like(v)
    rho = np.where(rho_candidates)[0][-1]
    theta = (cssv[rho] - cap) / (rho + 1)
    return np.maximum(v - theta, 0.0)


@dataclass
class InferenceConfig:
    beta: float = 0.12
    gamma: float = 0.08
    sigma_w: float = 0.35
    huber_kappa: float = 1.5
    lambda_w: float = 0.015
    lambda_g: float = 0.015
    lambda_row: float = 0.01
    geo_prior_weight: float = 0.01
    geo_prior_discount: float = 0.5
    row_sum_cap: float = 2.5
    adaptive_row_cap_min_factor: float = 0.50
    adaptive_row_cap_max_factor: float = 1.50
    posterior_temperature: float = 1.0
    delay_transition_strength: float = 0.0
    initialization: str = "delay_backprojection"
    warmup_outer_iter: int = 3
    graph_stage_outer_iter: int = 2
    max_outer_iter: int = 80
    max_x_iter: int = 30
    max_w_iter: int = 20
    tol_objective: float = 1e-5
    tol_x: float = 1e-5
    tol_w: float = 1e-5
    initial_step_x: float = 0.05
    initial_step_w: float = 0.03
    backtracking_factor: float = 0.5
    armijo: float = 1e-4
    min_step: float = 1e-8
    geographic_reference: np.ndarray | None = None
    candidate_mask: np.ndarray | None = None
    fixed_graph: np.ndarray | None = None
    oracle_delay: bool = False
    no_delay: bool = False
    update_graph: bool = True
    verbose: bool = False


@dataclass
class InferenceResult:
    x_hat: np.ndarray
    w_hat: np.ndarray
    posterior: list[np.ndarray]
    feasible_delays: list[np.ndarray]
    predicted_delays: np.ndarray
    objective_history: list[float]
    normalized_objective_history: list[float]
    x_change_history: list[float]
    w_change_history: list[float]
    runtime_seconds: float
    peak_memory_mb: float
    iterations: int
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DelayAwareRobustGraphInference:
    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self._adaptive_row_caps: np.ndarray | None = None

    @staticmethod
    def _validate_messages(messages: pd.DataFrame) -> pd.DataFrame:
        frame = messages.copy()
        if "received" in frame:
            if frame["received"].dtype != bool:
                frame["received"] = frame["received"].astype(str).str.lower().isin(["true", "1"])
            frame = frame[frame["received"]]
        required = ["node_id", "arrival_index", "observed_value", "delay"]
        missing = [col for col in required if col not in frame]
        if missing:
            raise ValueError(f"Messages are missing columns: {missing}")
        frame["arrival_index"] = pd.to_numeric(frame["arrival_index"], errors="coerce")
        frame["observed_value"] = pd.to_numeric(frame["observed_value"], errors="coerce")
        frame["delay"] = pd.to_numeric(frame["delay"], errors="coerce")
        frame = frame.dropna(subset=["arrival_index", "observed_value", "node_id"]).copy()
        frame["arrival_index"] = frame["arrival_index"].astype(int)
        frame["node_id"] = frame["node_id"].astype(int)
        frame["delay"] = frame["delay"].fillna(0).astype(int)
        return frame.sort_values(["node_id", "arrival_index"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _interpolate_grid(grid: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        result = np.asarray(grid, dtype=float).copy()
        if weights is not None:
            observed = np.asarray(weights, dtype=float) > EPS
            result = np.divide(result, np.maximum(weights, EPS), where=np.ones_like(result, dtype=bool))
            result[~observed] = np.nan
        index = np.arange(result.shape[1])
        for i in range(result.shape[0]):
            valid = np.isfinite(result[i])
            if not np.any(valid):
                result[i] = 0.0
            elif valid.sum() == 1:
                result[i] = result[i, valid][0]
            else:
                result[i] = np.interp(index, index[valid], result[i, valid])
        return np.maximum(result, 0.0)

    @classmethod
    def _initialize_arrival_aligned(cls, messages: pd.DataFrame, n_nodes: int, n_time: int) -> np.ndarray:
        grid = np.full((n_nodes, n_time), np.nan, dtype=float)
        grouped = messages.groupby(["node_id", "arrival_index"], as_index=False)["observed_value"].mean()
        for row in grouped.itertuples(index=False):
            if 0 <= int(row.node_id) < n_nodes and 0 <= int(row.arrival_index) < n_time:
                grid[int(row.node_id), int(row.arrival_index)] = float(row.observed_value)
        return cls._interpolate_grid(grid)

    @classmethod
    def _initialize_backprojection(
        cls,
        records: list[dict[str, Any]],
        supports: list[np.ndarray],
        n_nodes: int,
        n_time: int,
    ) -> np.ndarray:
        """Distribute each arrival backwards using its declared delay prior."""
        numer = np.zeros((n_nodes, n_time), dtype=float)
        denom = np.zeros((n_nodes, n_time), dtype=float)
        for record, support in zip(records, supports, strict=True):
            tau = record["arrival"] - support
            valid = (tau >= 0) & (tau < n_time)
            if not np.any(valid):
                continue
            prior = np.asarray(record["prior"], dtype=float)[valid]
            prior = prior / max(float(prior.sum()), EPS)
            node = int(record["node"])
            np.add.at(numer[node], tau[valid], prior * float(record["value"]))
            np.add.at(denom[node], tau[valid], prior)
        return cls._interpolate_grid(numer, denom)

    def _prepare_records(
        self,
        messages: pd.DataFrame,
        max_delay: int,
        delay_pmf_by_slice: dict[str, np.ndarray] | None,
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
        records: list[dict[str, Any]] = []
        posterior: list[np.ndarray] = []
        supports: list[np.ndarray] = []
        cfg = self.config
        for record_id, row in enumerate(messages.itertuples(index=False)):
            arrival = int(row.arrival_index)
            true_delay = int(row.delay)
            if cfg.oracle_delay:
                support = np.array([true_delay], dtype=int)
                prior = np.array([1.0], dtype=float)
            elif cfg.no_delay:
                support = np.array([0], dtype=int)
                prior = np.array([1.0], dtype=float)
            else:
                support = np.arange(0, min(max_delay, arrival) + 1, dtype=int)
                slice_id = str(getattr(row, "slice_id", "default"))
                if delay_pmf_by_slice and slice_id in delay_pmf_by_slice:
                    base = np.asarray(delay_pmf_by_slice[slice_id], dtype=float)
                elif delay_pmf_by_slice and "default" in delay_pmf_by_slice:
                    base = np.asarray(delay_pmf_by_slice["default"], dtype=float)
                else:
                    base = np.ones(max_delay + 1, dtype=float)
                if len(base) < max_delay + 1:
                    base = np.pad(base, (0, max_delay + 1 - len(base)))
                prior = np.maximum(base[support], 0.0)
                prior = prior / max(float(prior.sum()), EPS)
            record = {
                "record_id": record_id,
                "node": int(row.node_id),
                "arrival": arrival,
                "value": float(row.observed_value),
                "true_delay": true_delay,
                "prior": prior,
            }
            records.append(record)
            supports.append(support)
            posterior.append(prior.copy())
        return records, posterior, supports

    def _a_matrix(self, w: np.ndarray) -> np.ndarray:
        cfg = self.config
        degree = np.diag(w.sum(axis=1))
        return (1.0 - cfg.gamma) * np.eye(w.shape[0]) - cfg.beta * (degree - w)

    @staticmethod
    def _smooth_laplacian(w: np.ndarray) -> np.ndarray:
        ws = 0.5 * (w + w.T)
        return np.diag(ws.sum(axis=1)) - ws

    def _sparsity_weights(self, n: int) -> np.ndarray:
        cfg = self.config
        weights = np.full((n, n), float(cfg.lambda_w), dtype=float)
        if cfg.geographic_reference is not None and cfg.geo_prior_weight > 0:
            reference = np.asarray(cfg.geographic_reference, dtype=float)
            if reference.shape != (n, n):
                raise ValueError("geographic_reference has the wrong shape")
            reference = ((reference + reference.T) > 0).astype(float)
            np.fill_diagonal(reference, 0.0)
            # Geographic edges receive a smaller additional penalty; all other
            # edges remain possible and therefore this is a soft prior.
            non_geo = 1.0 - reference
            geo = reference
            weights += cfg.geo_prior_weight * (non_geo + (1.0 - cfg.geo_prior_discount) * geo)
        np.fill_diagonal(weights, 0.0)
        return weights

    def _build_adaptive_row_caps(self, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        n = x.shape[0]
        if n == 0:
            return np.empty(0)
        differences = np.diff(x, axis=1) if x.shape[1] > 1 else x
        scale = np.nanstd(differences, axis=1, ddof=0)
        finite_positive = scale[np.isfinite(scale) & (scale > EPS)]
        median = float(np.median(finite_positive)) if len(finite_positive) else 1.0
        ratio = np.divide(scale, median, out=np.ones_like(scale), where=np.isfinite(scale) & (scale > EPS))
        ratio = np.clip(ratio, cfg.adaptive_row_cap_min_factor, cfg.adaptive_row_cap_max_factor)
        base = max(float(cfg.row_sum_cap), EPS)
        caps = base * ratio
        # A safety bound keeps diagonal entries of A(W) nonnegative; unlike a
        # fixed equality constraint, it does not force equal row sums.
        if cfg.beta > EPS:
            stability_cap = max((1.0 - cfg.gamma) / cfg.beta, EPS)
            caps = np.minimum(caps, 0.98 * stability_cap)
        return caps

    def _row_penalty(self, w: np.ndarray) -> float:
        cfg = self.config
        if cfg.lambda_row <= 0 or self._adaptive_row_caps is None:
            return 0.0
        excess = np.maximum(w.sum(axis=1) - self._adaptive_row_caps, 0.0)
        return float(cfg.lambda_row * np.sum(excess**2))

    def _objective(
        self,
        x: np.ndarray,
        w: np.ndarray,
        records: list[dict[str, Any]],
        posterior: list[np.ndarray],
        supports: list[np.ndarray],
    ) -> float:
        cfg = self.config
        obs = 0.0
        temperature = max(float(cfg.posterior_temperature), EPS)
        for record, alpha, support in zip(records, posterior, supports, strict=True):
            tau = record["arrival"] - support
            residual = record["value"] - x[record["node"], tau]
            prior = np.maximum(record["prior"], EPS)
            a = np.maximum(alpha, EPS)
            obs += float(np.sum(alpha * huber_loss(residual, cfg.huber_kappa) + temperature * alpha * np.log(a / prior)))
        a_matrix = self._a_matrix(w)
        dynamic = x[:, 1:] - a_matrix @ x[:, :-1]
        dynamic_term = 0.5 / (cfg.sigma_w**2) * float(np.sum(dynamic**2))
        ls = self._smooth_laplacian(w)
        graph_term = cfg.lambda_g * float(np.sum(x * (ls @ x)))
        sparse_term = float(np.sum(self._sparsity_weights(w.shape[0]) * w))
        return obs + dynamic_term + graph_term + sparse_term + self._row_penalty(w)

    def _update_posterior(
        self,
        x: np.ndarray,
        records: list[dict[str, Any]],
        supports: list[np.ndarray],
    ) -> list[np.ndarray]:
        cfg = self.config
        temperature = max(float(cfg.posterior_temperature), EPS)
        updated: list[np.ndarray] = [np.empty(0)] * len(records)
        previous_delay_by_node: dict[int, float] = {}
        order = sorted(range(len(records)), key=lambda k: (records[k]["node"], records[k]["arrival"], k))
        for index in order:
            record = records[index]
            support = supports[index]
            tau = record["arrival"] - support
            residual = record["value"] - x[record["node"], tau]
            logits = np.log(np.maximum(record["prior"], EPS)) - huber_loss(residual, cfg.huber_kappa) / temperature
            previous = previous_delay_by_node.get(record["node"])
            if previous is not None and cfg.delay_transition_strength > 0:
                logits -= float(cfg.delay_transition_strength) * (support - previous) ** 2
            logits -= np.max(logits)
            probabilities = np.exp(logits)
            probabilities /= max(float(probabilities.sum()), EPS)
            updated[index] = probabilities
            previous_delay_by_node[record["node"]] = float(np.sum(probabilities * support))
        return updated

    def _gradient_x(
        self,
        x: np.ndarray,
        w: np.ndarray,
        records: list[dict[str, Any]],
        posterior: list[np.ndarray],
        supports: list[np.ndarray],
    ) -> np.ndarray:
        cfg = self.config
        gradient = np.zeros_like(x)
        for record, alpha, support in zip(records, posterior, supports, strict=True):
            tau = record["arrival"] - support
            residual = record["value"] - x[record["node"], tau]
            contribution = -alpha * huber_score(residual, cfg.huber_kappa)
            np.add.at(gradient[record["node"]], tau, contribution)

        a_matrix = self._a_matrix(w)
        residual_dyn = x[:, 1:] - a_matrix @ x[:, :-1]
        scale = 1.0 / (cfg.sigma_w**2)
        gradient[:, 0] += -scale * (a_matrix.T @ residual_dyn[:, 0])
        if x.shape[1] > 2:
            gradient[:, 1:-1] += scale * (residual_dyn[:, :-1] - a_matrix.T @ residual_dyn[:, 1:])
        gradient[:, -1] += scale * residual_dyn[:, -1]
        gradient += 2.0 * cfg.lambda_g * (self._smooth_laplacian(w) @ x)
        return gradient

    def _gradient_w(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        cfg = self.config
        n, t_count = x.shape
        gradient = np.zeros_like(w)
        a_matrix = self._a_matrix(w)
        residual_dyn = x[:, 1:] - a_matrix @ x[:, :-1]
        for t in range(t_count - 1):
            c = x[:, t][:, None] - x[:, t][None, :]
            gradient += (cfg.beta / (cfg.sigma_w**2)) * residual_dyn[:, t][:, None] * c
        graph_gradient = np.zeros_like(w)
        for t in range(t_count):
            c = x[:, t][:, None] - x[:, t][None, :]
            graph_gradient += c**2
        gradient += 0.5 * cfg.lambda_g * graph_gradient
        if cfg.lambda_row > 0 and self._adaptive_row_caps is not None:
            excess = np.maximum(w.sum(axis=1) - self._adaptive_row_caps, 0.0)
            gradient += (2.0 * cfg.lambda_row * excess)[:, None]
        np.fill_diagonal(gradient, 0.0)
        return gradient

    def _project_w(self, w: np.ndarray) -> np.ndarray:
        cfg = self.config
        result = np.maximum(w, 0.0)
        np.fill_diagonal(result, 0.0)
        if cfg.candidate_mask is not None:
            mask = np.asarray(cfg.candidate_mask, dtype=bool)
            if mask.shape != result.shape:
                raise ValueError("candidate_mask has the wrong shape")
            result[~mask] = 0.0
            np.fill_diagonal(result, 0.0)
        # Hard projection is only a broad numerical-stability guard. The
        # row-specific behaviour is controlled by the adaptive soft penalty.
        if cfg.beta > EPS:
            safety_cap = 0.995 * max((1.0 - cfg.gamma) / cfg.beta, EPS)
        else:
            safety_cap = np.inf
        for i in range(result.shape[0]):
            result[i] = capped_simplex_projection(result[i], safety_cap)
            result[i, i] = 0.0
        return result

    def fit(
        self,
        messages: pd.DataFrame,
        n_nodes: int,
        n_time: int,
        max_delay: int,
        delay_pmf_by_slice: dict[str, np.ndarray] | None = None,
        initial_x: np.ndarray | None = None,
        initial_w: np.ndarray | None = None,
    ) -> InferenceResult:
        cfg = self.config
        if cfg.posterior_temperature <= 0:
            raise ValueError("posterior_temperature must be positive")
        process = psutil.Process()
        memory_start = process.memory_info().rss
        peak_memory = memory_start
        start = time.perf_counter()

        received = self._validate_messages(messages)
        records, posterior, supports = self._prepare_records(received, max_delay, delay_pmf_by_slice)
        if not records:
            raise ValueError("No received records are available for inference.")

        if initial_x is not None:
            x = np.asarray(initial_x, dtype=float).copy()
        elif cfg.initialization == "arrival_interpolation":
            x = self._initialize_arrival_aligned(received, n_nodes, n_time)
        elif cfg.initialization == "delay_backprojection":
            x = self._initialize_backprojection(records, supports, n_nodes, n_time)
        else:
            raise ValueError(f"Unknown initialization: {cfg.initialization}")
        if x.shape != (n_nodes, n_time):
            raise ValueError(f"initial_x must have shape {(n_nodes, n_time)}")
        x = np.maximum(x, 0.0)
        self._adaptive_row_caps = self._build_adaptive_row_caps(x)

        if cfg.fixed_graph is not None:
            w = self._project_w(np.asarray(cfg.fixed_graph, dtype=float))
        elif initial_w is not None:
            w = self._project_w(np.asarray(initial_w, dtype=float))
        else:
            w = np.zeros((n_nodes, n_nodes), dtype=float)

        objective_history = [self._objective(x, w, records, posterior, supports)]
        normalizer = max(float(len(records) + n_nodes * max(n_time - 1, 1)), 1.0)
        normalized_history = [objective_history[0] / normalizer]
        x_changes: list[float] = []
        w_changes: list[float] = []
        phase_history: list[str] = []
        converged = False

        for outer in range(cfg.max_outer_iter):
            outer_x_start = x.copy()
            outer_w_start = w.copy()
            posterior = self._update_posterior(x, records, supports)

            if outer < cfg.warmup_outer_iter:
                phase = "state_warmup"
                update_x, update_w = True, False
            elif outer < cfg.warmup_outer_iter + cfg.graph_stage_outer_iter:
                phase = "graph_stage"
                update_x, update_w = False, bool(cfg.update_graph and cfg.fixed_graph is None)
            else:
                phase = "joint_refinement"
                update_x, update_w = True, bool(cfg.update_graph and cfg.fixed_graph is None)
            phase_history.append(phase)

            if update_x:
                for _ in range(cfg.max_x_iter):
                    x_inner_start = x.copy()
                    gradient = self._gradient_x(x, w, records, posterior, supports)
                    current = self._objective(x, w, records, posterior, supports)
                    step = cfg.initial_step_x
                    grad_norm_sq = float(np.sum(gradient**2))
                    accepted = False
                    while step >= cfg.min_step:
                        candidate = np.maximum(x - step * gradient, 0.0)
                        candidate_objective = self._objective(candidate, w, records, posterior, supports)
                        if candidate_objective <= current - cfg.armijo * step * grad_norm_sq or candidate_objective <= current:
                            x = candidate
                            accepted = True
                            break
                        step *= cfg.backtracking_factor
                    if not accepted:
                        break
                    relative = float(np.linalg.norm(x - x_inner_start) / max(np.linalg.norm(x_inner_start), 1.0))
                    if relative < cfg.tol_x:
                        break

            if update_w:
                sparsity_weights = self._sparsity_weights(n_nodes)
                for _ in range(cfg.max_w_iter):
                    w_inner_start = w.copy()
                    gradient_w = self._gradient_w(x, w)
                    current = self._objective(x, w, records, posterior, supports)
                    step = cfg.initial_step_w
                    accepted = False
                    while step >= cfg.min_step:
                        raw = w - step * gradient_w
                        candidate = self._project_w(np.maximum(raw - step * sparsity_weights, 0.0))
                        candidate_objective = self._objective(x, candidate, records, posterior, supports)
                        if candidate_objective <= current:
                            w = candidate
                            accepted = True
                            break
                        step *= cfg.backtracking_factor
                    if not accepted:
                        break
                    relative_w = float(np.linalg.norm(w - w_inner_start) / max(np.linalg.norm(w_inner_start), 1.0))
                    if relative_w < cfg.tol_w:
                        break

            objective = self._objective(x, w, records, posterior, supports)
            objective_history.append(objective)
            normalized_history.append(objective / normalizer)
            x_change = float(np.linalg.norm(x - outer_x_start) / max(np.linalg.norm(outer_x_start), 1.0))
            w_change = float(np.linalg.norm(w - outer_w_start) / max(np.linalg.norm(outer_w_start), 1.0))
            x_changes.append(x_change)
            w_changes.append(w_change)
            relative_obj = abs(objective_history[-1] - objective_history[-2]) / max(1.0, abs(objective_history[-2]))
            peak_memory = max(peak_memory, process.memory_info().rss)
            if cfg.verbose:
                print(f"outer={outer + 1} phase={phase} objective={objective:.6g} rel_obj={relative_obj:.3g}")
            minimum_joint_iteration = cfg.warmup_outer_iter + cfg.graph_stage_outer_iter
            if outer >= minimum_joint_iteration and relative_obj < cfg.tol_objective and max(x_change, w_change) < max(cfg.tol_x, cfg.tol_w) * 10:
                converged = True
                break

        runtime = time.perf_counter() - start
        predicted_delays = np.asarray(
            [float(np.sum(alpha * support)) for alpha, support in zip(posterior, supports, strict=True)]
        )
        off_diagonal = ~np.eye(n_nodes, dtype=bool)
        smooth_laplacian = self._smooth_laplacian(w)
        diagnostics = {
            "n_received_records": len(records),
            "n_nodes": n_nodes,
            "n_time": n_time,
            "max_delay": max_delay,
            "final_objective": float(objective_history[-1]),
            "final_normalized_objective": float(normalized_history[-1]),
            "objective_nonincreasing": bool(np.all(np.diff(objective_history) <= 1e-8)),
            "graph_density": float(np.mean(w[off_diagonal] > 1e-8)) if np.any(off_diagonal) else 0.0,
            "graph_row_sum_max": float(w.sum(axis=1).max()) if n_nodes else 0.0,
            "adaptive_row_caps": self._adaptive_row_caps.tolist() if self._adaptive_row_caps is not None else [],
            "smooth_laplacian_min_eigenvalue": float(np.linalg.eigvalsh(smooth_laplacian).min()) if n_nodes else 0.0,
            "phase_history": phase_history,
            "initialization": cfg.initialization,
            "posterior_temperature": float(cfg.posterior_temperature),
            "delay_transition_strength": float(cfg.delay_transition_strength),
            "geography_is_soft_prior": bool(cfg.geographic_reference is not None and cfg.candidate_mask is None),
        }
        return InferenceResult(
            x_hat=x,
            w_hat=w,
            posterior=posterior,
            feasible_delays=supports,
            predicted_delays=predicted_delays,
            objective_history=objective_history,
            normalized_objective_history=normalized_history,
            x_change_history=x_changes,
            w_change_history=w_changes,
            runtime_seconds=float(runtime),
            # Report the absolute process peak resident set size (RSS), not the
            # increment above the process baseline. The latter can be zero when
            # Python reuses already allocated pages and is therefore misleading
            # in scalability tables.
            peak_memory_mb=float(peak_memory / (1024**2)),
            iterations=len(objective_history) - 1,
            converged=converged,
            diagnostics=diagnostics,
        )
