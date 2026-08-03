"""Paired statistical tests, confidence intervals, effects, and corrections."""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


def bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean, confidence: float = 0.95, n_boot: int = 5000, seed: int = 12345) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        value = float(statistic(x))
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    estimates = np.asarray([statistic(x[idx]) for idx in indices], dtype=float)
    alpha = 1.0 - confidence
    return float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return adjusted
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, original in enumerate(order):
        candidate = (m - rank) * p[original]
        running = max(running, candidate)
        adjusted[original] = min(1.0, running)
    return adjusted


def paired_effects(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {}
    d = a - b
    sd = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
    cohen_dz = float(np.mean(d) / sd) if np.isfinite(sd) and sd > EPS else np.nan
    nonzero = d[d != 0]
    rank_biserial = np.nan
    if nonzero.size:
        ranks = stats.rankdata(np.abs(nonzero))
        pos = float(ranks[nonzero > 0].sum())
        neg = float(ranks[nonzero < 0].sum())
        rank_biserial = (pos - neg) / max(pos + neg, EPS)
    return {"mean_difference": float(np.mean(d)), "median_difference": float(np.median(d)), "cohen_dz": cohen_dz, "rank_biserial": float(rank_biserial)}


def pairwise_method_tests(seed_results: pd.DataFrame, metric: str, lower_is_better: bool = True, group_cols: tuple[str, ...] = ("scenario", "seed")) -> pd.DataFrame:
    needed = {*group_cols, "method", metric}
    missing = needed.difference(seed_results.columns)
    if missing:
        raise ValueError(f"Missing columns for paired tests: {sorted(missing)}")
    pivot = seed_results.pivot_table(index=list(group_cols), columns="method", values=metric, aggfunc="mean")
    rows = []
    for method_a, method_b in combinations(pivot.columns, 2):
        pair = pivot[[method_a, method_b]].dropna()
        a, b = pair[method_a].to_numpy(), pair[method_b].to_numpy()
        if len(pair) < 2:
            continue
        try:
            t = stats.ttest_rel(a, b, nan_policy="omit")
            t_stat, t_p = float(t.statistic), float(t.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        effects = paired_effects(a, b)
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        if lower_is_better:
            winner = method_a if mean_a < mean_b else method_b
        else:
            winner = method_a if mean_a > mean_b else method_b
        rows.append(
            {
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "n_pairs": len(pair),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "winner_by_mean": winner,
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                **effects,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["paired_t_p_holm"] = holm_adjust(result["paired_t_p"].to_numpy())
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"].to_numpy())
    return result


def summarize_seed_results(seed_results: pd.DataFrame, metrics: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["scenario", "family", "method"]
    rows = []
    for keys, group in seed_results.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys, strict=True))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            lo, hi = bootstrap_ci(values)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def friedman_test(seed_results: pd.DataFrame, metric: str, index_cols: tuple[str, ...] = ("scenario", "seed")) -> dict[str, float]:
    pivot = seed_results.pivot_table(index=list(index_cols), columns="method", values=metric, aggfunc="mean").dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {"friedman_stat": np.nan, "friedman_p": np.nan, "n_blocks": int(pivot.shape[0]), "n_methods": int(pivot.shape[1])}
    result = stats.friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
    return {"friedman_stat": float(result.statistic), "friedman_p": float(result.pvalue), "n_blocks": int(pivot.shape[0]), "n_methods": int(pivot.shape[1])}


def average_ranks(seed_results: pd.DataFrame, metric: str, lower_is_better: bool = True, index_cols: tuple[str, ...] = ("scenario", "seed")) -> pd.DataFrame:
    pivot = seed_results.pivot_table(index=list(index_cols), columns="method", values=metric, aggfunc="mean")
    ranks = pivot.rank(axis=1, method="average", ascending=lower_is_better)
    return pd.DataFrame({"method": ranks.columns, "average_rank": ranks.mean(axis=0).to_numpy(), "rank_std": ranks.std(axis=0).to_numpy()}).sort_values("average_rank")
