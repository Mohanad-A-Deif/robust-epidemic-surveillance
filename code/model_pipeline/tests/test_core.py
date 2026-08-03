from pathlib import Path

import numpy as np
import pandas as pd

from epidemic_results.demo_data import create_demo_dataset
from epidemic_results.io_utils import discover_message_files, load_processed_bundle, read_messages
from epidemic_results.metrics import graph_metrics, state_metrics
from epidemic_results.proposed_model import DelayAwareRobustGraphInference, InferenceConfig


def test_metrics_identity():
    x = np.arange(12, dtype=float).reshape(3, 4)
    metrics = state_metrics(x, x)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_graph_identity():
    w = np.array([[0.0, 1.0], [0.0, 0.0]])
    metrics = graph_metrics(w, w, threshold=0.5)
    assert metrics["graph_frobenius"] == 0.0
    assert metrics["graph_support_f1"] == 1.0


def test_proposed_model_smoke(tmp_path: Path):
    create_demo_dataset(tmp_path, n_time=35, seeds=[7])
    bundle = load_processed_bundle(tmp_path / "processed")
    file = discover_message_files(tmp_path / "scenarios")[0]
    messages = read_messages(file)
    config = InferenceConfig(max_outer_iter=3, max_x_iter=2, max_w_iter=2, candidate_mask=bundle.adjacency > 0)
    result = DelayAwareRobustGraphInference(config).fit(
        messages,
        n_nodes=bundle.x_log1p.shape[1],
        n_time=bundle.x_log1p.shape[0],
        max_delay=int(messages["delay"].max()),
        delay_pmf_by_slice={"default": np.ones(int(messages["delay"].max()) + 1)},
    )
    assert result.x_hat.shape == bundle.x_log1p.to_numpy().T.shape
    assert result.w_hat.shape == (bundle.x_log1p.shape[1], bundle.x_log1p.shape[1])
    assert np.all(result.x_hat >= 0)
    assert np.all(result.w_hat >= 0)
    assert np.allclose(np.diag(result.w_hat), 0)


def test_symmetrized_smooth_laplacian_is_psd():
    w = np.array([[0.0, 0.8, 0.0], [0.1, 0.0, 0.3], [0.4, 0.0, 0.0]])
    laplacian = DelayAwareRobustGraphInference._smooth_laplacian(w)
    assert np.allclose(laplacian, laplacian.T)
    assert np.linalg.eigvalsh(laplacian).min() >= -1e-10


def test_geography_is_soft_prior_not_hard_mask(tmp_path: Path):
    create_demo_dataset(tmp_path, n_time=35, seeds=[7])
    bundle = load_processed_bundle(tmp_path / "processed")
    file = discover_message_files(tmp_path / "scenarios")[0]
    messages = read_messages(file)
    config = InferenceConfig(
        max_outer_iter=4,
        max_x_iter=2,
        max_w_iter=2,
        geographic_reference=bundle.adjacency,
        candidate_mask=None,
        geo_prior_weight=0.01,
    )
    result = DelayAwareRobustGraphInference(config).fit(
        messages,
        n_nodes=bundle.x_log1p.shape[1],
        n_time=bundle.x_log1p.shape[0],
        max_delay=int(messages["delay"].max()),
        delay_pmf_by_slice={"default": np.ones(int(messages["delay"].max()) + 1)},
    )
    assert result.diagnostics["geography_is_soft_prior"] is True
    assert result.diagnostics["smooth_laplacian_min_eigenvalue"] >= -1e-9


def test_causal_message_filter_excludes_future_arrivals():
    from epidemic_results.study_protocol import received_messages_available_by

    messages = pd.DataFrame(
        {
            "node_id": [0, 0, 0],
            "arrival_index": [2, 5, 8],
            "observed_value": [1.0, 2.0, 3.0],
            "delay": [0, 1, 2],
            "received": [True, True, True],
        }
    )
    available = received_messages_available_by(messages, 5)
    assert available["arrival_index"].max() <= 5
    assert len(available) == 2
