"""Lightweight integrity tests for the M1 baseline model registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    MODEL_REGISTRY,
    RANDOM_STATE,
    get_model_factory,
    available_models,
)


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.sin(X.iloc[:, 0]) + rng.normal(size=40, scale=0.1))
    return X, y


def test_catalog_has_exactly_five_agreed_models():
    assert set(available_models()) == {
        "mean_dummy",
        "median_dummy",
        "ridge",
        "random_forest",
        "xgboost",
    }


def test_random_state_fixed_and_documented():
    # The plan fixes random_state=20240617 (plan §4.4); RF/XGB use it.
    assert RANDOM_STATE == 20240617


def test_every_model_fits_and_predicts(tiny_data):
    X, y = tiny_data
    for name in sorted(MODEL_REGISTRY):
        est = get_model_factory(name)()
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (len(X),)
        assert np.all(np.isfinite(pred))
