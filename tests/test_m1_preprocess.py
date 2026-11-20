"""Focused tests for M1.2 preprocessing (train-only encoder fit).

Covers the four agreed guarantees:
  1. the categorical encoder is fitted on training data only,
  2. feature columns and their order are identical across splits,
  3. unknown categories are handled safely (OneHotEncoder handle_unknown='ignore'),
  4. no split leakage (val/test never influence the fitted encoder).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_feature_matrix
from src.features.preprocess import FeaturePreprocessor
from src.features.spec import load_feature_spec

SPEC = load_feature_spec()
FULL_FEATURES = SPEC["feature_sets"]["full_operational"]["features"]
TARGET = "fuel_rate_lph"


def _make_df(classes: list[str]) -> pd.DataFrame:
    """Minimal frame covering every ``full_operational`` feature + target."""
    n = len(classes)
    data: dict[str, list] = {}
    for c in FULL_FEATURES:
        if c == "vehicle_class":
            data[c] = list(classes)
        else:
            data[c] = (np.arange(n, dtype=float) * 0.1 + 1.0).tolist()
    data[TARGET] = (np.arange(n, dtype=float) * 0.5).tolist()
    return pd.DataFrame(data)


def _matrices(classes):
    df = _make_df(classes)
    return build_feature_matrix(df, "full_operational", spec=SPEC)


@pytest.fixture
def train_m():
    return _matrices(["32T", "32T", "40T", "40T"])


@pytest.fixture
def val_m_with_unknown():
    # contains a class (49T) never seen in the training fixture
    return _matrices(["32T", "40T", "49T"])


# --------------------------------------------------------------------------
# 1. Training-only encoder fit
# --------------------------------------------------------------------------

def test_encoder_fit_uses_train_only_categories(train_m, val_m_with_unknown):
    X_train, _, _ = train_m
    pre = FeaturePreprocessor().fit(X_train)

    assert pre._fitted is True
    # categories_ must reflect ONLY train classes (32T, 40T) — not 49T.
    learned = [list(cat) for cat in pre.encoder_.categories_]
    assert learned == [["32T", "40T"]], learned
    assert "49T" not in learned[0]


# --------------------------------------------------------------------------
# 2. Consistent feature columns across splits
# --------------------------------------------------------------------------

def test_identical_columns_across_train_and_val(train_m, val_m_with_unknown):
    X_train, _, _ = train_m
    X_val, _, _ = val_m_with_unknown
    pre = FeaturePreprocessor().fit(X_train)

    Xt = pre.transform(X_train)
    Xv = pre.transform(X_val)

    assert list(Xt.columns) == list(Xv.columns) == pre.feature_names_
    assert len(Xt.columns) == len(pre.feature_names_) > 0
    # no raw categorical column survives
    assert "vehicle_class" not in Xt.columns


def test_transform_requires_fit_first(train_m):
    X_train, _, _ = train_m
    pre = FeaturePreprocessor()
    with pytest.raises(RuntimeError, match="fit"):
        pre.transform(X_train)


# --------------------------------------------------------------------------
# 3. Unknown-category handling
# --------------------------------------------------------------------------

def test_unknown_category_does_not_raise_and_zero_encoded(train_m, val_m_with_unknown):
    X_train, _, _ = train_m
    X_val, _, _ = val_m_with_unknown

    # Use drop=None so the unknown category maps to an all-zero vector clearly.
    pre = FeaturePreprocessor(drop=None).fit(X_train)
    Xv = pre.transform(X_val)

    idx_unknown = X_val.index[X_val["vehicle_class"] == "49T"]
    row = Xv.loc[idx_unknown].iloc[0]
    onehot_cols = [c for c in Xv.columns if c not in pre.numeric_cols_]
    assert onehot_cols, "expected one-hot columns to exist"
    assert (row[onehot_cols] == 0).all(), row[onehot_cols].values


# --------------------------------------------------------------------------
# 4. No split leakage
# --------------------------------------------------------------------------

def test_val_data_never_influences_encoder(train_m, val_m_with_unknown):
    X_train, _, _ = train_m
    X_val, _, _ = val_m_with_unknown

    pre_train_only = FeaturePreprocessor().fit(X_train)
    # Simulate a (forbidden) fit on combined train+val.
    X_combined = pd.concat([X_train, X_val], axis=0)
    pre_leaky = FeaturePreprocessor().fit(X_combined)

    # The leaky encoder sees 49T; the correct one must not.
    assert len(pre_train_only.encoder_.categories_[0]) == 2
    assert len(pre_leaky.encoder_.categories_[0]) == 3

    # transform(val) with the train-only encoder must equal the output for a
    # second call (deterministic), and must never refit.
    Xv1 = pre_train_only.transform(X_val)
    Xv2 = pre_train_only.transform(X_val)
    pd.testing.assert_frame_equal(Xv1, Xv2)


def test_val_transform_columns_equal_train_transform_columns(train_m, val_m_with_unknown):
    X_train, _, _ = train_m
    X_val, _, _ = val_m_with_unknown
    pre = FeaturePreprocessor().fit(X_train)
    pd.testing.assert_series_equal(
        pre.transform(X_train).columns.to_series(),
        pre.transform(X_val).columns.to_series(),
    )
