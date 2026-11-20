"""Preprocessing layer for M1 baseline models.

Wraps categorical encoding behind a small, dedicated API so that:

  * the encoder is fit on training data only,
  * validation/test transforms reuse that same fitted encoder,
  * the feature columns and their order are identical across splits,
  * unknown categories are handled safely via ``OneHotEncoder`` with
    ``handle_unknown="ignore"`` (never fitted on val/test data).

This module intentionally does not touch model code, the loader, the split
enforcement, or the feature builder — it only encodes a fitted geometric
``X`` into the numeric matrix the models consume.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DEFAULT_CATEGORICAL_COLUMNS = ("vehicle_class",)


class FeaturePreprocessor:
    """Fit-on-train, transform-anywhere categorical encoder.

    Parameters
    ----------
    categorical_columns : iterable of str
        Column names to treat as categorical. All other columns are passed
        through unchanged.
    drop : {"first", None, "if_binary"}
        Passed to :class:`sklearn.preprocessing.OneHotEncoder`. Defaults to
        ``"first"`` so the linear models (Ridge) do not suffer from perfect
        collinearity.
    """

    def __init__(
        self,
        categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS,
        drop: str | None = "first",
    ):
        self.categorical_columns = tuple(categorical_columns)
        self.drop = drop
        self.encoder_: OneHotEncoder | None = None
        self.feature_names_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit the encoder on the *training* matrix only."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected DataFrame, got {type(X).__name__}")

        cat = [c for c in X.columns if c in self.categorical_columns]
        self.categorical_cols_ = cat
        self.numeric_cols_ = [c for c in X.columns if c not in self.categorical_columns]

        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            drop=self.drop,
            sparse_output=False,
        )
        if cat:
            self.encoder_.fit(X[cat])

        if cat:
            encoded_names = self.encoder_.get_feature_names_out(cat).tolist()
            self.feature_names_ = self.numeric_cols_ + encoded_names
        else:
            self.feature_names_ = list(self.numeric_cols_)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode ``X`` using the fitted (training) encoder.

        No fitting happens here: the column layout is exactly the one
        established at :meth:`fit` time.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before transform()")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected DataFrame, got {type(X).__name__}")

        expected = self.numeric_cols_ + self.categorical_cols_
        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f"missing expected column(s): {sorted(missing)}")

        num = X[self.numeric_cols_]

        if self.categorical_cols_:
            encoded = self.encoder_.transform(X[self.categorical_cols_])
            encoded_names = list(self.encoder_.get_feature_names_out(self.categorical_cols_))
            encoded_df = pd.DataFrame(encoded, index=X.index, columns=encoded_names)
        else:
            encoded_df = pd.DataFrame(index=X.index)

        out = pd.concat([num, encoded_df], axis=1)
        # Guarantee identical column set and order as fitted on train.
        out = out.reindex(columns=self.feature_names_)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``X`` and return the transformed matrix."""
        return self.fit(X).transform(X)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        state = "fitted" if self._fitted else "unfitted"
        return (
            f"FeaturePreprocessor(categorical={self.categorical_cols_}, "
            f"numeric={self.numeric_cols_}, feature_names={self.feature_names_}, "
            f"{state})"
        )
