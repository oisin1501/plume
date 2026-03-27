"""
Plume ML Validation Suite
=========================
Proves that Plume's training pipeline produces correct results by running
well-known benchmark datasets through the *exact* code path that the app uses
(prepare_features -> train_supervised) and asserting that metrics meet
established performance floors.

These floors are intentionally conservative — any competent implementation of
the algorithm should clear them.  A failure here means something is broken in
preprocessing, encoding, splitting, or model fitting.

Run:  pytest test_validation.py -v
"""

import json
import io
import os
import sys
import tempfile
import contextlib

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    make_regression,
)

# Import Plume's own training function
sys.path.insert(0, os.path.dirname(__file__))
import plume_ml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp CSV and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def _train(params: dict) -> dict:
    """Call train_supervised and capture the JSON response it prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plume_ml.train_supervised(params)
    return json.loads(buf.getvalue())


# ---------------------------------------------------------------------------
# Classification benchmarks
# ---------------------------------------------------------------------------

class TestClassification:
    """Each test loads a gold-standard dataset, trains via Plume, and asserts
    that test-set metrics exceed a known minimum."""

    # -- Iris (150 rows, 4 features, 3 classes) --

    @pytest.fixture()
    def iris_csv(self):
        ds = load_iris(as_frame=True)
        df = ds.frame.copy()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        path = _make_csv(df)
        yield path, [c for c in df.columns if c != "target"], "target"
        os.unlink(path)

    @pytest.mark.parametrize("algo", ["random_forest", "logistic_regression"])
    def test_iris(self, iris_csv, algo, benchmark):
        path, features, target = iris_csv
        thresholds = {"accuracy": 0.90, "f1": 0.88}
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "classification",
            "algorithm": algo,
            "test_size": 0.2,
        })
        assert result["status"] == "success", result.get("error")
        m = result["metrics"]
        benchmark(
            dataset="Iris (150 rows, 3 classes)",
            algorithm=algo,
            task="classification",
            metrics={"accuracy": m["accuracy"], "f1": m["f1"]},
            thresholds=thresholds,
        )
        assert m["accuracy"] >= thresholds["accuracy"]
        assert m["f1"] >= thresholds["f1"]

    # -- Breast Cancer (569 rows, 30 features, 2 classes) --

    @pytest.fixture()
    def breast_cancer_csv(self):
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.copy()
        df["target"] = ds.target
        df.columns = [c.replace(" ", "_") for c in df.columns]
        path = _make_csv(df)
        yield path, [c for c in df.columns if c != "target"], "target"
        os.unlink(path)

    @pytest.mark.parametrize("algo", ["random_forest", "logistic_regression", "xgboost", "lightgbm"])
    def test_breast_cancer(self, breast_cancer_csv, algo, benchmark):
        path, features, target = breast_cancer_csv
        thresholds = {"accuracy": 0.92, "f1": 0.90}
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "classification",
            "algorithm": algo,
            "test_size": 0.2,
        })
        assert result["status"] == "success", result.get("error")
        m = result["metrics"]
        benchmark(
            dataset="Breast Cancer (569 rows, 2 classes)",
            algorithm=algo,
            task="classification",
            metrics={"accuracy": m["accuracy"], "f1": m["f1"]},
            thresholds=thresholds,
        )
        assert m["accuracy"] >= thresholds["accuracy"]
        assert m["f1"] >= thresholds["f1"]

    # -- Wine (178 rows, 13 features, 3 classes) --

    @pytest.fixture()
    def wine_csv(self):
        ds = load_wine(as_frame=True)
        df = ds.frame.copy()
        df["target"] = ds.target
        df.columns = [c.replace(" ", "_") for c in df.columns]
        path = _make_csv(df)
        yield path, [c for c in df.columns if c != "target"], "target"
        os.unlink(path)

    @pytest.mark.parametrize("algo", ["random_forest", "xgboost", "lightgbm"])
    def test_wine(self, wine_csv, algo, benchmark):
        path, features, target = wine_csv
        thresholds = {"accuracy": 0.85, "f1": 0.83}
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "classification",
            "algorithm": algo,
            "test_size": 0.2,
        })
        assert result["status"] == "success", result.get("error")
        m = result["metrics"]
        benchmark(
            dataset="Wine (178 rows, 3 classes)",
            algorithm=algo,
            task="classification",
            metrics={"accuracy": m["accuracy"], "f1": m["f1"]},
            thresholds=thresholds,
        )
        assert m["accuracy"] >= thresholds["accuracy"]
        assert m["f1"] >= thresholds["f1"]


# ---------------------------------------------------------------------------
# Regression benchmarks
# ---------------------------------------------------------------------------

class TestRegression:

    # -- Diabetes (442 rows, 10 features) --

    @pytest.fixture()
    def diabetes_csv(self):
        ds = load_diabetes(as_frame=True)
        df = ds.frame.copy()
        df["target"] = ds.target
        path = _make_csv(df)
        yield path, [c for c in df.columns if c != "target"], "target"
        os.unlink(path)

    @pytest.mark.parametrize("algo", ["random_forest", "linear_regression", "xgboost", "lightgbm"])
    def test_diabetes(self, diabetes_csv, algo, benchmark):
        path, features, target = diabetes_csv
        thresholds = {"r2": 0.30, "mae": 0.0}
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "regression",
            "algorithm": algo,
            "test_size": 0.2,
        })
        assert result["status"] == "success", result.get("error")
        m = result["metrics"]
        benchmark(
            dataset="Diabetes (442 rows, 10 features)",
            algorithm=algo,
            task="regression",
            metrics={"r2": m["r2"], "mae": m["mae"]},
            thresholds=thresholds,
        )
        assert m["r2"] >= thresholds["r2"]
        assert m["mae"] > 0

    # -- Synthetic regression (1000 rows, 10 features, known signal) --

    @pytest.fixture()
    def synthetic_csv(self):
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=6,
            noise=10.0, random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["target"] = y
        path = _make_csv(df)
        yield path, [f"f{i}" for i in range(10)], "target"
        os.unlink(path)

    @pytest.mark.parametrize("algo", ["random_forest", "linear_regression", "xgboost", "lightgbm"])
    def test_synthetic_regression(self, synthetic_csv, algo, benchmark):
        path, features, target = synthetic_csv
        thresholds = {"r2": 0.75, "mae": 0.0}
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "regression",
            "algorithm": algo,
            "test_size": 0.2,
        })
        assert result["status"] == "success", result.get("error")
        m = result["metrics"]
        benchmark(
            dataset="Synthetic (1000 rows, 6 informative)",
            algorithm=algo,
            task="regression",
            metrics={"r2": m["r2"], "mae": m["mae"]},
            thresholds=thresholds,
        )
        assert m["r2"] >= thresholds["r2"]
        assert m["mae"] > 0


# ---------------------------------------------------------------------------
# Cross-validation sanity
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Verify that cross-validation runs without error and returns sensible
    fold scores."""

    @pytest.fixture()
    def iris_csv(self):
        ds = load_iris(as_frame=True)
        df = ds.frame.copy()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        path = _make_csv(df)
        yield path, [c for c in df.columns if c != "target"], "target"
        os.unlink(path)

    def test_cv_returns_fold_scores(self, iris_csv, benchmark):
        path, features, target = iris_csv
        result = _train({
            "path": path,
            "features": features,
            "target": target,
            "task": "classification",
            "algorithm": "random_forest",
            "test_size": 0.2,
            "use_cv": True,
            "cv_folds": 5,
        })
        assert result["status"] == "success"
        cv = result["cv_scores"]
        assert cv["folds"] == 5
        assert len(cv["scores"]) == 5
        all_above = all(s >= 0.85 for s in cv["scores"])
        benchmark(
            dataset="Iris (150 rows, 3 classes)",
            algorithm="random_forest",
            task="cross-validation",
            metrics={"mean": cv["mean"]},
            thresholds={"mean": 0.90},
            extra={
                "folds": cv["folds"],
                "scores": cv["scores"],
                "mean": cv["mean"],
                "std": cv["std"],
            },
        )
        assert all_above
        assert cv["mean"] >= 0.90


# ---------------------------------------------------------------------------
# Guardrail / edge-case checks
# ---------------------------------------------------------------------------

class TestGuardrails:
    """Verify that Plume's built-in safety checks work."""

    def test_too_few_rows(self, benchmark):
        df = pd.DataFrame({"x": [1], "y": [0]})
        path = _make_csv(df)
        try:
            result = _train({
                "path": path,
                "features": ["x"],
                "target": "y",
                "task": "classification",
                "algorithm": "random_forest",
            })
            ok = "error" in result
            benchmark(
                dataset="Reject datasets with < 2 rows",
                algorithm="n/a",
                task="guardrail",
                metrics={"pass": 1.0 if ok else 0.0},
                thresholds={"pass": 1.0},
                extra={"detail": "Returned error for 1-row dataset"},
            )
            assert ok
        finally:
            os.unlink(path)

    def test_single_class_rejected(self, benchmark):
        df = pd.DataFrame({"x": list(range(20)), "y": [1] * 20})
        path = _make_csv(df)
        try:
            result = _train({
                "path": path,
                "features": ["x"],
                "target": "y",
                "task": "classification",
                "algorithm": "random_forest",
            })
            ok = "error" in result
            benchmark(
                dataset="Reject single-class targets",
                algorithm="n/a",
                task="guardrail",
                metrics={"pass": 1.0 if ok else 0.0},
                thresholds={"pass": 1.0},
                extra={"detail": "Returned error when target has only 1 unique value"},
            )
            assert ok
        finally:
            os.unlink(path)

    def test_leakage_detection(self, benchmark):
        """A feature that is a copy of the target should trigger a leakage warning."""
        np.random.seed(42)
        target = np.random.randint(0, 100, size=100).astype(float)
        df = pd.DataFrame({
            "real_feature": np.random.randn(100),
            "leaky_feature": target + np.random.randn(100) * 0.01,
            "target": target,
        })
        path = _make_csv(df)
        try:
            result = _train({
                "path": path,
                "features": ["real_feature", "leaky_feature"],
                "target": "target",
                "task": "regression",
                "algorithm": "linear_regression",
            })
            assert result["status"] == "success"
            leaky_cols = [w["feature"] for w in result.get("leakage_warnings", [])]
            ok = "leaky_feature" in leaky_cols
            benchmark(
                dataset="Detect data leakage",
                algorithm="n/a",
                task="guardrail",
                metrics={"pass": 1.0 if ok else 0.0},
                thresholds={"pass": 1.0},
                extra={"detail": "Flagged feature with > 0.95 correlation to target"},
            )
            assert ok
        finally:
            os.unlink(path)

    def test_categorical_features_handled(self, benchmark):
        """Plume should encode categorical features and train successfully."""
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 20,
            "size": ["small", "medium", "large"] * 20,
            "target": [0, 1, 0] * 20,
        })
        path = _make_csv(df)
        try:
            result = _train({
                "path": path,
                "features": ["color", "size"],
                "target": "target",
                "task": "classification",
                "algorithm": "random_forest",
            })
            ok = result.get("status") == "success"
            benchmark(
                dataset="Handle categorical features",
                algorithm="n/a",
                task="guardrail",
                metrics={"pass": 1.0 if ok else 0.0},
                thresholds={"pass": 1.0},
                extra={"detail": "Encoded string columns and trained successfully"},
            )
            assert ok
        finally:
            os.unlink(path)

    def test_missing_values_handled(self, benchmark):
        """Plume should impute missing values and not crash."""
        np.random.seed(42)
        x = np.random.randn(100)
        x[::10] = np.nan  # 10% missing
        df = pd.DataFrame({
            "x": x,
            "y": np.random.randint(0, 2, size=100),
        })
        path = _make_csv(df)
        try:
            result = _train({
                "path": path,
                "features": ["x"],
                "target": "y",
                "task": "classification",
                "algorithm": "random_forest",
            })
            ok = result.get("status") == "success"
            benchmark(
                dataset="Impute missing values (NaN)",
                algorithm="n/a",
                task="guardrail",
                metrics={"pass": 1.0 if ok else 0.0},
                thresholds={"pass": 1.0},
                extra={"detail": "Imputed 10% NaN values and trained successfully"},
            )
            assert ok
        finally:
            os.unlink(path)
