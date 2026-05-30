"""Tests for BaseMLModel protocol conformance."""

from __future__ import annotations

from src.ml.models.base import BaseMLModel
from src.ml.models.linear import LinearRegressionModel, RidgeRegressionModel
from src.ml.models.logistic import LogisticRegressionModel


def test_linear_satisfies_protocol():
    assert isinstance(LinearRegressionModel(), BaseMLModel)


def test_ridge_satisfies_protocol():
    assert isinstance(RidgeRegressionModel(), BaseMLModel)


def test_logistic_satisfies_protocol():
    assert isinstance(LogisticRegressionModel(), BaseMLModel)


def test_protocol_rejects_object_without_fit_predict():
    class NotAModel:
        pass

    assert not isinstance(NotAModel(), BaseMLModel)


def test_protocol_rejects_partial_implementation():
    class OnlyFit:
        def fit(self, dataset):  # type: ignore[override]
            pass

    assert not isinstance(OnlyFit(), BaseMLModel)
