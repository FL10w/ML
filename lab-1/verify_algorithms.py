"""Numerical verification against sklearn; run: python verify_algorithms.py."""

import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler

from linear_regression import (
    LinearRegression,
    MinMaxScaler,
    ZScoreScaler,
    k_fold_indices,
    leave_one_out_indices,
    loo_predictions_ols,
    mae,
    mape,
    mse,
    r2,
)


def verify():
    generator = np.random.default_rng(2026)
    training = generator.normal(size=(500, 5))
    training[:, -1] = 7.0
    validation = generator.normal(size=(37, 5))
    validation[:, -1] = 7.0
    for own, reference in [
        (ZScoreScaler(), StandardScaler()),
        (MinMaxScaler(), SklearnMinMaxScaler()),
    ]:
        scaled = own.fit_transform(training)
        assert_allclose(scaled, reference.fit_transform(training), atol=1e-12)
        assert_allclose(own.transform(validation), reference.transform(validation), atol=1e-12)
        assert_allclose(own.inverse_transform(scaled), training, atol=1e-12)

    for actual, predicted in [
        (np.array([0.0, -2.0, 3.0]), np.array([0.25, -1.5, 4.0])),
        (np.ones(4), np.ones(4)),
        (np.ones(4), np.zeros(4)),
        (generator.normal(size=50), generator.normal(size=50)),
    ]:
        for own, reference in [
            (mse, mean_squared_error),
            (mae, mean_absolute_error),
            (r2, r2_score),
            (mape, mean_absolute_percentage_error),
        ]:
            assert_allclose(own(actual, predicted), reference(actual, predicted), atol=1e-12)

    X = generator.normal(size=(600, 6))
    X[:, -1] = X[:, 0] + X[:, 1]  # Intentionally rank deficient.
    y = 4.0 + X @ np.array([1.2, -3.0, 0.5, 2.0, -0.3, 0.7]) + generator.normal(0, 0.05, 600)
    reference = SklearnLinearRegression().fit(X, y)
    analytic = LinearRegression().fit(X, y)
    assert_allclose(analytic.predict(X), reference.predict(X), atol=1e-10)
    assert_allclose(analytic.intercept_, reference.intercept_, atol=1e-10)
    gd = LinearRegression(method="gd", max_iter=10000, tol=1e-13).fit(X, y)
    sgd = LinearRegression(method="sgd", max_iter=160, tol=1e-12).fit(X, y)
    assert_allclose(gd.predict(X), analytic.predict(X), atol=1e-5)
    assert_allclose(sgd.predict(X), analytic.predict(X), atol=0.01)
    repeated = LinearRegression(method="sgd", max_iter=160, tol=1e-12).fit(X, y)
    assert_allclose(sgd.coef_, repeated.coef_, atol=0.0, rtol=0.0)

    for n_samples, n_splits in [(11, 3), (5, 5)]:
        splits = list(k_fold_indices(n_samples, n_splits, shuffle=True))
        assert_allclose(np.sort(np.concatenate([test for _, test in splits])), np.arange(n_samples))
        for train, test in splits:
            assert np.intersect1d(train, test).size == 0
            assert train.size + test.size == n_samples
        own = k_fold_indices(n_samples, n_splits, shuffle=False)
        for (train, test), (ref_train, ref_test) in zip(own, KFold(n_splits).split(np.arange(n_samples))):
            assert_allclose(train, ref_train)
            assert_allclose(test, ref_test)

    for (train, test), (ref_train, ref_test) in zip(
        leave_one_out_indices(11), LeaveOneOut().split(np.arange(11))
    ):
        assert_allclose(train, ref_train)
        assert_allclose(test, ref_test)

    # Ordinary, collinear, and rank-changing deletion cases for PRESS.
    for design in [
        generator.normal(size=(23, 4)),
        np.column_stack((np.arange(12), np.arange(12) * 2, np.ones(12))),
        np.eye(5),
        np.zeros((5, 2)),
    ]:
        target = generator.normal(size=design.shape[0])
        for fit_intercept in [True, False]:
            expected = np.empty(target.size)
            for train, test in leave_one_out_indices(target.size):
                expected[test] = SklearnLinearRegression(fit_intercept=fit_intercept).fit(
                    design[train], target[train]
                ).predict(design[test])
            actual = loo_predictions_ols(design, target, fit_intercept=fit_intercept)
            assert_allclose(actual, expected, atol=1e-9)

    print("All algorithm checks passed: scalers, four metrics, OLS/GD/SGD, k-fold and exact LOO.")
    print(f"Synthetic training MSE: OLS={analytic.training_mse_:.8f}; GD={gd.training_mse_:.8f}; SGD={sgd.training_mse_:.8f}")


if __name__ == "__main__":
    verify()
