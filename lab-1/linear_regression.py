"""NumPy implementations for laboratory 1; no regularization is used.

The notebook contains the same definitions and can run without importing this
file.  This module also makes the numerical checks independently reproducible.
"""

from __future__ import annotations

import numpy as np


def _features(X):
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional matrix.")
    if not np.isfinite(result).all():
        raise ValueError("X contains NaN or infinity; preprocess it first.")
    return result


def _target(y, n_samples=None):
    result = np.asarray(y, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("y must be a nonempty one-dimensional array.")
    if n_samples is not None and result.size != n_samples:
        raise ValueError("X and y have different numbers of samples.")
    if not np.isfinite(result).all():
        raise ValueError("y contains NaN or infinity.")
    return result


def _paired_targets(y_true, y_pred):
    actual = _target(y_true)
    predicted = _target(y_pred, actual.size)
    return actual, predicted


# %% Metrics
def mse(y_true, y_pred):
    """Mean squared error, in squared units of the target."""
    actual, predicted = _paired_targets(y_true, y_pred)
    return float(np.mean((actual - predicted) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error, in units of the target."""
    actual, predicted = _paired_targets(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def r2(y_true, y_pred):
    """R squared with sklearn's default force_finite=True convention.

    For a constant target return 1 for an exact prediction and 0 otherwise.
    For fewer than two observations R squared is undefined (NaN).
    """
    actual, predicted = _paired_targets(y_true, y_pred)
    if actual.size < 2:
        return float("nan")
    numerator = float(np.sum((actual - predicted) ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def mape(y_true, y_pred):
    """Relative MAPE, as in sklearn; multiply by 100 to obtain a percentage.

    Machine epsilon in the denominator makes the zero-target convention
    explicit. A value near zero in y_true can make this metric very large.
    """
    actual, predicted = _paired_targets(y_true, y_pred)
    denominator = np.maximum(np.abs(actual), np.finfo(np.float64).eps)
    return float(np.mean(np.abs(actual - predicted) / denominator))


# %% Scalers
class ZScoreScaler:
    """z = (x - training_mean) / training_std; population standard deviation."""

    def fit(self, X):
        values = _features(X)
        self.n_features_in_ = values.shape[1]
        self.mean_ = values.mean(axis=0)
        self.var_ = values.var(axis=0, ddof=0)
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X):
        values = _features(X)
        if not hasattr(self, "mean_"):
            raise ValueError("Call fit before transform.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from fit.")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        values = _features(X)
        if not hasattr(self, "mean_"):
            raise ValueError("Call fit before inverse_transform.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from fit.")
        return values * self.scale_ + self.mean_


class MinMaxScaler:
    """Scale to the training range [0, 1]; unseen values are not clipped."""

    def fit(self, X):
        values = _features(X)
        self.n_features_in_ = values.shape[1]
        self.data_min_ = values.min(axis=0)
        self.data_max_ = values.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        self.scale_ = 1.0 / safe_range
        self.min_ = -self.data_min_ * self.scale_
        return self

    def transform(self, X):
        values = _features(X)
        if not hasattr(self, "scale_"):
            raise ValueError("Call fit before transform.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from fit.")
        return values * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        values = _features(X)
        if not hasattr(self, "scale_"):
            raise ValueError("Call fit before inverse_transform.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from fit.")
        return (values - self.min_) / self.scale_


# %% Linear regression
class LinearRegression:
    """Ordinary least squares trained analytically, with GD, or with SGD.

    ``max_iter`` counts gradient steps for GD and complete epochs for SGD.
    All three methods minimize mean squared error without regularization.
    With an intercept, X and y are centered using training statistics; the
    intercept is then recovered as mean(y) - mean(X) @ coef_. This eliminates
    the intercept analytically while preserving the least-squares objective.

    GD uses X.T @ X / n and X.T @ y / n as sufficient statistics. Its update
    is exactly the full-batch gradient, without rereading all n rows each step.
    SGD really visits individual shuffled observations, with a decreasing
    learning rate. ``loss_history_`` contains the initial loss and one loss
    per GD step / SGD epoch; ``n_iter_`` is the number of completed steps.
    """

    def __init__(
        self,
        method="analytic",
        learning_rate=None,
        max_iter=5000,
        tol=1e-9,
        random_state=42,
        fit_intercept=True,
    ):
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        values = _features(X)
        target = _target(y, values.shape[0])
        if self.method not in {"analytic", "gd", "sgd"}:
            raise ValueError("method must be 'analytic', 'gd', or 'sgd'.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        if self.learning_rate is not None and (
            not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be positive and finite.")

        self.n_features_in_ = values.shape[1]
        self.x_mean_ = (
            values.mean(axis=0) if self.fit_intercept else np.zeros(values.shape[1])
        )
        self.y_mean_ = float(target.mean()) if self.fit_intercept else 0.0
        centered_X = np.ascontiguousarray(values - self.x_mean_)
        centered_y = target - self.y_mean_
        self.loss_history_ = [float(np.mean(centered_y**2))]
        self.converged_ = self.method == "analytic"
        self.n_iter_ = 0

        if self.method == "analytic":
            # Analytic formula w = X^+ y, X^+ = V S^+ U.T.
            # Unlike inverse(X.T @ X), it also handles collinear columns.
            left, self.singular_, right = np.linalg.svd(centered_X, full_matrices=False)
            cutoff = (
                np.finfo(np.float64).eps * max(centered_X.shape) * self.singular_[0]
            )
            retained = self.singular_ > cutoff
            self.rank_ = int(retained.sum())
            self.coef_ = right[retained].T @ (
                (left[:, retained].T @ centered_y) / self.singular_[retained]
            )
            self.loss_history_.append(mse(centered_y, centered_X @ self.coef_))
            self.n_iter_ = 1
        else:
            self.coef_ = self._fit_iterative(centered_X, centered_y)

        self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        self.training_mse_ = mse(target, self.predict(values))
        return self

    def _fit_iterative(self, X, y):
        n_samples, n_features = X.shape
        gram = X.T @ X / n_samples
        cross = X.T @ y / n_samples
        target_square = float(y @ y / n_samples)
        weights = np.zeros(n_features)

        def objective(current):
            # A tiny negative value can occur from cancellation near a perfect fit.
            return max(0.0, float(current @ gram @ current - 2 * current @ cross + target_square))

        if self.method == "gd":
            largest_eigenvalue = float(np.linalg.eigvalsh(gram)[-1])
            default_rate = 1.0 / (2.0 * max(largest_eigenvalue, 1e-15))
            self.learning_rate_ = (
                default_rate if self.learning_rate is None else float(self.learning_rate)
            )
            for iteration in range(self.max_iter):
                gradient = 2.0 * (gram @ weights - cross)
                weights -= self.learning_rate_ * gradient
                loss = objective(weights)
                if not np.isfinite(loss):
                    raise FloatingPointError("GD diverged; reduce learning_rate.")
                previous = self.loss_history_[-1]
                self.loss_history_.append(loss)
                self.n_iter_ = iteration + 1
                if abs(previous - loss) <= self.tol * max(1.0, previous):
                    self.converged_ = True
                    break
        else:
            generator = np.random.default_rng(self.random_state)
            largest_row_square = float(np.einsum("ij,ij->i", X, X).max())
            default_rate = min(0.01, 0.5 / max(largest_row_square, 1e-15))
            self.learning_rate_ = (
                default_rate if self.learning_rate is None else float(self.learning_rate)
            )
            updates = 0
            stable_epochs = 0
            best_loss = self.loss_history_[0]
            best_weights = weights.copy()
            for epoch in range(self.max_iter):
                for index in generator.permutation(n_samples):
                    # One observation, one gradient; no batch approximation.
                    eta = self.learning_rate_ / (1.0 + updates / n_samples) ** 0.75
                    error = float(X[index] @ weights - y[index])
                    weights -= 2.0 * eta * error * X[index]
                    updates += 1
                loss = objective(weights)
                if not np.isfinite(loss):
                    raise FloatingPointError("SGD diverged; reduce learning_rate.")
                previous = self.loss_history_[-1]
                self.loss_history_.append(loss)
                self.n_iter_ = epoch + 1
                if loss < best_loss:
                    best_loss, best_weights = loss, weights.copy()
                if abs(previous - loss) <= self.tol * max(1.0, previous):
                    stable_epochs += 1
                else:
                    stable_epochs = 0
                if stable_epochs >= 10:
                    self.converged_ = True
                    break
            # Retain the best complete epoch to avoid a final stochastic fluctuation.
            weights = best_weights
        return weights

    def predict(self, X):
        values = _features(X)
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise ValueError("Call fit before predict.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from fit.")
        return values @ self.coef_ + self.intercept_


# %% Cross-validation
def k_fold_indices(n_samples, n_splits=5, shuffle=True, random_state=42):
    """Yield each manual (train_indices, validation_indices) k-fold split."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    if not isinstance(n_splits, (int, np.integer)) or not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits must be between 2 and n_samples.")
    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.default_rng(random_state).permutation(indices)
    folds = np.array_split(indices, n_splits)
    for fold_number, validation in enumerate(folds):
        training = np.concatenate(
            [fold for number, fold in enumerate(folds) if number != fold_number]
        )
        yield training, validation


def leave_one_out_indices(n_samples):
    """Yield all n genuine leave-one-out train/validation partitions."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    indices = np.arange(n_samples)
    for index in range(n_samples):
        yield np.concatenate((indices[:index], indices[index + 1 :])), indices[index : index + 1]


def loo_predictions_ols(X, y, fit_intercept=True):
    """Exact OLS leave-one-out predictions for a FIXED numeric design matrix.

    PRESS identity: prediction_LOO[i] = y[i] - residual[i] / (1 - leverage[i]).
    Leverages are the diagonal of the orthogonal projector onto the design
    column space, calculated from the retained left singular vectors. This
    also works for collinear columns when deletion preserves the rank.

    If 1 - leverage is almost zero, deletion can reduce the rank. Such cases
    are recomputed directly rather than dividing by zero. This function does
    NOT refit an imputer, encoder or scaler for each fold. To evaluate a whole
    preprocessing pipeline, use leave_one_out_indices and fit it per split.
    """
    values = _features(X)
    target = _target(y, values.shape[0])
    if target.size < 2:
        raise ValueError("LOO requires at least two observations.")
    design = (
        np.column_stack((np.ones(values.shape[0]), values))
        if fit_intercept
        else values
    )
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(design.shape) * singular[0]
    retained = singular > cutoff
    basis = left[:, retained]
    fitted = basis @ (basis.T @ target)
    leverages = np.einsum("ij,ij->i", basis, basis)
    denominator = 1.0 - leverages
    predictions = np.empty_like(target)
    stable = denominator > 1e-10
    predictions[stable] = target[stable] - (target[stable] - fitted[stable]) / denominator[stable]
    for index in np.flatnonzero(~stable):
        keep = np.arange(target.size) != index
        model = LinearRegression(fit_intercept=fit_intercept).fit(values[keep], target[keep])
        predictions[index] = model.predict(values[index : index + 1])[0]
    return predictions
