"""Train-only preprocessing for the linear-regression laboratory.

The engineered design is still linear in its learned coefficients. Logs,
polynomial terms, and binary interactions are deterministic feature maps;
no alternative prediction model or regularization is used.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from linear_regression import ZScoreScaler


def clean_training_data(frame):
    """Remove objectively damaged observations and report each removal.

    The missing-data block affects nine predictors simultaneously. The test
    set is complete, so these damaged training observations are excluded;
    the preprocessor still supports imputation for future incomplete inputs.
    Valid large financial values are retained. The target cut only removes
    values of order ten million, separated from ordinary scores by five
    orders of magnitude; no quantile trimming of valid targets is performed.
    """
    if "RiskScore" not in frame:
        raise ValueError("The training frame must contain RiskScore.")
    data = frame.copy()
    audit = []

    def record(step, mask):
        nonlocal data
        removed = int(mask.sum())
        data = data.loc[~mask].copy()
        audit.append({"step": step, "removed": removed, "remaining": len(data)})

    audit.append({"step": "Исходные данные", "removed": 0, "remaining": len(data)})
    record("Полностью пустые строки", data.isna().all(axis=1))
    record("Полные дубли", data.duplicated())
    target = pd.to_numeric(data["RiskScore"], errors="coerce")
    record("Отсутствующая или явно повреждённая цель", ~np.isfinite(target) | target.abs().ge(1e6))
    predictors = data.drop(columns=["RiskScore", "ID"], errors="ignore")
    record("Повреждённые строки с блоком пропусков", predictors.isna().sum(axis=1).ge(9))
    return data.reset_index(drop=True), pd.DataFrame(audit)


class TabularPreprocessor:
    """Median imputation, one-hot encoding, and own z-score normalization.

    With ``engineered=True``, add logarithms of financial quantities and
    degree 2--4 polynomial terms over six influential quantitative fields.
    Their smooth nonlinear relationships are represented by a linear model
    over an expanded design matrix. The six polynomial inputs are clipped
    at training 0.1/99.9 percentiles before centering and scaling to prevent
    high monetary tails from exploding under fourth powers. Fixed feature
    definitions are reused for every split; all fitted values come from its
    training part only.
    """

    MONEY_COLUMNS = (
        "AnnualIncome", "LoanAmount", "MonthlyDebtPayments",
        "SavingsAccountBalance", "CheckingAccountBalance", "TotalAssets",
        "TotalLiabilities", "MonthlyIncome", "NetWorth", "MonthlyLoanPayment",
    )
    POLYNOMIAL_COLUMNS = (
        "CreditScore", "MonthlyIncome", "NetWorth",
        "TotalDebtToIncomeRatio", "InterestRate", "DebtToIncomeRatio",
    )
    LOG_POLYNOMIAL_COLUMNS = ("MonthlyIncome", "NetWorth", "TotalDebtToIncomeRatio")

    def __init__(self, engineered=False):
        self.engineered = bool(engineered)

    @staticmethod
    def _predictors(frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Pass a pandas DataFrame with named columns.")
        return frame.drop(columns=["ID", "RiskScore", "ApplicationDate"], errors="ignore").copy()

    def _filled(self, frame):
        data = self._predictors(frame)
        missing_columns = set(self.input_columns_) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing input columns: {sorted(missing_columns)}")
        data = data.loc[:, self.input_columns_]
        for col in self.numeric_columns_:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(self.medians_[col])
        for col in self.categorical_columns_:
            data[col] = data[col].astype("string").fillna("__missing__")
        return data

    def _polynomial_inputs(self, data):
        inputs = data.loc[:, self.POLYNOMIAL_COLUMNS].copy()
        for col in self.LOG_POLYNOMIAL_COLUMNS:
            inputs[col] = np.log1p(inputs[col].clip(lower=0))
        return inputs

    def _design(self, data):
        columns = {col: data[col].to_numpy(dtype=float) for col in self.numeric_columns_}
        for col, levels in self.categories_.items():
            # Dropping the first training category avoids the dummy-variable trap.
            # An unseen category maps to zero in every dummy of this field.
            for level in levels[1:]:
                columns[f"{col}={level}"] = data[col].eq(level).to_numpy(dtype=float)
        if self.engineered:
            for col in self.MONEY_COLUMNS:
                columns[f"log1p({col})"] = np.log1p(data[col].clip(lower=0).to_numpy(dtype=float))
            columns["DTI_capped_1"] = data["TotalDebtToIncomeRatio"].clip(upper=1).to_numpy(dtype=float)
            poly = self._polynomial_inputs(data)
            poly = poly.clip(lower=self.poly_lower_, upper=self.poly_upper_, axis=1)
            values = (poly.to_numpy(dtype=float) - self.poly_mean_) / self.poly_scale_
            for indices in self.polynomial_terms_:
                label = " * ".join(self.POLYNOMIAL_COLUMNS[i] for i in indices)
                columns[f"poly({label})"] = np.prod(values[:, indices], axis=1)
            indicators = {
                "BankruptcyHistory": data["BankruptcyHistory"].to_numpy(dtype=float),
                "PreviousLoanDefaults": data["PreviousLoanDefaults"].to_numpy(dtype=float),
                "Unemployed": data["EmploymentStatus"].eq("Unemployed").to_numpy(dtype=float),
                "SelfEmployed": data["EmploymentStatus"].eq("Self-Employed").to_numpy(dtype=float),
            }
            for indicator_name, flag in indicators.items():
                for j, col in enumerate(self.POLYNOMIAL_COLUMNS):
                    columns[f"{indicator_name} * scaled({col})"] = flag * values[:, j]
        result = pd.DataFrame(columns, index=data.index)
        return result

    def fit(self, frame):
        raw = self._predictors(frame)
        if raw.empty:
            raise ValueError("Cannot fit preprocessing on an empty frame.")
        self.input_columns_ = list(raw.columns)
        self.numeric_columns_ = list(raw.select_dtypes(include="number").columns)
        self.categorical_columns_ = [col for col in raw if col not in self.numeric_columns_]
        self.medians_ = raw[self.numeric_columns_].replace([np.inf, -np.inf], np.nan).median().fillna(0)
        data = self._filled(frame)
        self.categories_ = {
            col: sorted(data[col].unique().tolist())
            for col in self.categorical_columns_
        }
        if self.engineered:
            poly = self._polynomial_inputs(data)
            self.poly_lower_ = poly.quantile(0.001)
            self.poly_upper_ = poly.quantile(0.999)
            poly = poly.clip(lower=self.poly_lower_, upper=self.poly_upper_, axis=1)
            self.poly_mean_ = poly.to_numpy(dtype=float).mean(axis=0)
            self.poly_scale_ = poly.to_numpy(dtype=float).std(axis=0)
            self.poly_scale_[self.poly_scale_ == 0] = 1.0
            self.polynomial_terms_ = [
                indices
                for degree in range(2, 5)
                for indices in combinations_with_replacement(range(len(self.POLYNOMIAL_COLUMNS)), degree)
            ]
        design = self._design(data)
        self.feature_names_ = list(design.columns)
        self.scaler_ = ZScoreScaler().fit(design.to_numpy(dtype=float))
        return self

    def transform(self, frame):
        if not hasattr(self, "scaler_"):
            raise ValueError("Call fit before transform.")
        design = self._design(self._filled(frame))
        return self.scaler_.transform(design.loc[:, self.feature_names_].to_numpy(dtype=float))

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
