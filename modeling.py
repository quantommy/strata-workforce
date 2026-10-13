"""Reusable, training-fold-safe preprocessing for automation-risk classification."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

TARGET = "Automation_Risk"
RISK_ORDER = ["Low", "Medium", "High"]
CATEGORICAL_COLUMNS = [
    "Job_Title", "Industry", "Company_Size", "Location", "AI_Adoption_Level",
    "Required_Skills", "Remote_Friendly", "Job_Growth_Projection",
]
INPUT_COLUMNS = CATEGORICAL_COLUMNS + ["Salary_USD"]
ENGINEERED_CATEGORICAL = ["Role_Industry", "Role_Skill", "Industry_Adoption"]
ENGINEERED_NUMERIC = [
    "Log_Salary", "Salary_vs_Role_Median", "Salary_vs_Location_Median",
]


class ProfileFeatures(TransformerMixin, BaseEstimator):
    """Preserve raw predictors and optionally add interactions and salary context.

    Group medians and missing-salary fallback are learned only in fit(). Unknown
    roles or locations use the training-fold global salary median.
    """

    def __init__(self, engineered=False):
        self.engineered = engineered

    def _clean(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provide job profiles as a pandas DataFrame.")
        missing = set(INPUT_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        out = X.loc[:, INPUT_COLUMNS].copy()
        for column in CATEGORICAL_COLUMNS:
            out[column] = out[column].fillna("Unknown").astype(str)
        out["Salary_USD"] = pd.to_numeric(out["Salary_USD"], errors="raise")
        nonmissing_salary = out["Salary_USD"].dropna()
        if ((nonmissing_salary <= 0) | ~np.isfinite(nonmissing_salary)).any():
            raise ValueError("Salary_USD must be a finite positive value or missing.")
        return out

    def fit(self, X, y=None):
        clean = self._clean(X)
        self.salary_median_ = float(clean["Salary_USD"].median())
        if not np.isfinite(self.salary_median_):
            raise ValueError("Training profiles must contain at least one salary.")
        self.role_medians_ = clean.groupby("Job_Title")["Salary_USD"].median()
        self.location_medians_ = clean.groupby("Location")["Salary_USD"].median()
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def transform(self, X):
        clean = self._clean(X)
        clean["Salary_USD"] = clean["Salary_USD"].fillna(self.salary_median_)
        if self.engineered:
            clean["Log_Salary"] = np.log(clean["Salary_USD"])
            for group, medians, output in [
                ("Job_Title", self.role_medians_, "Salary_vs_Role_Median"),
                ("Location", self.location_medians_, "Salary_vs_Location_Median"),
            ]:
                reference = clean[group].map(medians).fillna(self.salary_median_)
                clean[output] = clean["Salary_USD"] / reference
            for a, b, output in [
                ("Job_Title", "Industry", "Role_Industry"),
                ("Job_Title", "Required_Skills", "Role_Skill"),
                ("Industry", "AI_Adoption_Level", "Industry_Adoption"),
            ]:
                clean[output] = clean[a] + " | " + clean[b]
        return clean

    def get_feature_names_out(self, input_features=None):
        names = INPUT_COLUMNS.copy()
        if self.engineered:
            names += ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
        return np.array(names, dtype=object)


def make_model(kind, engineered=False, seed=42):
    features = ProfileFeatures(engineered=engineered)
    categorical = CATEGORICAL_COLUMNS + (ENGINEERED_CATEGORICAL if engineered else [])
    numeric = ["Salary_USD"] + (ENGINEERED_NUMERIC if engineered else [])
    if kind == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=180, depth=4, learning_rate=0.05, l2_leaf_reg=8,
            loss_function="MultiClass", cat_features=tuple(categorical),
            random_seed=seed, thread_count=2, verbose=False,
            allow_writing_files=False,
        )
        return Pipeline([("features", features), ("classifier", model)])
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    if kind == "logistic":
        model = LogisticRegression(C=0.3, max_iter=2000, random_state=seed)
    elif kind == "forest":
        model = RandomForestClassifier(
            n_estimators=180, max_depth=6, min_samples_leaf=5,
            random_state=seed, n_jobs=2,
        )
    elif kind == "majority":
        model = DummyClassifier(strategy="most_frequent", random_state=seed)
    elif kind == "stratified":
        model = DummyClassifier(strategy="stratified", random_state=seed)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([
        ("features", features), ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def candidate_models(seed=42):
    return {
        "Baseline: majority": make_model("majority", seed=seed),
        "Baseline: stratified random": make_model("stratified", seed=seed),
        "Logistic: raw": make_model("logistic", seed=seed),
        "Logistic: engineered": make_model("logistic", engineered=True, seed=seed),
        "Random forest: raw": make_model("forest", seed=seed),
        "CatBoost: raw": make_model("catboost", seed=seed),
        "CatBoost: engineered": make_model("catboost", engineered=True, seed=seed),
    }


def make_mlp(hidden_layer_sizes=(32,), solver="lbfgs", alpha=1.0, seed=42):
    """Use the same raw-input preprocessing as the original classifiers."""
    pipeline = make_model("logistic", seed=seed)
    network = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver=solver,
        alpha=alpha, random_state=seed, learning_rate_init=0.001,
        max_iter=1500 if solver == "lbfgs" else 800,
        max_fun=30000, early_stopping=(solver == "adam"),
        validation_fraction=0.2, n_iter_no_change=30,
    )
    return pipeline.set_params(classifier=network)
