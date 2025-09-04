"""
Prototype PD Model & Expected Loss Calculator (Retail Loans)
------------------------------------------------------------
- Trains on provided file: "Task 3 and 4_Loan_Data.csv"
- Features used:
    ['credit_lines_outstanding','loan_amt_outstanding','total_debt_outstanding',
     'income','years_employed','fico_score']
- Target: 'default' (1 = defaulted previously, 0 = otherwise)
- Model: GradientBoostingClassifier in a sklearn Pipeline with imputation & scaling
- Outputs:
    * predict_pd(loan_dict) -> PD (probability of default in [0,1])
    * predict_expected_loss(loan_dict, recovery_rate=0.10) -> Expected Loss = PD * (1-recovery_rate) * loan_amt_outstanding
- Includes a simple CLI for quick testing.

Dependencies: pandas, numpy, scikit-learn
"""

from __future__ import annotations
import sys, json
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_CANDIDATES: List[Path] = [
    Path("Task 3 and 4_Loan_Data.csv"),
    Path("./Task 3 and 4_Loan_Data.csv"),
    Path("/mnt/data/Task 3 and 4_Loan_Data.csv"),
]

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

TARGET = "default"

def _find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate 'Task 3 and 4_Loan_Data.csv'. Place it next to this script.")

def _load_data() -> pd.DataFrame:
    path = _find_data()
    df = pd.read_csv(path)
    return df

def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), FEATURES),
        ]
    )
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

# Train at import for a simple prototype experience
_df = _load_data()
_X = _df[FEATURES].copy()
_y = _df[TARGET].astype(int).copy()

# Train/test split for internal validation
X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
_model = _build_pipeline()
_model.fit(X_train, y_train)

# Simple diagnostics (printed when run as script, silent on import unless desired)
try:
    _pd_test = _model.predict_proba(X_test)[:, 1]
    _auc = roc_auc_score(y_test, _pd_test)
    _brier = brier_score_loss(y_test, _pd_test)
    _DIAGNOSTICS = {"roc_auc": float(_auc), "brier": float(_brier)}
except Exception as _e:
    _DIAGNOSTICS = {"roc_auc": None, "brier": None}

def _row_from_dict(loan_dict: Dict[str, Any]) -> pd.DataFrame:
    """Validate keys and make a one-row DataFrame for prediction."""
    row = {k: loan_dict.get(k, np.nan) for k in FEATURES}
    # Basic type coercion to float
    for k in row:
        try:
            row[k] = float(row[k])
        except Exception:
            row[k] = np.nan
    return pd.DataFrame([row])

def predict_pd(loan_dict: Dict[str, Any]) -> float:
    """Return PD in [0,1] for a single loan (dict with required FEATURES)."""
    X_row = _row_from_dict(loan_dict)
    pd_hat = float(_model.predict_proba(X_row)[:, 1][0])
    return pd_hat

def predict_expected_loss(loan_dict: Dict[str, Any], recovery_rate: float = 0.10) -> float:
    """
    Expected Loss = PD * (1 - recovery_rate) * EAD
    Here, EAD is taken as 'loan_amt_outstanding' from the input loan_dict.
    """
    if "loan_amt_outstanding" not in loan_dict:
        raise ValueError("loan_dict must include 'loan_amt_outstanding' to compute expected loss.")
    pd_hat = predict_pd(loan_dict)
    ead = float(loan_dict["loan_amt_outstanding"])
    lgd = 1.0 - float(recovery_rate)
    el = pd_hat * lgd * ead
    return float(el)

def diagnostics() -> Dict[str, float]:
    """Return basic holdout metrics for reference."""
    return dict(_DIAGNOSTICS)

# ------------------------------ CLI ------------------------------
CLI_HELP = f"""
Usage:
  python loan_pd_model.py '{{"credit_lines_outstanding": 2, "loan_amt_outstanding": 5000,
                              "total_debt_outstanding": 12000, "income": 45000,
                              "years_employed": 3, "fico_score": 620}}'

  # Or interactively with no args (you'll be prompted)

Features required:
{FEATURES}
"""

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] in {"-h", "--help"}:
            print(CLI_HELP)
            sys.exit(0)
        try:
            loan = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            print("First argument must be a JSON object with feature values.")
            print(CLI_HELP)
            sys.exit(1)
    else:
        print("Enter loan fields as prompted (press Enter to leave blank / use NaN):")
        loan = {}
        for f in FEATURES:
            val = input(f"{f}: ").strip()
            loan[f] = float(val) if val != "" else np.nan

    pd_hat = predict_pd(loan)
    el = predict_expected_loss(loan, recovery_rate=0.10)
    print(json.dumps({
        "input": loan,
        "metrics": diagnostics(),
        "prob_default": pd_hat,
        "expected_loss": el,
        "assumptions": {"recovery_rate": 0.10, "EAD_field": "loan_amt_outstanding"}
    }, indent=2))
