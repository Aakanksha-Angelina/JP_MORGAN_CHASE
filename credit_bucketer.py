"""
Credit Rating Bucketer (FICO → Ratings via Optimal Quantization)
----------------------------------------------------------------
- Input data: "Task 3 and 4_Loan_Data (1) for task 4.csv" with columns:
    * fico_score (int)
    * default (0/1)
- Goal: Given K buckets, find FICO boundaries that best summarize data.
- Methods:
    1) "mse"      : minimize within-bucket squared error (optimal 1D DP).
    2) "loglik"   : maximize sum over buckets of k*log(k/n) + (n-k)*log(1-k/n) (dynamic programming).

- Output:
    * Boundaries (inclusive ranges) for K buckets.
    * A rating map where LOWER rating = BETTER credit (rating 1 = highest FICO bucket).

- CLI examples:
    python credit_bucketer.py --K 10 --method loglik
    python credit_bucketer.py --K 8  --method mse --dump_map buckets.json

- Programmatic use:
    from credit_bucketer import fit_buckets, assign_ratings
    bounds, info = fit_buckets(K=10, method="loglik")
    ratings = assign_ratings(df["fico_score"], bounds, low_is_good=True)

References:
- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)
- Likelihood function: https://en.wikipedia.org/wiki/Likelihood_function
- Dynamic programming: https://en.wikipedia.org/wiki/Dynamic_programming#Computer_programming
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import math
import json

DATA_CANDIDATES = [
    Path("Task 3 and 4_Loan_Data (1) for task 4.csv"),
    Path("./Task 3 and 4_Loan_Data (1) for task 4.csv"),
    Path("/mnt/data/Task 3 and 4_Loan_Data (1) for task 4.csv"),
]

FICO_COL = "fico_score"
DEFAULT_COL = "default"

@dataclass
class Histogram:
    scores: np.ndarray   # sorted unique FICO scores
    n: np.ndarray        # count per unique score
    k: np.ndarray        # default count per unique score

def _load_df(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        for p in DATA_CANDIDATES:
            if p.exists():
                path = p
                break
    if path is None:
        raise FileNotFoundError("Could not find the loan data CSV for task 4.")
    df = pd.read_csv(path)
    if FICO_COL not in df.columns or DEFAULT_COL not in df.columns:
        raise ValueError(f"Expected columns '{FICO_COL}' and '{DEFAULT_COL}' not found.")
    return df

def _build_hist(df: pd.DataFrame) -> Histogram:
    grp = df.groupby(FICO_COL, as_index=False)[DEFAULT_COL].agg(['count','sum']).reset_index()
    grp.columns = [FICO_COL, 'n','k']
    grp = grp.sort_values(FICO_COL)
    scores = grp[FICO_COL].to_numpy(int)
    n = grp['n'].to_numpy(int)
    k = grp['k'].to_numpy(int)
    return Histogram(scores, n, k)

# ---------- Prefix sums utilities for fast range computations ----------

@dataclass
class PrefixSums:
    w: np.ndarray       # weights (n)
    x: np.ndarray       # scores
    wx: np.ndarray      # prefix sum of w*x
    wx2: np.ndarray     # prefix sum of w*x^2
    wsum: np.ndarray    # prefix sum of w
    ksum: np.ndarray    # prefix sum of defaults

def _prefix(hist: Histogram) -> PrefixSums:
    w = hist.n.astype(float)
    x = hist.scores.astype(float)
    wx = np.cumsum(w * x)
    wx2 = np.cumsum(w * x * x)
    wsum = np.cumsum(w)
    ksum = np.cumsum(hist.k.astype(float))
    return PrefixSums(w=w, x=x, wx=wx, wx2=wx2, wsum=wsum, ksum=ksum)

def _range_sum(prefix: np.ndarray, i: int, j: int) -> float:
    return float(prefix[j] - (prefix[i-1] if i > 0 else 0.0))

# SSE cost for MSE objective on [i..j]
def _sse_cost(ps: PrefixSums, i: int, j: int) -> float:
    w = _range_sum(ps.wsum, i, j)
    if w <= 0:
        return 0.0
    sx = _range_sum(ps.wx, i, j)
    sx2 = _range_sum(ps.wx2, i, j)
    # SSE = sum w*(x - mean)^2 = sum (w*x^2) - (sum w*x)^2 / (sum w)
    return sx2 - (sx * sx) / w

# Log-likelihood score for bucket [i..j] using MLE p=k/n (0 log 0 handled)
def _loglik_score(ps: PrefixSums, i: int, j: int) -> float:
    n = _range_sum(ps.wsum, i, j)
    k = _range_sum(ps.ksum, i, j)
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0  # k*log(0) term -> define 0*log0 = 0, remaining term (n)*log(1) = 0
    if k >= n:
        return 0.0  # (n-k)*log(0) term -> define as 0 similarly (maximum at p=1 gives 0 here)
    p = k / n
    return float(k * math.log(p) + (n - k) * math.log(1.0 - p))

# ---------- DP partition (generic) ----------

def _dp_partition(K: int, M: int, cost_fn, minimize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    DP over the index 0..M-1 (inclusive). cost_fn(i,j) gives cost of bucket [i..j].
    Returns (dp, prev) where dp[k,j] is best cost for k buckets ending at j, and prev traces argmin/argmax.
    """
    INF = 1e300
    sign = 1.0 if minimize else -1.0
    dp = np.full((K+1, M), np.inf)
    prev = np.full((K+1, M), -1, dtype=int)

    # Base case: 1 bucket covering [0..j]
    for j in range(M):
        dp[1, j] = cost_fn(0, j) * sign
        prev[1, j] = -1

    for k in range(2, K+1):
        for j in range(k-1, M):  # need at least k items to make k buckets
            best = np.inf
            best_i = -1
            # split at i (bucket is [i..j], previous is up to i-1)
            # enforce at least one item per bucket
            for i in range(k-2, j):
                v = dp[k-1, i] + cost_fn(i+1, j) * sign
                if v < best:
                    best = v
                    best_i = i
            dp[k, j] = best
            prev[k, j] = best_i

    return dp, prev

def _reconstruct_bounds(prev: np.ndarray, K: int, M: int) -> List[Tuple[int,int]]:
    """Return list of index ranges [(i0,j0),...,(iK-1,jK-1)] covering 0..M-1."""
    bounds = []
    j = M - 1
    for k in range(K, 0, -1):
        i = prev[k, j]
        start = 0 if i == -1 else i + 1
        bounds.append((start, j))
        j = i
    bounds.reverse()
    return bounds

# ---------- Public API ----------

def fit_buckets(K: int = 10, method: str = "loglik", data_path: Optional[str] = None):
    """
    Fit K buckets on FICO using "mse" or "loglik" objective.
    Returns (boundaries, info) where boundaries is a list of (low, high) inclusive FICO bounds.
    """
    df = _load_df(Path(data_path) if data_path else None)
    hist = _build_hist(df)
    ps = _prefix(hist)
    M = len(hist.scores)

    if method.lower() == "mse":
        cost_fn = lambda i, j: _sse_cost(ps, i, j)
        minimize = True
    elif method.lower() == "loglik":
        # maximize sum of log-likelihoods -> minimize negative
        cost_fn = lambda i, j: -_loglik_score(ps, i, j)
        minimize = True
    else:
        raise ValueError("method must be one of {'mse','loglik'}")

    dp, prv = _dp_partition(K, M, cost_fn, minimize=minimize)
    idx_bounds = _reconstruct_bounds(prv, K, M)

    # Convert index bounds to FICO ranges
    boundaries: List[Tuple[int,int]] = []
    for (i, j) in idx_bounds:
        lo = int(hist.scores[i])
        hi = int(hist.scores[j])
        boundaries.append((lo, hi))

    # Build rating mapping: rating 1 = highest FICO bucket (best), K = lowest
    # Sort boundaries by lower bound then reverse to get highest-first
    boundaries_sorted = sorted(boundaries, key=lambda t: (t[0], t[1]))
    boundaries_desc = list(reversed(boundaries_sorted))

    rating_map = []
    for rank, (lo, hi) in enumerate(boundaries_desc, start=1):
        rating_map.append({"rating": rank, "fico_low": lo, "fico_high": hi})

    info = {
        "method": method,
        "K": K,
        "objective_value": float(dp[K, M-1]) if minimize else -float(dp[K, M-1]),
        "counts_per_bucket": [
            int(np.sum(hist.n[(hist.scores >= lo) & (hist.scores <= hi)])) for (lo, hi) in boundaries
        ],
        "defaults_per_bucket": [
            int(np.sum(hist.k[(hist.scores >= lo) & (hist.scores <= hi)])) for (lo, hi) in boundaries
        ],
    }
    return rating_map, info

def assign_ratings(fico_series: pd.Series, rating_map: List[Dict], low_is_good: bool = True) -> pd.Series:
    """
    Assign ratings to a pandas Series of FICO scores using a rating_map from fit_buckets().
    If low_is_good=True, rating 1 is best (highest FICO range). This function expects
    rating_map entries already numbered that way.
    """
    # Build interval lookup
    def rate_one(f: float) -> int:
        for entry in rating_map:
            lo, hi = entry["fico_low"], entry["fico_high"]
            if lo <= f <= hi:
                return int(entry["rating"])
        # If outside trained range, clamp to nearest bucket
        if f < rating_map[-1]["fico_low"]:
            return int(rating_map[-1]["rating"])
        if f > rating_map[0]["fico_high"]:
            return int(rating_map[0]["rating"])
        return int(rating_map[-1]["rating"])
    return fico_series.apply(rate_one).astype(int)

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10, help="Number of rating buckets")
    ap.add_argument("--method", type=str, default="loglik", choices=["loglik","mse"], help="Optimization objective")
    ap.add_argument("--data", type=str, default=None, help="Path to CSV (optional)")
    ap.add_argument("--dump_map", type=str, default=None, help="Path to write rating_map JSON")
    args = ap.parse_args()

    rating_map, info = fit_buckets(K=args.K, method=args.method, data_path=args.data)

    print("=== Rating Map (rating 1 = best) ===")
    for r in rating_map:
        print(f"Rating {r['rating']:2d}: {r['fico_low']} .. {r['fico_high']}")

    print("\n=== Info ===")
    print(json.dumps(info, indent=2))

    if args.dump_map:
        with open(args.dump_map, "w") as f:
            json.dump({"rating_map": rating_map, "info": info}, f, indent=2)
        print(f"\nSaved rating_map to {args.dump_map}")

if __name__ == "__main__":
    main()
