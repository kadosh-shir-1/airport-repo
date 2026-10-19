"""Deterministic normalization and weighted scoring (pure functions, no DB, no LLM).

All metrics are oriented so that higher = stronger expansion signal
(more growth, fuller planes, more congestion, bigger airport).
"""
import numpy as np
import pandas as pd

import config


def normalize(values, method="minmax", log=False):
    """Scale a Series to 0-100 within the peer group. NaN stays NaN.

    A single airport, or all airports tied, get 50 (no divide-by-zero, no false ranking).
    """
    s = pd.Series(values, dtype=float)
    if log:
        s = np.log10(s.clip(lower=1))
    valid = s.dropna()
    if valid.empty:
        return s
    if method == "percentile":
        if len(valid) == 1:
            return s.where(s.isna(), 50.0)
        return (s.rank(method="average") - 1) / (len(valid) - 1) * 100
    if method != "minmax":
        raise ValueError(f"Unknown normalization method: {method}")
    lo, hi = valid.min(), valid.max()
    if hi - lo < 1e-12:
        return s.where(s.isna(), 50.0)
    return (s - lo) / (hi - lo) * 100


def validate_weights(score_name, overrides=None):
    """Default weights for score_name, updated with overrides and re-normalized to sum to 1."""
    if score_name not in config.SCORE_WEIGHTS:
        raise ValueError(f"Unknown score '{score_name}'. Options: {sorted(config.SCORE_WEIGHTS)}")
    weights = dict(config.SCORE_WEIGHTS[score_name])
    for key, value in (overrides or {}).items():
        if key not in weights:
            raise ValueError(f"Unknown weight '{key}' for score '{score_name}'. Options: {sorted(weights)}")
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Weight '{key}' must be a non-negative number, got {value!r}")
        weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return {k: v / total for k, v in weights.items()}


def _normalized_metric(df, metric, method):
    if metric == "congestion":
        components = [normalize(df[c], method) for c in config.CONGESTION_COMPONENTS]
        return pd.concat(components, axis=1).mean(axis=1)  # average of available components
    return normalize(df[metric], method, log=metric in config.LOG_SCALE_METRICS)


def score_airports(df, score_name="investment", weights=None, method="minmax"):
    """Score every airport in df (the peer group) and explain each score.

    Returns a dict with:
      weights   - weights actually used (validated, sum to 1)
      scores    - DataFrame [iata, score, rank], best first
      breakdown - DataFrame [iata, metric, raw, normalized, weight, contribution];
                  contributions of an airport sum to its score
      caveats   - list of data-quality notes (e.g. missing metric, weight redistributed)
    """
    w = validate_weights(score_name, weights)
    df = df.reset_index(drop=True)
    norm = pd.DataFrame({m: _normalized_metric(df, m, method) for m in w})

    # Per airport: drop weights of missing metrics and re-normalize the rest.
    weight_matrix = pd.DataFrame({m: w[m] for m in w}, index=df.index).where(norm.notna(), 0.0)
    effective = weight_matrix.div(weight_matrix.sum(axis=1).replace(0, np.nan), axis=0)
    contributions = norm.fillna(0) * effective
    score = contributions.sum(axis=1, min_count=1).where(effective.notna().all(axis=1))

    caveats = []
    for m in w:
        missing = df.loc[norm[m].isna(), "iata"].tolist()
        if missing:
            caveats.append(f"{m} missing for {', '.join(missing)}; its weight was redistributed for those airports")
    if "congestion" in w:
        for i in df.index[norm["congestion"].notna()]:
            absent = [c for c in config.CONGESTION_COMPONENTS if pd.isna(df.at[i, c])]
            if absent:
                present = [c for c in config.CONGESTION_COMPONENTS if c not in absent]
                caveats.append(f"congestion for {df.at[i, 'iata']} is based only on {', '.join(present)} "
                               f"(missing: {', '.join(absent)})")

    scores = pd.DataFrame({"iata": df["iata"], "score": score}).sort_values("score", ascending=False)
    scores["rank"] = scores["score"].rank(ascending=False, method="min")
    breakdown = pd.DataFrame([
        {
            "iata": df.at[i, "iata"],
            "metric": m,
            "raw": df.at[i, m] if m in df.columns else np.nan,
            "normalized": norm.at[i, m],
            "weight": effective.at[i, m],
            "contribution": contributions.at[i, m],
        }
        for i in df.index for m in w
    ])
    return {"weights": w, "scores": scores.reset_index(drop=True), "breakdown": breakdown, "caveats": caveats}
