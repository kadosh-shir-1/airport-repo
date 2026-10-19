import math

import pandas as pd
import pytest

import config
from scoring import normalize, score_airports, validate_weights


# --- normalize ---------------------------------------------------------------
def test_minmax_scales_to_0_100():
    assert normalize([10, 20, 30]).tolist() == [0, 50, 100]


def test_single_airport_gets_50():
    assert normalize([42]).tolist() == [50]


def test_all_ties_get_50():
    assert normalize([5, 5, 5]).tolist() == [50, 50, 50]


def test_nan_is_preserved():
    result = normalize([10, None, 30])
    assert result[0] == 0 and math.isnan(result[1]) and result[2] == 100


def test_log_scale_compresses_large_values():
    # 1M, 10M, 100M passengers -> evenly spaced on a log scale
    assert normalize([1e6, 1e7, 1e8], log=True).round(6).tolist() == [0, 50, 100]


def test_percentile_handles_outliers():
    # min-max squashes the middle values because of the outlier; percentile does not
    assert normalize([1, 2, 3, 1000], method="percentile").round(2).tolist() == [0, 33.33, 66.67, 100]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        normalize([1, 2], method="zscore")


# --- validate_weights --------------------------------------------------------
def test_default_weights_sum_to_one():
    for name in config.SCORE_WEIGHTS:
        assert sum(validate_weights(name).values()) == pytest.approx(1)


def test_override_is_renormalized():
    w = validate_weights("unmet_demand", {"yoy_growth": 2.0})
    # defaults are 1/3 each; growth=2 -> 2 / (1/3 + 1/3 + 2)
    assert w["yoy_growth"] == pytest.approx(2 / (2 + 2 / 3))
    assert sum(w.values()) == pytest.approx(1)


@pytest.mark.parametrize("overrides", [{"unknown_metric": 0.5}, {"load_factor": -1}, {"load_factor": "high"}])
def test_invalid_overrides_raise(overrides):
    with pytest.raises(ValueError):
        validate_weights("investment", overrides)


def test_all_zero_weights_raise():
    zeros = {k: 0 for k in config.SCORE_WEIGHTS["congestion"]}
    with pytest.raises(ValueError):
        validate_weights("congestion", zeros)


def test_unknown_score_raises():
    with pytest.raises(ValueError):
        validate_weights("profitability")


# --- score_airports ----------------------------------------------------------
def peers():
    return pd.DataFrame({
        "iata": ["AAA", "BBB", "CCC"],
        "passengers": [1e6, 5e6, 2e7],
        "load_factor": [0.70, 0.80, 0.85],
        "yoy_growth": [-0.02, 0.05, 0.03],
        "delay_rate": [0.15, 0.20, 0.25],
        "avg_taxi_out": [12, 15, 20],
        "departures_per_runway": [10_000, 20_000, 40_000],
        "cancel_rate": [0.01, 0.02, 0.03],
    })


def test_breakdown_contributions_sum_to_score():
    result = score_airports(peers(), "investment")
    totals = result["breakdown"].groupby("iata")["contribution"].sum()
    for _, row in result["scores"].iterrows():
        assert totals[row["iata"]] == pytest.approx(row["score"])


def test_dominant_airport_ranks_first():
    result = score_airports(peers(), "congestion")
    assert result["scores"].iloc[0]["iata"] == "CCC"   # highest on every congestion metric
    assert result["scores"].iloc[0]["score"] == pytest.approx(100)
    assert result["scores"].iloc[-1]["score"] == pytest.approx(0)


def test_weight_override_changes_ranking():
    default = score_airports(peers(), "investment")["scores"]["iata"].tolist()
    growth_only = {"yoy_growth": 1, "load_factor": 0, "congestion": 0, "passengers": 0}
    overridden = score_airports(peers(), "investment", growth_only)["scores"]["iata"].tolist()
    assert overridden[0] == "BBB"   # best growth
    assert overridden != default


def test_missing_metric_redistributes_weight_and_adds_caveat():
    df = peers()
    df.loc[0, "departures_per_runway"] = None   # e.g. no runway data
    result = score_airports(df, "congestion")
    weights_aaa = result["breakdown"].query("iata == 'AAA'").set_index("metric")["weight"]
    assert weights_aaa["departures_per_runway"] == 0
    assert weights_aaa.sum() == pytest.approx(1)
    assert any("AAA" in c for c in result["caveats"])


def test_partial_congestion_adds_caveat():
    df = peers()
    df.loc[1, ["delay_rate", "avg_taxi_out"]] = None   # e.g. carrier not in on-time reporting
    result = score_airports(df, "investment")
    assert any(c.startswith("congestion for BBB is based only on departures_per_runway") for c in result["caveats"])


def test_single_airport_peer_group():
    result = score_airports(peers().head(1), "investment")
    assert result["scores"]["score"].tolist() == [50]
