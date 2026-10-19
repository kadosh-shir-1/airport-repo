import math

import pytest

from metrics import airport_metrics, data_period, long_haul_detail


def test_core_metrics(fixture_db):
    m = airport_metrics(fixture_db).set_index("iata")
    big = m.loc["BIG"]
    assert big["passengers"] == pytest.approx(12 * 2000)
    assert big["load_factor"] == pytest.approx(2000 / 2400)
    assert big["yoy_growth"] == pytest.approx(0.10)
    assert big["departures_per_runway"] == pytest.approx(12 * 40 / 4)
    assert big["delay_rate"] == pytest.approx(25 / 98)
    assert big["avg_taxi_out"] == pytest.approx(2000 / 98)
    assert big["cancel_rate"] == pytest.approx(2 / 100)
    # passenger long haul: LHR (10) out of 20 passenger departures
    assert big["long_haul_share"] == pytest.approx(0.5)


def test_missing_data_is_nan_not_error(fixture_db):
    m = airport_metrics(fixture_db).set_index("iata")
    nrw = m.loc["NRW"]
    assert math.isnan(nrw["departures_per_runway"])   # runway_count = 0
    assert math.isnan(nrw["delay_rate"])              # no on-time data
    assert nrw["long_haul_share"] == 0                 # no long-haul flights -> 0, not NaN


def test_long_haul_with_and_without_cargo(fixture_db):
    pax = long_haul_detail("BIG", con=fixture_db)
    cargo = long_haul_detail("BIG", include_cargo=True, con=fixture_db)
    assert pax["long_haul_share"] == pytest.approx(0.5)
    assert cargo["long_haul_share"] == pytest.approx(30 / 40)
    assert cargo["top_destinations"].iloc[0]["dest"] == "PVG"


def test_unknown_airport_long_haul(fixture_db):
    assert long_haul_detail("XXX", con=fixture_db)["long_haul_share"] is None


def test_data_period_windows(fixture_db):
    period = data_period(fixture_db)
    assert period["current_window"] == "2025-08 to 2026-07"
    assert period["prior_window"] == "2024-08 to 2025-07"
