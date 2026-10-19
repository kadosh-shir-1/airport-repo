"""Shared fixtures: a tiny in-memory DuckDB with the same schema ingest.py produces."""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def fixture_db():
    con = duckdb.connect(":memory:")
    airports = pd.DataFrame({
        "iata": ["BIG", "MID", "SML", "NRW"],
        "name": ["Big Intl", "Mid Regional", "Small Field", "No Runway Data"],
        "city": ["Bigtown", "Midtown", "Smallville", "Nowhere"],
        "state": ["MA", "CT", "VT", "MA"],
        "airport_type": ["large_airport", "medium_airport", "small_airport", "medium_airport"],
        "lat": [42.0, 41.0, 44.0, 42.5],
        "lon": [-71.0, -72.0, -73.0, -71.5],
        "runway_count": [4, 2, 1, 0],
    })
    rows = []
    # 24 months: Aug 2024 .. Jul 2026. Prior year has 10% fewer passengers at BIG.
    for i in range(24):
        year, month = divmod(2024 * 12 + 7 + i, 12)
        month += 1
        current = i >= 12
        big_pax = 1000 if current else 1000 / 1.1
        rows += [
            ("BIG", "MID", "US", year, month, 500.0, True, 10, 1200, big_pax, 0),
            ("BIG", "LHR", "GB", year, month, 3265.0, True, 10, 1200, big_pax, 0),   # long haul
            ("BIG", "PVG", "CN", year, month, 7000.0, False, 20, 0, 0, 50000),      # long-haul cargo
            ("MID", "BIG", "US", year, month, 500.0, True, 5, 500, 400, 0),
            ("NRW", "BIG", "US", year, month, 100.0, True, 2, 100, 90, 0),
        ]
    t100 = pd.DataFrame(rows, columns=["origin", "dest", "dest_country", "year", "month", "distance",
                                       "is_passenger_service", "departures", "seats", "passengers", "freight"])
    ontime = pd.DataFrame([
        ("BIG", 2026, 7, 100, 2.0, 98, 25.0, 2000.0, 98),
        ("MID", 2026, 7, 50, 1.0, 49, 5.0, 600.0, 49),
    ], columns=["origin", "year", "month", "flights", "cancelled", "operated_with_delay_info",
                "delayed15", "taxi_out_sum", "taxi_out_n"])
    metadata = pd.DataFrame({"key": ["t100_first_month", "t100_last_month"], "value": ["2024-08", "2026-07"]})
    for name, df in {"airports": airports, "t100_monthly": t100, "ontime_monthly": ontime, "metadata": metadata}.items():
        con.register("tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    yield con
    con.close()
