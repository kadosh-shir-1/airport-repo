"""Per-airport metrics computed with SQL over the DuckDB tables built by ingest.py.

Every function takes an optional DuckDB connection so tests can pass a small fixture DB.
"""
import duckdb
import pandas as pd

import config


def connect():
    return duckdb.connect(str(config.DB_PATH), read_only=True)


def _month_label(month_index):
    year, month = divmod(int(month_index) - 1, 12)
    return f"{year}-{month + 1:02d}"


def data_period(con=None):
    """Metadata plus the exact current / prior 12-month windows used for metrics."""
    con = con or connect()
    meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
    last_mi = con.execute("SELECT max(year * 12 + month) FROM t100_monthly").fetchone()[0]
    meta["current_window"] = f"{_month_label(last_mi - 11)} to {_month_label(last_mi)}"
    meta["prior_window"] = f"{_month_label(last_mi - 23)} to {_month_label(last_mi - 12)}"
    return meta


METRICS_SQL = """
WITH t AS (SELECT *, year * 12 + month AS mi FROM t100_monthly),
last AS (SELECT max(mi) AS last_mi FROM t),
cur AS (
    SELECT origin,
           sum(passengers) AS passengers,
           sum(seats) FILTER (WHERE is_passenger_service) AS seats,
           sum(departures) AS departures,
           sum(departures) FILTER (WHERE is_passenger_service) AS passenger_departures,
           coalesce(sum(departures) FILTER (WHERE is_passenger_service AND distance > $long_haul), 0)
               AS long_haul_departures
    FROM t, last WHERE mi > last_mi - 12
    GROUP BY origin),
prev AS (
    SELECT origin, sum(passengers) AS passengers_prev
    FROM t, last WHERE mi > last_mi - 24 AND mi <= last_mi - 12
    GROUP BY origin),
o AS (SELECT *, year * 12 + month AS mi FROM ontime_monthly),
olast AS (SELECT max(mi) AS last_mi FROM o),
ot AS (
    SELECT origin,
           sum(flights) AS ontime_flights,
           sum(delayed15) / nullif(sum(operated_with_delay_info), 0) AS delay_rate,
           sum(cancelled) / nullif(sum(flights), 0) AS cancel_rate,
           sum(taxi_out_sum) / nullif(sum(taxi_out_n), 0) AS avg_taxi_out
    FROM o, olast WHERE mi > last_mi - 12
    GROUP BY origin)
SELECT a.iata, a.name, a.city, a.state, a.runway_count,
       c.passengers, p.passengers_prev, c.seats, c.departures, ot.ontime_flights,
       c.passengers / nullif(c.seats, 0)                    AS load_factor,
       c.passengers / nullif(p.passengers_prev, 0) - 1      AS yoy_growth,
       ot.delay_rate, ot.cancel_rate, ot.avg_taxi_out,
       c.departures / nullif(a.runway_count, 0)             AS departures_per_runway,
       c.long_haul_departures / nullif(c.passenger_departures, 0) AS long_haul_share
FROM cur c
JOIN airports a ON a.iata = c.origin
LEFT JOIN prev p USING (origin)
LEFT JOIN ot USING (origin)
ORDER BY c.passengers DESC
"""


def airport_metrics(con=None, long_haul_miles=config.LONG_HAUL_MILES):
    """One row per airport with all metrics for the latest 12 T-100 months (no universe filter)."""
    con = con or connect()
    return con.execute(METRICS_SQL, {"long_haul": long_haul_miles}).df()


def long_haul_detail(code, threshold_miles=config.LONG_HAUL_MILES, include_cargo=False, top_n=10, con=None):
    """Long-haul share of departures for one airport plus its top long-haul destinations."""
    con = con or connect()
    service_filter = "" if include_cargo else "AND is_passenger_service"
    params = {"code": code, "threshold": threshold_miles}
    base = f"""
        WITH t AS (SELECT *, year * 12 + month AS mi FROM t100_monthly),
        last AS (SELECT max(mi) AS last_mi FROM t)
        SELECT * FROM t, last
        WHERE origin = $code AND mi > last_mi - 12 {service_filter}
    """
    totals = con.execute(f"""
        SELECT coalesce(sum(departures), 0) AS departures,
               coalesce(sum(departures) FILTER (WHERE distance > $threshold), 0) AS long_haul_departures
        FROM ({base})
    """, params).fetchone()
    destinations = con.execute(f"""
        SELECT dest, dest_country, max(distance) AS distance_miles,
               sum(departures) AS departures, sum(passengers) AS passengers, sum(freight) AS freight_lbs
        FROM ({base})
        WHERE distance > $threshold
        GROUP BY dest, dest_country
        ORDER BY departures DESC
        LIMIT {int(top_n)}
    """, params).df()
    departures, long_haul = totals
    return {
        "departures": departures,
        "long_haul_departures": long_haul,
        "long_haul_share": long_haul / departures if departures else None,
        "top_destinations": destinations,
    }
