"""Export one airport's routes (latest 12 months) to a CSV you can open in Excel.

Usage:  python export_routes.py ANC
Output: data/exports/ANC_routes.csv  - one row per destination and flight type
"""
import sys

import duckdb

import config


def export_routes(code):
    code = code.upper()
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    df = con.execute("""
        WITH t AS (SELECT *, year * 12 + month AS mi FROM t100_monthly)
        SELECT dest, dest_country,
               max(distance) AS distance_miles,
               CASE WHEN is_passenger_service THEN 'passenger' ELSE 'cargo' END AS flight_type,
               sum(departures) AS departures,
               sum(passengers) AS passengers,
               sum(seats) AS seats
        FROM t
        WHERE origin = ? AND mi > (SELECT max(mi) FROM t) - 12
        GROUP BY dest, dest_country, flight_type
        ORDER BY departures DESC
    """, [code]).df()
    if df.empty:
        print(f"No routes found for {code}.")
        return
    out_dir = config.DATA_DIR / "exports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{code}_routes.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {path}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    export_routes(sys.argv[1] if len(sys.argv) > 1 else "ANC")
