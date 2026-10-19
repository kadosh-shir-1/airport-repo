"""Look inside airports.duckdb and check the numbers by hand.

Usage (from the project folder, with the virtual environment active):
    python explore_db.py            # tables + row counts, export CSVs for Excel
    python explore_db.py BOS        # also: step-by-step manual check for one airport
"""
import sys

import duckdb

import config
import metrics
import tools

EXPORT_DIR = config.DATA_DIR / "exports"


def show_tables(con):
    print("\n=== Tables in the database ===")
    for (name,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
        rows = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        print(f"  {name:<16} {rows:>10,} rows")
    print("\n=== metadata ===")
    print(con.execute("SELECT * FROM metadata").df().to_string(index=False))


def export_csv(con):
    """Write CSVs you can open in Excel (utf-8-sig so Excel reads them correctly)."""
    EXPORT_DIR.mkdir(exist_ok=True)
    files = {
        "airport_metrics.csv": metrics.airport_metrics(con),           # one row per airport, all metrics
        "airports.csv": con.execute("SELECT * FROM airports").df(),
        "ontime_monthly.csv": con.execute("SELECT * FROM ontime_monthly ORDER BY origin, year, month").df(),
    }
    for name, df in files.items():
        df.to_csv(EXPORT_DIR / name, index=False, encoding="utf-8-sig")
    print(f"\nExported {', '.join(files)} to {EXPORT_DIR}")
    print("t100_monthly is too big for Excel; use the manual check below for one airport instead.")


def manual_check(con, code):
    """Recompute one airport's main metrics from the raw monthly rows and compare with the tool."""
    code = code.upper()
    last = con.execute("SELECT max(year * 12 + month) FROM t100_monthly").fetchone()[0]
    window = f"year * 12 + month > {last - 12}"
    prior = f"year * 12 + month > {last - 24} AND year * 12 + month <= {last - 12}"

    print(f"\n=== Manual check for {code} (window: {metrics.data_period(con)['current_window']}) ===")
    monthly = con.execute(f"""
        SELECT year, month, sum(passengers) AS passengers,
               sum(seats) FILTER (WHERE is_passenger_service) AS seats,
               sum(departures) AS departures
        FROM t100_monthly WHERE origin = ? AND {window}
        GROUP BY year, month ORDER BY year, month""", [code]).df()
    print("\nStep 1 - monthly totals from t100_monthly:")
    print(monthly.to_string(index=False))

    pax, seats, deps = monthly["passengers"].sum(), monthly["seats"].sum(), monthly["departures"].sum()
    pax_prev = con.execute(f"SELECT sum(passengers) FROM t100_monthly WHERE origin = ? AND {prior}",
                           [code]).fetchone()[0]
    runways = con.execute("SELECT runway_count FROM airports WHERE iata = ?", [code]).fetchone()
    runways = runways[0] if runways else None
    ot = con.execute(f"""
        SELECT sum(delayed15) AS delayed, sum(operated_with_delay_info) AS operated
        FROM ontime_monthly WHERE origin = ?
          AND year * 12 + month > (SELECT max(year * 12 + month) FROM ontime_monthly) - 12""", [code]).fetchone()

    print("\nStep 2 - the calculations:")
    print(f"  load factor      = passengers / seats         = {pax:,.0f} / {seats:,.0f} = {pax / seats:.4f}")
    if pax_prev:
        print(f"  YoY growth       = this year / last year - 1 = {pax:,.0f} / {pax_prev:,.0f} - 1 "
              f"= {pax / pax_prev - 1:.4f}")
    if runways:
        print(f"  deps per runway  = departures / runways       = {deps:,.0f} / {runways} = {deps / runways:,.1f}")
    if ot and ot[1]:
        print(f"  delay rate       = delayed / operated         = {ot[0]:,.0f} / {ot[1]:,.0f} = {ot[0] / ot[1]:.4f}")

    tool = tools.airport_profile(code)["results"]
    print("\nStep 3 - what the tool returned (should match Step 2):")
    for key in ["load_factor", "yoy_growth", "departures_per_runway", "delay_rate"]:
        print(f"  {key:<22} {tool.get(key)}")


if __name__ == "__main__":
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    tools.set_connection(con)
    show_tables(con)
    export_csv(con)
    if len(sys.argv) > 1:
        manual_check(con, sys.argv[1])
