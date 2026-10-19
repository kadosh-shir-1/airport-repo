"""Download / read raw data, clean it, and write aggregated tables to DuckDB.

Usage:
    python ingest.py               # full run
    python ingest.py --skip-ontime # skip the (large) on-time download/aggregation
"""
import argparse
import zipfile
from datetime import date, datetime

import duckdb
import pandas as pd
import requests

import config


# --- helpers -----------------------------------------------------------------
def download(url, dest):
    """Download url to dest. Returns True on success, False if the file does not exist."""
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            return False
        chunks = r.iter_content(chunk_size=1 << 20)
        first = next(chunks, b"")
        # Servers/filters sometimes answer with an HTML page instead of the file.
        if dest.suffix == ".zip" and not first.startswith(b"PK"):
            return False
        if dest.suffix == ".csv" and first.lstrip().lower().startswith(b"<"):
            return False
        with open(dest, "wb") as f:
            f.write(first)
            for chunk in chunks:
                f.write(chunk)
    return True


def recent_months(n_back):
    """(year, month) tuples from the current month backwards."""
    y, m = date.today().year, date.today().month
    for _ in range(n_back):
        yield y, m
        m -= 1
        if m == 0:
            y, m = y - 1, 12


def _try_download(url, dest):
    try:
        return download(url, dest)
    except requests.RequestException:
        return False


# --- OurAirports -------------------------------------------------------------
def load_airports():
    """Returns the airports table, or None if the OurAirports files are unavailable."""
    config.OURAIRPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.OURAIRPORTS_FILES:
        path = config.OURAIRPORTS_DIR / name
        if path.exists():
            continue
        print(f"Downloading {name} ...")
        urls = [m.format(name=name) for m in config.OURAIRPORTS_MIRRORS]
        if not any(_try_download(url, path) for url in urls):
            print(f"WARNING: could not download {name} from any mirror. "
                  f"Place it manually in {config.OURAIRPORTS_DIR} and re-run (see README).")
            return None

    ap = pd.read_csv(config.OURAIRPORTS_DIR / "airports.csv", keep_default_na=False, na_values=[""])
    ap = ap[
        (ap["iso_country"] == "US")
        & ap["iata_code"].str.fullmatch(r"[A-Z]{3}", na=False)
        & ap["type"].isin(["large_airport", "medium_airport", "small_airport"])
    ]

    rw = pd.read_csv(config.OURAIRPORTS_DIR / "runways.csv", keep_default_na=False, na_values=[""])
    usable = rw[
        (rw["closed"] == 0)
        & (rw["length_ft"] >= config.MIN_USABLE_RUNWAY_FT)
        & ~rw["surface"].fillna("").str.upper().str.contains("WAT")
    ]
    runway_counts = usable.groupby("airport_ident").size().rename("runway_count")

    ap = ap.merge(runway_counts, left_on="ident", right_index=True, how="left")
    return pd.DataFrame({
        "iata": ap["iata_code"],
        "name": ap["name"],
        "city": ap["municipality"],
        "state": ap["iso_region"].str.replace("US-", "", regex=False),
        "airport_type": ap["type"],
        "lat": ap["latitude_deg"],
        "lon": ap["longitude_deg"],
        "runway_count": ap["runway_count"].fillna(0).astype(int),
    }).drop_duplicates("iata")


# --- T-100 Segment -----------------------------------------------------------
# TranStats headers vary slightly between download types; map normalized names -> ours.
T100_COLUMNS = {
    "DEPARTURESPERFORMED": "departures", "DEPPERFORMED": "departures",
    "SEATS": "seats", "PASSENGERS": "passengers", "FREIGHT": "freight", "DISTANCE": "distance",
    "ORIGIN": "origin", "ORIGINCOUNTRY": "origin_country",
    "DEST": "dest", "DESTCOUNTRY": "dest_country",
    "AIRCRAFTCONFIG": "aircraft_config", "YEAR": "year", "MONTH": "month",
}


def load_t100():
    files = sorted(list(config.T100_DIR.glob("*.zip")) + list(config.T100_DIR.glob("*.csv")))
    if not files:
        print(f"WARNING: no T-100 files in {config.T100_DIR} - see README for manual download steps.")
        return None

    frames = []
    for path in files:
        print(f"Reading T-100 file {path.name} ...")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.upper().replace("_", "").strip() for c in df.columns]
        df = df[[c for c in df.columns if c in T100_COLUMNS]].rename(columns=T100_COLUMNS)
        missing = set(T100_COLUMNS.values()) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        df["source_file"] = path.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # If the same month appears in several files, keep it from one file only (no double counting).
    first_source = raw.groupby(["year", "month"])["source_file"].min().rename("keep_file")
    raw = raw.join(first_source, on=["year", "month"])
    raw = raw[raw["source_file"] == raw["keep_file"]]

    raw = raw[raw["origin_country"] == "US"]
    for col in ["departures", "seats", "passengers", "freight", "distance"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    # AircraftConfig: 1 passenger, 2 freight-only, 3 combined passenger/freight, 4 seaplane.
    # Some cargo flights are reported with config 1/3 and zero (or only crew) seats and no passengers,
    # so a passenger flight must also have seats and passengers.
    raw["is_passenger_service"] = (raw["aircraft_config"] != 2) & (raw["seats"] > 0) & (raw["passengers"] > 0)

    return (
        raw.groupby(["origin", "dest", "dest_country", "year", "month", "distance", "is_passenger_service"],
                    as_index=False)[["departures", "seats", "passengers", "freight"]]
        .sum()
    )


# --- On-Time Performance -----------------------------------------------------
def download_ontime():
    """Download the latest ONTIME_MONTHS published monthly zips; returns list of (year, month, zip_path)."""
    config.ONTIME_DIR.mkdir(parents=True, exist_ok=True)
    found = []
    for year, month in recent_months(config.ONTIME_LOOKBACK_MONTHS):
        if len(found) == config.ONTIME_MONTHS:
            break
        path = config.ONTIME_DIR / f"ontime_{year}_{month:02d}.zip"
        # Also accept files downloaded manually under their original BTS name.
        manual = config.ONTIME_DIR / f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        existing = next((p for p in (path, manual) if p.exists()), None)
        if existing:
            found.append((year, month, existing))
            continue
        for pattern in config.ONTIME_URL_PATTERNS:
            url = pattern.format(year=year, month=month)
            if download(url, path):
                print(f"Downloaded on-time {year}-{month:02d}")
                found.append((year, month, path))
                break
        else:
            print(f"On-time {year}-{month:02d} not published yet, skipping")
    return found


def aggregate_ontime(con, months):
    """Extract each monthly CSV (kept on disk for inspection) and aggregate it per origin airport."""
    csv_dir = config.ONTIME_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    frames = []
    for year, month, zip_path in months:
        csv_path = csv_dir / f"ontime_{year}_{month:02d}.csv"
        if not csv_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                inner = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(inner) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
        print(f"Aggregating on-time {year}-{month:02d} ...")
        # DepDel15 is NULL for cancelled flights, so delay rate is computed over operated flights.
        frames.append(con.execute(f"""
            SELECT Origin AS origin, {year} AS year, {month} AS month,
                   count(*)                              AS flights,
                   sum(CAST(Cancelled AS DOUBLE))        AS cancelled,
                   count(DepDel15)                       AS operated_with_delay_info,
                   sum(DepDel15)                         AS delayed15,
                   sum(TaxiOut)                          AS taxi_out_sum,
                   count(TaxiOut)                        AS taxi_out_n
            FROM read_csv('{csv_path.as_posix()}', header=true, ignore_errors=true,
                          types={{'DepDel15': 'DOUBLE', 'Cancelled': 'DOUBLE', 'TaxiOut': 'DOUBLE'}})
            GROUP BY Origin
        """).df())
    return pd.concat(frames, ignore_index=True) if frames else None


# --- main --------------------------------------------------------------------
def month_label(year, month):
    return f"{int(year)}-{int(month):02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ontime", action="store_true", help="skip on-time download/aggregation")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    config.T100_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    meta = {"ingested_at": datetime.now().isoformat(timespec="seconds")}

    airports = load_airports()
    if airports is not None:
        con.execute("CREATE OR REPLACE TABLE airports AS SELECT * FROM airports")

    t100 = load_t100()
    if t100 is not None:
        con.execute("CREATE OR REPLACE TABLE t100_monthly AS SELECT * FROM t100")
        periods = t100[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        meta["t100_first_month"] = month_label(*periods.iloc[0])
        meta["t100_last_month"] = month_label(*periods.iloc[-1])
        meta["t100_month_count"] = str(len(periods))

    if not args.skip_ontime:
        months = download_ontime()
        ontime = aggregate_ontime(con, months)
        if ontime is not None:
            con.execute("CREATE OR REPLACE TABLE ontime_monthly AS SELECT * FROM ontime")
            periods = ontime[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
            meta["ontime_first_month"] = month_label(*periods.iloc[0])
            meta["ontime_last_month"] = month_label(*periods.iloc[-1])
            meta["ontime_month_count"] = str(len(periods))

    # Keep metadata from earlier runs (e.g. on-time period when re-running with --skip-ontime).
    existing = {}
    if "metadata" in con.execute("SHOW TABLES").df()["name"].tolist():
        existing = dict(con.execute("SELECT key, value FROM metadata").fetchall())
    existing.update(meta)
    meta_df = pd.DataFrame(list(existing.items()), columns=["key", "value"])
    con.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM meta_df")

    print_summary(con)
    con.close()


def print_summary(con):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    print("\n=== Row counts ===")
    for t in tables:
        print(f"{t:16s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>10,}")
    for t in tables:
        print(f"\n=== Sample: {t} ===")
        print(con.execute(f"SELECT * FROM {t} LIMIT 5").df().to_string(index=False))
    if "t100_monthly" in tables:
        print("\n=== Top 5 airports by passengers (whole loaded period) ===")
        print(con.execute("""
            SELECT origin, sum(passengers)::BIGINT AS passengers
            FROM t100_monthly GROUP BY origin ORDER BY passengers DESC LIMIT 5
        """).df().to_string(index=False))


if __name__ == "__main__":
    main()
