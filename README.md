# US Airport Terminal Expansion Agent

A chat assistant for investment analysts. It finds US airports where **demand exceeds capacity** and
explains why, so they can be shortlisted for terminal expansion. All numbers are computed
deterministically from public BTS / OurAirports data, and Gemini only picks the right tool and
explains the result. See [DESIGN.md](DESIGN.md) for the methodology.

## Quick start

The repo already includes the processed database (`data/airports.duckdb`), so you can skip the data download.

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env
```

Put a Gemini API key in `.env` (free at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key-here
# Optional: more keys, used in turn when one key's daily free quota runs out
GEMINI_EXTRA_API_KEYS=key2,key3
```

Run the app:

```bash
streamlit run app.py
```

Run the tests (offline, they use a small fixture database):

```bash
pytest
```

## Example questions

1. Which airports in New England are strong candidates for terminal expansion?
2. Compare LA and Santa Ana airport congestion levels.
3. What is the percentage of long haul flights out of Anchorage airport?
4. What is the unmet flight demand in SFO airport and why?

The sidebar has these as buttons. It also has sliders for the score weights and a box for excluding airports.

## Data

The data was downloaded from public sources and is already processed into `data/airports.duckdb`
(included in the repo), so no download is needed to run the app.

| Source | What | Period |
|---|---|---|
| BTS T-100 Segment (All Carriers) | passengers, seats, departures, distance, freight | Jan 2024 – Jul 2026 |
| BTS On-Time Performance | delays, cancellations, taxi-out | Aug 2025 – Jul 2026 |
| OurAirports | airport locations, runways | current |

To rebuild from scratch: put the T-100 zip files (downloaded manually from TranStats) in `data/raw/t100/`
and run `python ingest.py`. On-time and OurAirports download automatically. The T-100 fields used are
listed in `ingest.py`.

To check the numbers by hand, use `python explore_db.py BOS`. It exports CSVs for Excel and recomputes one
airport step by step. To export one airport's routes, use `python export_routes.py ANC`.
