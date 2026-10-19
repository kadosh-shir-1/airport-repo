"""Central configuration: paths, thresholds, scoring weights, regions and aliases."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
T100_DIR = RAW_DIR / "t100"
ONTIME_DIR = RAW_DIR / "ontime"
OURAIRPORTS_DIR = RAW_DIR / "ourairports"
DB_PATH = DATA_DIR / "airports.duckdb"
LOG_DIR = ROOT / "logs"
FEEDBACK_PATH = ROOT / "feedback.jsonl"

# --- Data sources ----------------------------------------------------------
# Mirrors tried in order; if all fail, place the files manually in data/raw/ourairports/.
OURAIRPORTS_MIRRORS = [
    "https://ourairports.com/data/{name}",
    "https://davidmegginson.github.io/ourairports-data/{name}",
    "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/{name}",
]
OURAIRPORTS_FILES = ["airports.csv", "runways.csv"]
# BTS publishes two naming variants for the same monthly file; try both.
ONTIME_URL_PATTERNS = [
    "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip",
    "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip",
]
ONTIME_MONTHS = 12          # how many recent months of on-time data to load
ONTIME_LOOKBACK_MONTHS = 24  # how far back from today to probe for published files
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

# --- Scope & thresholds ----------------------------------------------------
# Default airport universe. T-100 counts departing (enplaned) passengers only;
# 500k enplanements ~= 1M total passengers (arriving + departing).
MIN_ANNUAL_ENPLANEMENTS = 500_000
LONG_HAUL_MILES = 3000
MIN_USABLE_RUNWAY_FT = 3000        # runways shorter than this are ignored in the capacity proxy
LOW_CONFIDENCE_MIN_DEPARTURES = 1000
LOW_CONFIDENCE_MIN_PEERS = 3

# --- Gemini ----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional extra keys (comma-separated). Used in order when the previous key's quota runs out.
GEMINI_API_KEYS = list(dict.fromkeys(
    k.strip().strip("\"'") for k in [GEMINI_API_KEY or "", *os.getenv("GEMINI_EXTRA_API_KEYS", "").split(",")]
    if k.strip().strip("\"'")))
# gemini-2.5-flash (suggested in the spec) is no longer available to new API users.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Free tier allows ~20 requests/day per model, and each model has its own quota. When the main model's
# daily quota is used up, the agent moves on to the next model in this list.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash").split(",") if m.strip()]
# The topic check is a simple yes/no question: a light model is enough and saves the main model's quota.
GEMINI_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", "gemini-3.1-flash-lite")
MAX_TOOL_ITERATIONS = 6
RATE_LIMIT_RETRIES = 2          # short retries on HTTP 429 when the limit is per-minute
RATE_LIMIT_WAIT_SECONDS = 20    # wait 20s, 40s between those retries
RATE_LIMIT_MAX_WAIT_SECONDS = 60  # if the server asks to wait longer (daily quota), switch model instead
SERVER_BUSY_WAIT_SECONDS = 3    # HTTP 500/503 (Google overloaded): wait 3s, 6s, then try the next model

# --- Scoring ---------------------------------------------------------------
# "congestion" is a composite: mean of the normalized CONGESTION_COMPONENTS.
CONGESTION_COMPONENTS = ["delay_rate", "avg_taxi_out", "departures_per_runway"]

SCORE_WEIGHTS = {
    "investment": {"yoy_growth": 0.30, "load_factor": 0.25, "congestion": 0.25, "passengers": 0.20},
    "unmet_demand": {"load_factor": 1 / 3, "congestion": 1 / 3, "yoy_growth": 1 / 3},
    "congestion": {"delay_rate": 0.25, "avg_taxi_out": 0.25, "departures_per_runway": 0.25, "cancel_rate": 0.25},
}
LOG_SCALE_METRICS = {"passengers"}

# --- Regions ---------------------------------------------------------------
REGIONS = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "mid-atlantic": ["NY", "NJ", "PA", "DE", "MD", "DC"],
    "southeast": ["VA", "NC", "SC", "GA", "FL", "AL", "MS", "TN", "KY"],
    "midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "southwest": ["AZ", "NM", "TX", "OK", "NV"],
    "mountain": ["CO", "UT", "WY", "MT", "ID"],
    "west coast": ["CA", "OR", "WA"],
    "pacific northwest": ["OR", "WA"],
    "california": ["CA"],
    "texas": ["TX"],
    "florida": ["FL"],
    "alaska": ["AK"],
    "hawaii": ["HI"],
}

# City areas with several airports. They can cross state lines (Newark is in NJ), so they are airport lists.
METRO_AREAS = {
    "new york city area": ["JFK", "LGA", "EWR", "HPN", "ISP"],
    "chicago area": ["ORD", "MDW"],
    "washington dc area": ["DCA", "IAD", "BWI"],
    "los angeles area": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "san francisco bay area": ["SFO", "OAK", "SJC"],
    "houston area": ["IAH", "HOU"],
    "dallas area": ["DFW", "DAL"],
    "south florida": ["MIA", "FLL", "PBI"],
}
METRO_ALIASES = {"nyc area": "new york city area", "nyc": "new york city area", "la area": "los angeles area",
                 "bay area": "san francisco bay area", "dc area": "washington dc area"}

# Plain-language names shown to analysts.
SCORE_LABELS = {
    "investment": "Investment opportunity score",
    "unmet_demand": "Unmet demand index",
    "congestion": "Congestion score",
}
METRIC_LABELS = {
    "yoy_growth": "passenger growth", "load_factor": "how full the planes are",
    "congestion": "congestion (delays, taxi time, runway use)", "passengers": "airport size",
    "delay_rate": "share of delayed flights", "avg_taxi_out": "taxi time to takeoff",
    "departures_per_runway": "departures per runway", "cancel_rate": "cancellations",
}

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}

# Unambiguous names/aliases -> IATA
AIRPORT_ALIASES = {
    "la": "LAX", "l.a.": "LAX", "los angeles": "LAX",
    "santa ana": "SNA", "john wayne": "SNA", "orange county": "SNA",
    "anchorage": "ANC", "ted stevens": "ANC",
    "sf": "SFO", "san francisco": "SFO",
    "boston": "BOS", "logan": "BOS",
    "seattle": "SEA", "sea-tac": "SEA",
    "atlanta": "ATL", "denver": "DEN", "dallas fort worth": "DFW", "dfw": "DFW",
    "las vegas": "LAS", "phoenix": "PHX", "miami": "MIA", "orlando": "MCO",
    "hartford": "BDL", "bradley": "BDL", "providence": "PVD", "manchester nh": "MHT",
    "portland maine": "PWM", "burlington": "BTV", "honolulu": "HNL", "san diego": "SAN",
    "burbank": "BUR", "long beach": "LGB", "ontario": "ONT", "oakland": "OAK", "san jose": "SJC",
    "jfk": "JFK", "kennedy": "JFK", "laguardia": "LGA", "newark": "EWR",
    "o'hare": "ORD", "ohare": "ORD", "midway": "MDW",
}

# Names that map to several airports -> analyst must choose
AMBIGUOUS_NAMES = {
    "new york": ["JFK", "LGA", "EWR"],
    "nyc": ["JFK", "LGA", "EWR"],
    "la area": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "los angeles area": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "southern california": ["LAX", "BUR", "LGB", "ONT", "SNA", "SAN", "PSP"],
    "washington": ["DCA", "IAD", "BWI"],
    "dc": ["DCA", "IAD", "BWI"],
    "chicago": ["ORD", "MDW"],
    "houston": ["IAH", "HOU"],
    "dallas": ["DFW", "DAL"],
    "bay area": ["SFO", "OAK", "SJC"],
    "portland": ["PDX", "PWM"],
}
