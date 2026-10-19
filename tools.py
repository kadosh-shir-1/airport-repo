"""Tools exposed to the LLM. Each returns a JSON-safe dict:

    {results, assumptions, data_period, caveats, needs_review, review_reasons}

All numbers come from metrics.py / scoring.py; tools only select data and package it.
Callable and testable without the LLM.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process, utils

import config
import metrics
import scoring

# --- connection & caches -------------------------------------------------------
_con = None
_cache = {}


def set_connection(con):
    """Use a specific DuckDB connection (tests pass a fixture DB)."""
    global _con
    _con = con
    _cache.clear()


def _get_con():
    global _con
    if _con is None:
        _con = metrics.connect()
    return _con


def _metrics(long_haul_miles=config.LONG_HAUL_MILES):
    key = ("metrics", long_haul_miles)
    if key not in _cache:
        _cache[key] = metrics.airport_metrics(_get_con(), long_haul_miles)
    return _cache[key]


def _data_period():
    if "period" not in _cache:
        _cache["period"] = metrics.data_period(_get_con())
    return _cache["period"]


def _airports():
    if "airports" not in _cache:
        _cache["airports"] = _get_con().execute("SELECT * FROM airports").df()
    return _cache["airports"]


# --- response helpers ----------------------------------------------------------
def _jsonable(obj):
    """Convert DataFrames / numpy types / NaN to plain JSON values (floats rounded)."""
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _response(results, assumptions=None, caveats=None, review_reasons=None, status="ok"):
    return _jsonable({
        "status": status,
        "results": results,
        "assumptions": assumptions or {},
        "data_period": _data_period(),
        "caveats": caveats or [],
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons or [],
    })


def _quality_issues(row):
    """Low-confidence reasons for one airport's metrics row."""
    issues = []
    code = row["iata"]
    if not row["runway_count"]:
        issues.append(f"{code}: no runway data, departures_per_runway unavailable")
    if pd.isna(row["delay_rate"]):
        issues.append(f"{code}: not in BTS on-time reporting (no delay / taxi-out / cancellation data)")
    if row["departures"] < config.LOW_CONFIDENCE_MIN_DEPARTURES:
        issues.append(f"{code}: only {int(row['departures'])} departures in the window (small sample)")
    if pd.isna(row["yoy_growth"]):
        issues.append(f"{code}: no prior-year data, growth unavailable")
    return issues


AIRPORT_COLUMNS = ["iata", "name", "city", "state", "passengers"]


# --- tools -----------------------------------------------------------------------
def resolve_airport(text):
    """Map a name / city / alias / code to an IATA code.

    status: "ok" (one airport), "ambiguous" (analyst must choose), "not_found".
    """
    query = text.strip().lower()
    generic = {"airport", "international", "intl", "regional", "the"}
    stripped = " ".join(w for w in query.split() if w not in generic)
    query = stripped or query
    m = _metrics().set_index("iata")
    known = set(_airports()["iata"])

    def candidates(codes):
        rows = []
        for code in codes:
            if code in m.index:
                r = m.loc[code]
                rows.append({"iata": code, "name": r["name"], "city": r["city"], "state": r["state"],
                             "passengers": r["passengers"]})
        return rows

    if query in config.AMBIGUOUS_NAMES:
        return _response(candidates(config.AMBIGUOUS_NAMES[query]), status="ambiguous",
                         caveats=[f"'{text}' matches several airports; ask the analyst which one."])

    caveats = []
    if query.upper() in config.US_STATES and query in config.AIRPORT_ALIASES:
        caveats.append(f"'{text.upper()}' is also the state code for {config.US_STATES[query.upper()]}.")

    code = config.AIRPORT_ALIASES.get(query) or (query.upper() if query.upper() in known else None)
    if code and code in known:
        return _response(candidates([code]) or [{"iata": code}], caveats=caveats)

    # Exact city match among airports with commercial traffic; prefer those above the size threshold.
    in_city = m[m["city"].str.lower() == query]
    if len(in_city):
        big = in_city[in_city["passengers"] >= config.MIN_ANNUAL_ENPLANEMENTS]
        codes = list((big if len(big) else in_city.head(1)).index)
        if len(codes) == 1:
            return _response(candidates(codes), caveats=caveats)
        return _response(candidates(codes), status="ambiguous",
                         caveats=[f"'{text}' has several airports; ask the analyst which one."])

    # Fuzzy match on names and cities of airports that have commercial traffic.
    choices = {}
    for code, r in m.iterrows():
        choices[r["name"]] = code
        choices.setdefault(str(r["city"]), code)
    matches = process.extract(query, list(choices), scorer=fuzz.WRatio, limit=5, score_cutoff=85,
                              processor=utils.default_process)
    codes = list(dict.fromkeys(choices[name] for name, _, _ in matches))
    if not codes:
        return _response([], status="not_found", caveats=[f"No US airport matches '{text}'."])
    top_score = matches[0][1]
    strong = [choices[name] for name, score, _ in matches if score >= top_score - 3]
    strong = list(dict.fromkeys(strong))
    if len(strong) == 1:
        return _response(candidates(strong), caveats=caveats)
    return _response(candidates(strong), status="ambiguous",
                     caveats=[f"'{text}' matches several airports; ask the analyst which one."])


def _resolve_region(region_or_states):
    """Return (states, label, ambiguity_message_or_None)."""
    if isinstance(region_or_states, (list, tuple)):
        states = [s.strip().upper() for s in region_or_states]
        return states, ", ".join(states), None
    text = region_or_states.strip()
    key = text.lower()
    if key in config.REGIONS:
        return config.REGIONS[key], text, None
    if text.upper() in config.US_STATES:
        if key in config.AIRPORT_ALIASES or key in config.AMBIGUOUS_NAMES:
            return None, text, (f"'{text}' could mean the state {config.US_STATES[text.upper()]} "
                                f"or the city/airport alias '{text}'. Ask the analyst which one.")
        return [text.upper()], config.US_STATES[text.upper()], None
    by_name = {v.lower(): k for k, v in config.US_STATES.items()}
    if key in by_name:
        if key in config.AMBIGUOUS_NAMES:   # "New York", "Washington": the state or the city's airports?
            codes = ", ".join(config.AMBIGUOUS_NAMES[key])
            return None, text, (f"'{text}' could mean the state {config.US_STATES[by_name[key]]} "
                                f"or the {text} city area airports ({codes}). Ask the analyst which one.")
        return [by_name[key]], text, None
    parts = [p.strip().upper() for p in text.split(",")]
    if len(parts) > 1 and all(p in config.US_STATES for p in parts):
        return parts, text, None
    return None, text, f"Unknown region or state '{text}'. Known regions: {sorted(config.REGIONS)}."


def list_airports(region_or_states, min_enplanements=config.MIN_ANNUAL_ENPLANEMENTS):
    """Airports in a region / state list that pass the size threshold."""
    states, label, problem = _resolve_region(region_or_states)
    if problem:
        status = "ambiguous" if "could mean" in problem else "not_found"
        return _response([], status=status, caveats=[problem])
    m = _metrics()
    in_states = m[m["state"].isin(states)]
    selected = in_states[in_states["passengers"] >= min_enplanements]
    below = in_states[in_states["passengers"] < min_enplanements]
    assumptions = {"region": label, "states": states,
                   "min_annual_enplanements": min_enplanements, "window": _data_period()["current_window"]}
    caveats = []
    if len(below):
        caveats.append(f"{len(below)} smaller airports in these states are below the threshold and excluded.")
    review = []
    if len(selected) < config.LOW_CONFIDENCE_MIN_PEERS:
        review.append(f"Only {len(selected)} airports in the peer group (< {config.LOW_CONFIDENCE_MIN_PEERS}).")
    return _response(selected[AIRPORT_COLUMNS], assumptions, caveats, review)


def airport_profile(code):
    """All metrics for one airport, plus its national rank by passengers."""
    code = code.strip().upper()
    m = _metrics()
    if code not in set(m["iata"]):
        return _response({}, status="not_found", caveats=[f"No traffic data for airport code '{code}'."])
    row = m[m["iata"] == code].iloc[0]
    universe = m[m["passengers"] >= config.MIN_ANNUAL_ENPLANEMENTS]
    result = row.to_dict()
    result["national_passenger_rank"] = int((universe["passengers"] > row["passengers"]).sum() + 1)
    result["universe_size"] = len(universe)
    caveats = []
    if row["passengers"] < config.MIN_ANNUAL_ENPLANEMENTS:
        caveats.append(f"{code} is below the default size threshold "
                       f"({config.MIN_ANNUAL_ENPLANEMENTS:,} annual enplanements).")
    assumptions = {
        "window": _data_period()["current_window"],
        "growth_compared_to": _data_period()["prior_window"],
        "long_haul_threshold_miles": config.LONG_HAUL_MILES,
        "runways_counted": f"open, non-water, >= {config.MIN_USABLE_RUNWAY_FT} ft",
        "passengers": "departing (enplaned) passengers, T-100",
    }
    return _response(result, assumptions, caveats, _quality_issues(row))


# --- scoring helpers -------------------------------------------------------------
METRIC_COLUMNS = ["passengers", "yoy_growth", "load_factor", "delay_rate", "avg_taxi_out",
                  "cancel_rate", "departures_per_runway", "long_haul_share"]


def _parse_codes(codes):
    """Accept a list or a comma-separated string of IATA codes; return unique upper-case codes."""
    if not codes:
        return []
    if isinstance(codes, str):
        codes = codes.split(",")
    return list(dict.fromkeys(c.strip().upper() for c in codes if c and c.strip()))


def _universe(min_enplanements):
    m = _metrics()
    return m[m["passengers"] >= min_enplanements]


def _input_metrics(weights):
    """Raw metric columns behind a score's weights ('congestion' expands to its components)."""
    cols = []
    for name in weights:
        cols += config.CONGESTION_COMPONENTS if name == "congestion" else [name]
    return cols


def _scored_rows(peer, result, codes):
    """Per airport: identity, score, rank, raw metrics, full breakdown and the top 2 drivers."""
    scores = result["scores"].set_index("iata")
    breakdown = result["breakdown"]
    peer = peer.set_index("iata")
    rows = []
    for code in codes:
        parts = breakdown[breakdown["iata"] == code].drop(columns="iata")
        parts = parts.sort_values("contribution", ascending=False)
        row = {"iata": code, "name": peer.at[code, "name"], "city": peer.at[code, "city"],
               "state": peer.at[code, "state"], "score": scores.at[code, "score"],
               "rank": scores.at[code, "rank"]}
        row["metrics"] = {c: peer.at[code, c] for c in METRIC_COLUMNS}
        row["breakdown"] = parts.to_dict(orient="records")
        row["top_drivers"] = [p["metric"] for p in row["breakdown"][:2]
                              if pd.notna(p["contribution"]) and p["contribution"] > 0]
        rows.append(row)
    return rows


def _run_score(peer, score, weights, method):
    try:
        return scoring.score_airports(peer, score, weights, method), None
    except ValueError as exc:
        return None, str(exc)


def peer_scope(region_or_states=None, airports=None, exclude=None,
               min_enplanements=config.MIN_ANNUAL_ENPLANEMENTS):
    """Which airports a ranking compares. Shared by rank_airports and the analyst's confirmation card.

    Returns {"peer": DataFrame, "label", "states", "airports", "below_threshold", "status", "problem"}.
    An explicit airport list (or a city area such as "New York City area") is taken as given:
    the analyst chose those airports, so the size threshold is not applied.
    """
    m = _metrics()
    excluded = _parse_codes(exclude)
    scope = {"states": None, "airports": None, "below_threshold": 0, "status": "ok", "problem": None}
    key = region_or_states.strip().lower() if isinstance(region_or_states, str) else None
    key = config.METRO_ALIASES.get(key, key)
    if airports or key in config.METRO_AREAS:
        codes = _parse_codes(airports) if airports else config.METRO_AREAS[key]
        scope["airports"] = codes
        scope["label"] = region_or_states if key in config.METRO_AREAS and not airports else \
            "Selected airports: " + ", ".join(codes)
        in_scope = m[m["iata"].isin(codes)]
        big_enough = pd.Series(True, index=in_scope.index)
    else:
        if region_or_states:
            states, label, problem = _resolve_region(region_or_states)
            if problem:
                scope.update(status="ambiguous" if "could mean" in problem else "not_found",
                             problem=problem, label=label, peer=m.iloc[0:0])
                return scope
        else:
            states, label = None, "All US airports"
        scope.update(states=states, label=label)
        in_scope = m if states is None else m[m["state"].isin(states)]
        big_enough = in_scope["passengers"] >= min_enplanements
    scope["below_threshold"] = int((~big_enough).sum())
    scope["peer"] = in_scope[big_enough & ~in_scope["iata"].isin(excluded)]
    scope["excluded"] = excluded
    return scope


# --- tools (continued) -------------------------------------------------------------
def rank_airports(region_or_states=None, score="investment", weights=None, top_n=10,
                  min_enplanements=config.MIN_ANNUAL_ENPLANEMENTS, exclude=None, method="minmax",
                  airports=None):
    """Rank airports in a region, a city area, an explicit airport list, or nationally.

    weights: optional overrides, e.g. {"yoy_growth": 0.5}; re-normalized to sum to 1.
    exclude: airport codes the analyst removed from the peer group.
    airports: explicit list of IATA codes to rank (e.g. a metro area that crosses state lines).
    """
    scope = peer_scope(region_or_states, airports, exclude, min_enplanements)
    if scope["problem"]:
        return _response([], status=scope["status"], caveats=[scope["problem"]])
    peer, label, states, excluded = scope["peer"], scope["label"], scope["states"], scope["excluded"]
    if peer.empty:
        return _response([], status="not_found",
                         caveats=[f"No airports in {label} above {min_enplanements:,} annual enplanements."])

    result, error = _run_score(peer, score, weights, method)
    if error:
        return _response([], status="error", caveats=[error])

    top_codes = list(result["scores"]["iata"].head(int(top_n)))
    assumptions = {
        "score": score, "weights_used": result["weights"], "normalization": f"{method}, 0-100 within peer group",
        "region": label, "states": states or ("selected airports" if scope["airports"] else "all"),
        "airports_compared": list(peer["iata"]),
        "min_annual_enplanements": "not applied (airports chosen explicitly)" if scope["airports"] else min_enplanements,
        "window": _data_period()["current_window"], "growth_compared_to": _data_period()["prior_window"],
        "excluded_by_analyst": excluded, "manual_override": bool(weights) or bool(excluded),
    }
    caveats = list(result["caveats"])
    missing = [c for c in (scope["airports"] or []) if c not in set(peer["iata"]) and c not in excluded]
    if missing:
        caveats.append(f"No traffic data for: {', '.join(missing)}.")
    below = scope["below_threshold"]
    if below:
        caveats.append(f"{below} smaller airports in scope are below the size threshold and excluded.")
    caveats.append("Scores are relative to this peer group only; they are not comparable across different groups.")
    review = []
    if len(peer) < config.LOW_CONFIDENCE_MIN_PEERS:
        review.append(f"Only {len(peer)} airports in the peer group (< {config.LOW_CONFIDENCE_MIN_PEERS}).")
    peer_idx = peer.set_index("iata")
    for code in top_codes:
        review += _quality_issues(peer_idx.loc[code].to_dict() | {"iata": code})
    results = {"peer_group_size": len(peer), "ranking": _scored_rows(peer, result, top_codes)}
    return _response(results, assumptions, caveats, review)


def compare_airports(codes, score="congestion", weights=None,
                     min_enplanements=config.MIN_ANNUAL_ENPLANEMENTS, method="minmax"):
    """Side-by-side metrics and scores for 2+ airports.

    Scores are computed against the national peer group (plus the compared airports), so that
    comparing two airports does not force one to 0 and the other to 100.
    """
    wanted = _parse_codes(codes)
    m = _metrics()
    found = [c for c in wanted if c in set(m["iata"])]
    missing = [c for c in wanted if c not in found]
    if len(found) < 2:
        return _response([], status="not_found",
                         caveats=[f"Need at least two airports with traffic data; not found: {missing or 'none'}."])

    universe = _universe(min_enplanements)
    peer = pd.concat([universe, m[m["iata"].isin(found) & ~m["iata"].isin(universe["iata"])]])
    result, error = _run_score(peer, score, weights, method)
    if error:
        return _response([], status="error", caveats=[error])

    rows = _scored_rows(peer, result, found)
    per_metric = {}
    for col in _input_metrics(result["weights"]):
        values = {r["iata"]: r["metrics"][col] for r in rows if pd.notna(r["metrics"][col])}
        if values:
            per_metric[col] = {"values": values, "highest": max(values, key=values.get)}
    best = max(rows, key=lambda r: -1 if pd.isna(r["score"]) else r["score"])

    assumptions = {
        "score": score, "weights_used": result["weights"],
        "peer_group": f"{len(peer)} US airports above {min_enplanements:,} annual enplanements (plus compared airports)",
        "normalization": f"{method}, 0-100 within peer group", "window": _data_period()["current_window"],
        "manual_override": bool(weights),
    }
    # Keep only caveats about the compared airports (not about other airports in the national peer group).
    caveats = [c for c in result["caveats"] if any(code in c for code in found)]
    if missing:
        caveats.append(f"No traffic data for: {', '.join(missing)}.")
    review = []
    peer_idx = peer.set_index("iata")
    for code in found:
        review += _quality_issues(peer_idx.loc[code].to_dict() | {"iata": code})
    results = {"airports": rows, "by_metric": per_metric, "highest_score": best["iata"],
               "peer_group_size": len(peer)}
    return _response(results, assumptions, caveats, review)


def long_haul_share(code, threshold_miles=config.LONG_HAUL_MILES, include_cargo=False, top_n=10):
    """Share of departures from one airport flying more than threshold_miles, plus top destinations."""
    code = code.strip().upper()
    if code not in set(_metrics()["iata"]):
        return _response({}, status="not_found", caveats=[f"No traffic data for airport code '{code}'."])

    con = _get_con()
    main = metrics.long_haul_detail(code, threshold_miles, include_cargo, top_n, con=con)
    other = metrics.long_haul_detail(code, threshold_miles, not include_cargo, 0, con=con)
    if not main["departures"]:
        return _response({}, status="not_found", caveats=[f"No departures recorded for {code} in the window."])

    results = {
        "airport": code,
        "long_haul_share": main["long_haul_share"],
        "departures": main["departures"],
        "long_haul_departures": main["long_haul_departures"],
        "top_long_haul_destinations": main["top_destinations"],
        ("share_including_cargo" if not include_cargo else "share_passenger_only"): other["long_haul_share"],
    }
    assumptions = {
        "long_haul_threshold_miles": threshold_miles,
        "flights_counted": "passenger and cargo" if include_cargo else "passenger service only",
        "unit": "departures performed (non-stop segments), all US and foreign carriers, T-100",
        "window": _data_period()["current_window"],
    }
    caveats = ["There is no single official definition of long haul; the result depends on the threshold."]
    if not include_cargo and other["long_haul_share"] is not None \
            and abs(other["long_haul_share"] - main["long_haul_share"]) >= 0.05:
        caveats.append(f"Including cargo flights changes the share to {other['long_haul_share']:.1%} "
                       f"(cargo is a large part of long-haul traffic here).")
    review = []
    if main["departures"] < config.LOW_CONFIDENCE_MIN_DEPARTURES:
        review.append(f"{code}: only {int(main['departures'])} departures in the window (small sample)")
    return _response(results, assumptions, caveats, review)


def unmet_demand(code, region_or_states=None, weights=None,
                 min_enplanements=config.MIN_ANNUAL_ENPLANEMENTS, method="minmax"):
    """Unmet Demand Index for one airport (proxy: load factor + congestion + growth) and its drivers."""
    code = code.strip().upper()
    m = _metrics()
    if code not in set(m["iata"]):
        return _response({}, status="not_found", caveats=[f"No traffic data for airport code '{code}'."])

    if region_or_states:
        states, label, problem = _resolve_region(region_or_states)
        if problem:
            status = "ambiguous" if "could mean" in problem else "not_found"
            return _response({}, status=status, caveats=[problem])
        scope = m[m["state"].isin(states)]
    else:
        label, scope = "All US airports", m
    peer = scope[scope["passengers"] >= min_enplanements]
    caveats = ["Unmet demand cannot be observed directly (people who did not fly are not in the data). "
               "This is a relative proxy index, not a passenger count."]
    if code not in set(peer["iata"]):
        peer = pd.concat([peer, m[m["iata"] == code]])
        caveats.append(f"{code} is outside the default peer group and was added for comparison.")

    result, error = _run_score(peer, "unmet_demand", weights, method)
    if error:
        return _response({}, status="error", caveats=[error])
    caveats += [c for c in result["caveats"] if code in c]   # only caveats about this airport

    row = _scored_rows(peer, result, [code])[0]
    signals = {}
    for col in _input_metrics(result["weights"]):
        value, median = row["metrics"][col], peer[col].median()
        if pd.notna(value) and pd.notna(median):
            signals[col] = {"value": value, "peer_median": median,
                            "vs_peers": "above" if value > median else "below" if value < median else "equal"}
    results = {
        "airport": code, "name": row["name"],
        "unmet_demand_index": row["score"], "rank": row["rank"], "peer_group_size": len(peer),
        "drivers": row["breakdown"], "top_drivers": row["top_drivers"], "signals_vs_peers": signals,
    }
    assumptions = {
        "definition": "Unmet Demand Index = weighted mix of load factor (full planes), congestion "
                      "(delays, taxi-out, departures per runway) and YoY passenger growth",
        "weights_used": result["weights"], "peer_group": label, "min_annual_enplanements": min_enplanements,
        "normalization": f"{method}, 0-100 within peer group", "window": _data_period()["current_window"],
        "manual_override": bool(weights),
    }
    review = _quality_issues(peer.set_index("iata").loc[code].to_dict() | {"iata": code})
    if len(peer) < config.LOW_CONFIDENCE_MIN_PEERS:
        review.append(f"Only {len(peer)} airports in the peer group (< {config.LOW_CONFIDENCE_MIN_PEERS}).")
    return _response(results, assumptions, caveats, review)


def live_status(code=None, timeout=10):
    """Current delays / ground stops / closures from the FAA NAS status feed (optional, live).

    Never raises: if the feed is unreachable the agent gets status "unavailable".
    """
    try:
        resp = requests.get(config.FAA_STATUS_URL, timeout=timeout,
                            headers={"User-Agent": "airport-investment-agent/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        return _response([], status="unavailable", caveats=[
            f"FAA live status feed is unavailable right now ({type(exc).__name__}). "
            "Historical analysis is not affected."])

    events = []
    for delay_type in root.iter("Delay_type"):
        kind = (delay_type.findtext("Name") or "").strip()
        for el in delay_type.iter():
            airport = el.findtext("ARPT")
            if airport is None:
                continue
            details = {}
            for child in el.iter():
                if child is el or child.tag == "ARPT":
                    continue
                details.update({f"{child.tag}_{k}": v for k, v in child.attrib.items()})
                if len(child) == 0 and (child.text or "").strip():
                    details[child.tag] = child.text.strip()
            events.append({"airport": airport.strip(), "type": kind, **details})

    wanted = code.strip().upper() if code else None
    if wanted:
        events = [e for e in events if e["airport"] == wanted]
    results = {"update_time": (root.findtext("Update_Time") or "").strip(), "airport": wanted or "all",
               "events": events}
    caveats = ["Real-time snapshot of current conditions only; it is not used in any score.",
               "The FAA feed lists only active delays/closures; no events usually means normal operations. "
               "Some 'closure' entries are partial restrictions."]
    return _response(results, {"source": config.FAA_STATUS_URL}, caveats)
