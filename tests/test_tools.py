import json

import pytest

import tools


@pytest.fixture(autouse=True)
def use_fixture_db(fixture_db):
    tools.set_connection(fixture_db)
    yield
    tools.set_connection(None)


def test_responses_are_json_serializable():
    for result in [tools.resolve_airport("BIG"), tools.list_airports("New England", 0), tools.airport_profile("BIG")]:
        json.dumps(result)
        assert {"results", "assumptions", "data_period", "caveats", "needs_review"} <= set(result)


# --- resolve_airport -------------------------------------------------------------
def test_resolve_exact_code():
    r = tools.resolve_airport("big")
    assert r["status"] == "ok" and r["results"][0]["iata"] == "BIG"


def test_resolve_fuzzy_name():
    r = tools.resolve_airport("Midtown")   # city name, not in the alias table
    assert r["status"] == "ok" and r["results"][0]["iata"] == "MID"


def test_resolve_ambiguous_name():
    assert tools.resolve_airport("New York")["status"] == "ambiguous"


def test_resolve_not_found():
    assert tools.resolve_airport("Atlantis Spaceport")["status"] == "not_found"


# --- list_airports ---------------------------------------------------------------
def test_list_region_applies_threshold():
    all_ne = tools.list_airports("New England", min_enplanements=0)
    assert {a["iata"] for a in all_ne["results"]} == {"BIG", "MID", "NRW"}   # SML has no traffic
    big_only = tools.list_airports("New England", min_enplanements=10_000)
    assert [a["iata"] for a in big_only["results"]] == ["BIG"]
    assert big_only["needs_review"]   # peer group < 3


def test_list_state_name_and_code():
    assert tools.list_airports("Massachusetts", 0)["assumptions"]["states"] == ["MA"]
    assert tools.list_airports("CT", 0)["assumptions"]["states"] == ["CT"]


def test_list_la_is_ambiguous():
    r = tools.list_airports("LA")
    assert r["status"] == "ambiguous" and "Louisiana" in r["caveats"][0]


def test_list_unknown_region():
    assert tools.list_airports("Middle Earth")["status"] == "not_found"


# --- airport_profile -------------------------------------------------------------
def test_profile_metrics_and_period():
    r = tools.airport_profile("BIG")
    assert r["results"]["load_factor"] == pytest.approx(2000 / 2400, abs=1e-4)
    assert r["data_period"]["current_window"] == "2025-08 to 2026-07"
    assert not r["needs_review"] or all("BIG" in x for x in r["review_reasons"])


def test_profile_flags_missing_data():
    r = tools.airport_profile("NRW")
    reasons = " ".join(r["review_reasons"])
    assert r["needs_review"] and "runway" in reasons and "on-time" in reasons


def test_profile_unknown_code():
    assert tools.airport_profile("XXX")["status"] == "not_found"


# --- rank_airports -----------------------------------------------------------------
def test_rank_region_returns_breakdown_that_sums_to_score():
    r = tools.rank_airports("New England", min_enplanements=0)
    json.dumps(r)
    ranking = r["results"]["ranking"]
    assert [a["rank"] for a in ranking] == sorted(a["rank"] for a in ranking)
    for a in ranking:
        if a["score"] is not None:
            total = sum(p["contribution"] for p in a["breakdown"] if p["contribution"] is not None)
            assert total == pytest.approx(a["score"], abs=1e-3)
    assert r["assumptions"]["region"] == "New England"


def test_rank_weight_override_and_exclusion_are_reported():
    r = tools.rank_airports("New England", weights={"yoy_growth": 1.0}, exclude=["NRW"], min_enplanements=0)
    assert r["assumptions"]["manual_override"]
    assert r["assumptions"]["excluded_by_analyst"] == ["NRW"]
    assert "NRW" not in {a["iata"] for a in r["results"]["ranking"]}


def test_rank_bad_weight_is_an_error_not_a_crash():
    assert tools.rank_airports("New England", weights={"bogus": 1}, min_enplanements=0)["status"] == "error"


def test_rank_unknown_region():
    assert tools.rank_airports("Middle Earth")["status"] == "not_found"


def test_rank_explicit_airport_list_ignores_size_threshold():
    r = tools.rank_airports(airports=["BIG", "MID", "NRW"])      # all far below 500k passengers
    assert r["status"] == "ok"
    assert set(r["assumptions"]["airports_compared"]) == {"BIG", "MID", "NRW"}


def test_rank_city_area(monkeypatch):
    monkeypatch.setitem(tools.config.METRO_AREAS, "test city area", ["BIG", "MID", "XXX"])
    r = tools.rank_airports("Test City Area")
    assert r["assumptions"]["region"] == "Test City Area"
    assert {a["iata"] for a in r["results"]["ranking"]} == {"BIG", "MID"}
    assert any("XXX" in c for c in r["caveats"])                  # listed but no data


def test_new_york_region_is_ambiguous():
    r = tools.rank_airports("New York")
    assert r["status"] == "ambiguous" and "city area" in r["caveats"][0]


# --- compare_airports --------------------------------------------------------------
def test_compare_two_airports():
    r = tools.compare_airports(["BIG", "MID"], min_enplanements=0)
    json.dumps(r)
    assert {a["iata"] for a in r["results"]["airports"]} == {"BIG", "MID"}
    assert r["results"]["by_metric"]["delay_rate"]["highest"] == "BIG"   # 25% vs ~10%


def test_compare_accepts_comma_string_and_reports_missing():
    r = tools.compare_airports("BIG, MID, XXX", min_enplanements=0)
    assert r["status"] == "ok" and any("XXX" in c for c in r["caveats"])


def test_compare_needs_two_known_airports():
    assert tools.compare_airports(["BIG", "XXX"])["status"] == "not_found"


# --- long_haul_share ---------------------------------------------------------------
def test_long_haul_passenger_only_vs_cargo():
    r = tools.long_haul_share("BIG")
    json.dumps(r)
    assert r["results"]["long_haul_share"] == pytest.approx(0.5)          # LHR vs MID, equal departures
    assert r["results"]["share_including_cargo"] == pytest.approx(0.75)   # + PVG cargo
    assert any("cargo" in c for c in r["caveats"])
    assert r["results"]["top_long_haul_destinations"][0]["dest"] == "LHR"


def test_long_haul_threshold_is_configurable():
    assert tools.long_haul_share("BIG", threshold_miles=8000)["results"]["long_haul_share"] == 0


def test_long_haul_unknown_airport():
    assert tools.long_haul_share("XXX")["status"] == "not_found"


# --- unmet_demand ------------------------------------------------------------------
def test_unmet_demand_explains_drivers():
    r = tools.unmet_demand("BIG", min_enplanements=0)
    json.dumps(r)
    res = r["results"]
    assert 0 <= res["unmet_demand_index"] <= 100
    assert res["top_drivers"] and res["signals_vs_peers"]
    assert any("proxy" in c for c in r["caveats"])


def test_unmet_demand_adds_small_airport_to_peer_group():
    r = tools.unmet_demand("MID")   # default threshold: no fixture airport qualifies
    assert r["status"] == "ok" and any("added for comparison" in c for c in r["caveats"])


# --- live_status -------------------------------------------------------------------
FAA_XML = b"""<AIRPORT_STATUS_INFORMATION><Update_Time>Thu Oct 15 17:19:45 2026 GMT</Update_Time>
<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>
<Ground_Delay><ARPT>BOS</ARPT><Reason>runway construction</Reason><Avg>12 minutes</Avg></Ground_Delay>
<Ground_Delay><ARPT>SAN</ARPT><Reason>volume</Reason><Avg>39 minutes</Avg></Ground_Delay>
</Ground_Delay_List></Delay_type></AIRPORT_STATUS_INFORMATION>"""


class _FakeResponse:
    content = FAA_XML

    def raise_for_status(self):
        pass


def test_live_status_parses_and_filters(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    r = tools.live_status("bos")
    events = r["results"]["events"]
    assert len(events) == 1 and events[0]["Reason"] == "runway construction"


def test_live_status_unavailable_does_not_crash(monkeypatch):
    def boom(*a, **k):
        raise tools.requests.ConnectionError("blocked")
    monkeypatch.setattr(tools.requests, "get", boom)
    assert tools.live_status("BOS")["status"] == "unavailable"
