"""Streamlit chat UI for the airport investment agent.

Run:  streamlit run app.py

This file only displays things. All logic lives in agent.py (LLM + loop) and tools.py / metrics.py /
scoring.py (all numbers).
"""
import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import config
import scoring
import tools
from agent import Agent

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]
WEIGHT_LABELS = {k: v.capitalize() for k, v in config.METRIC_LABELS.items()}
TOOL_LABELS = {
    "resolve_airport": "Find the airport", "list_airports": "List airports in the region",
    "airport_profile": "Airport profile", "rank_airports": "Rank airports",
    "compare_airports": "Compare airports", "long_haul_share": "Long-haul share",
    "unmet_demand": "Unmet demand index", "live_status": "Live FAA status",
}
METRIC_FORMATS = {   # how raw metrics are shown in tables
    "passengers": "{:,.0f}", "load_factor": "{:.1%}", "yoy_growth": "{:+.1%}", "delay_rate": "{:.1%}",
    "cancel_rate": "{:.2%}", "avg_taxi_out": "{:.1f} min", "departures_per_runway": "{:,.0f}",
    "long_haul_share": "{:.1%}",
}

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])          # [{"role": "user", "content"} | {"role": "assistant", "turn"}]
    st.session_state.setdefault("feedback_saved", set())


def get_agent():
    if "agent" not in st.session_state:                   # one Agent (= one conversation) per browser session
        st.session_state.agent = Agent()
    return st.session_state.agent


def save_feedback(entry):
    with open(config.FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def fmt(metric, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    pattern = METRIC_FORMATS.get(metric)
    return pattern.format(value) if pattern else value


# ---------------------------------------------------------------------------------
# Sidebar: data period, assumptions, weight sliders, exclusions, examples
# ---------------------------------------------------------------------------------
def reset_weights():
    for k, v in config.SCORE_WEIGHTS["investment"].items():
        st.session_state[f"w_{k}"] = float(v)
    st.session_state["exclude"] = []


def sidebar():
    """Returns (analyst_overrides, confirm_ranking)."""
    with st.sidebar:
        st.title("✈️ Airport Investment Agent")
        try:
            period = tools._data_period()
            metrics_df = tools._metrics()
        except Exception:
            st.error("Database not found. Run `python ingest.py` first (see README).")
            st.stop()
        st.caption(f"**Data:** {period['current_window']} · growth vs {period['prior_window']}")

        with st.expander("Scope & assumptions"):
            st.markdown(
                f"- US airports only, ≥ {config.MIN_ANNUAL_ENPLANEMENTS:,} departing passengers/year "
                f"(≈ 1M total passengers)\n"
                f"- Window: latest 12 months available (BTS publishes with a lag)\n"
                f"- Growth = passengers vs previous 12 months (YoY), not a forecast\n"
                f"- Long haul = segment > {config.LONG_HAUL_MILES:,} miles (passenger flights by default)\n"
                f"- Runways counted: open, ≥ {config.MIN_USABLE_RUNWAY_FT:,} ft\n"
                f"- *Unmet demand* is a proxy index (full planes + congestion + growth)\n"
                f"- All numbers are computed in Python; the AI only explains them")

        st.subheader("Investment score weights")
        defaults = config.SCORE_WEIGHTS["investment"]
        weights = {k: st.slider(WEIGHT_LABELS[k], 0.0, 1.0, float(v), 0.05, key=f"w_{k}")
                   for k, v in defaults.items()}
        changed = any(abs(weights[k] - defaults[k]) > 1e-9 for k in defaults)
        if changed:
            try:
                used = scoring.validate_weights("investment", weights)
                st.caption("Applied to rankings (re-normalized): " +
                           ", ".join(f"{WEIGHT_LABELS[k]} {v:.0%}" for k, v in used.items()))
            except ValueError as exc:
                st.warning(str(exc))
                changed = False

        universe = metrics_df[metrics_df["passengers"] >= config.MIN_ANNUAL_ENPLANEMENTS]
        exclude = st.multiselect("Exclude airports from rankings", sorted(universe["iata"]), key="exclude")
        st.button("Reset weights & exclusions", on_click=reset_weights)
        confirm = st.toggle("Confirm assumptions before ranking", value=True, key="confirm",
                            help="Show the peer group and weights and wait for 'Run' before any ranking.")

        st.subheader("Example questions")
        for n, q in enumerate(EXAMPLES):
            if st.button(q, key=f"example_{n}"):
                st.session_state.queued = q

        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.session_state.feedback_saved = set()
            get_agent().reset()
            st.rerun()

    return {"weights": weights if changed else None, "exclude": exclude}, confirm


# ---------------------------------------------------------------------------------
# Human-in-the-loop: confirm assumptions before a ranking runs
# ---------------------------------------------------------------------------------
def month_range(window):
    """'2025-08 to 2026-07' -> 'Aug 2025 – Jul 2026'."""
    try:
        start, end = (datetime.strptime(x.strip(), "%Y-%m").strftime("%b %Y") for x in window.split(" to "))
        return f"{start} – {end}"
    except ValueError:
        return window


def render_confirmation(i, turn):
    """Plain-language summary of what is about to run, so the analyst can check it before any ranking."""
    p = turn["pending"]
    weights = p["weights"] if "error" not in p["weights"] else {}
    st.markdown("**Before I run this, please check these assumptions:**")
    if p.get("problem"):
        st.warning(p["problem"])
    lines = [f"**What I'll do:** {p['action']}"]
    names = [f"{a['iata']} – {a['name']}" for a in p.get("airports", [])]
    if p["tool"] == "rank_airports":
        size_rule = "" if p.get("explicit_airports") else \
            f" *(only airports with {p['min_annual_enplanements']:,}+ departing passengers a year)*"
        shown = names if len(names) <= 15 else names[:15] + [f"… and {len(names) - 15} more"]
        lines.append(f"**Airports included ({len(names)}):** " + " · ".join(shown) + size_rule)
    else:
        lines.append(f"**Compared with:** {len(names)} airports with {p['min_annual_enplanements']:,}+ "
                     f"departing passengers a year")
    if weights:
        lines.append("**How the score is built:** " +
                     " · ".join(f"{config.METRIC_LABELS.get(k, k)} {v:.0%}" for k, v in weights.items()))
    else:
        lines.append(f"**Weights:** {p['weights'].get('error')}")
    if p["excluded"]:
        lines.append(f"**Left out by you:** {', '.join(p['excluded'])}")
    lines.append(f"**Data period:** {month_range(p['window'])}, growth compared with "
                 f"{month_range(p['growth_compared_to'])}")
    st.markdown("\n\n".join(lines))

    agent = get_agent()
    col1, col2 = st.columns(2)
    if col1.button("▶️ Run with these", key=f"run_{i}", type="primary"):
        with st.spinner("Running..."):
            st.session_state.messages[i]["turn"] = agent.resume()
            st.session_state.messages[i]["turn"]["model"] = agent.model_used
        st.rerun()
    if col2.button("✏️ Adjust", key=f"adjust_btn_{i}"):
        st.session_state[f"adjust_{i}"] = True

    if st.session_state.get(f"adjust_{i}"):
        with st.form(f"form_{i}"):
            new_weights = {k: st.slider(config.METRIC_LABELS.get(k, k).capitalize(), 0.0, 1.0,
                                        round(float(v), 2), 0.05, key=f"fw_{i}_{k}")
                           for k, v in weights.items()}
            new_exclude = None
            if p["tool"] == "rank_airports":
                m = tools._metrics()
                options = sorted(m.loc[m["passengers"] >= config.MIN_ANNUAL_ENPLANEMENTS, "iata"])
                new_exclude = st.multiselect("Exclude airports", options, default=[c for c in p["excluded"] if c in options])
            if st.form_submit_button("Run with adjusted assumptions", type="primary"):
                override = {"weights": new_weights}
                if new_exclude is not None:
                    override["exclude"] = new_exclude
                with st.spinner("Running..."):
                    st.session_state.messages[i]["turn"] = agent.resume(args_override=override)
                    st.session_state.messages[i]["turn"]["model"] = agent.model_used
                st.session_state[f"adjust_{i}"] = False
                st.rerun()


# ---------------------------------------------------------------------------------
# Rendering one answer: badges, text, clarification buttons, chart, calculation details, feedback
# ---------------------------------------------------------------------------------
def score_rows(turn):
    """(title, rows) from the last ranking / comparison in this turn, for the bar chart."""
    for call in reversed(turn["tool_calls"]):
        res = call["result"].get("results")
        if not isinstance(res, dict):
            continue
        label = config.SCORE_LABELS.get(call["result"]["assumptions"].get("score"), "Score")
        if call["name"] == "rank_airports":
            return f"{label} (0-100, within this group)", res.get("ranking", [])
        if call["name"] == "compare_airports":
            return f"{label} (0-100, vs all US airports)", res.get("airports", [])
    return None, []


def render_chart(turn):
    title, rows = score_rows(turn)
    data = pd.DataFrame([{"airport": r["iata"], "score": r["score"]} for r in rows if r.get("score") is not None])
    if data.empty:
        return
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("score:Q", title=title, scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("airport:N", sort="-x", title=None),
        tooltip=["airport", alt.Tooltip("score:Q", format=".1f")],
    ).properties(height=max(120, 32 * len(data)))
    st.altair_chart(chart)


def metrics_table(rows):
    table = []
    for r in rows:
        line = {"airport": r["iata"], "score": round(r["score"], 1) if r.get("score") is not None else None,
                "rank": r.get("rank")}
        line.update({k: fmt(k, v) for k, v in r.get("metrics", {}).items()})
        table.append(line)
    return pd.DataFrame(table)


def breakdown_table(rows):
    parts = [{"airport": r["iata"], **p} for r in rows for p in r.get("breakdown", [])]
    return pd.DataFrame(parts).round(3) if parts else pd.DataFrame()


def render_tool_result(name, result):
    res = result.get("results")
    if name in ("rank_airports", "compare_airports") and isinstance(res, dict):
        rows = res.get("ranking") or res.get("airports") or []
        st.dataframe(metrics_table(rows), hide_index=True)
        st.caption("Score breakdown: normalized value (0-100) × weight = contribution")
        st.dataframe(breakdown_table(rows), hide_index=True)
    elif name == "unmet_demand" and isinstance(res, dict) and res:
        st.markdown(f"Index **{res['unmet_demand_index']:.1f}**, rank {res['rank']:.0f} of {res['peer_group_size']}")
        st.dataframe(pd.DataFrame(res["drivers"]).round(3), hide_index=True)
        signals = pd.DataFrame(res["signals_vs_peers"]).T.reset_index(names="metric")
        st.dataframe(signals, hide_index=True)
    elif name == "long_haul_share" and isinstance(res, dict) and res:
        st.markdown(f"Share **{res['long_haul_share']:.1%}** "
                    f"({res['long_haul_departures']:,.0f} of {res['departures']:,.0f} departures)")
        st.dataframe(pd.DataFrame(res["top_long_haul_destinations"]), hide_index=True)
    elif isinstance(res, list) and res:
        st.dataframe(pd.DataFrame(res), hide_index=True)
    elif res:
        st.json(res, expanded=False)
    if result.get("caveats"):
        st.markdown("Caveats:\n" + "\n".join(f"- {c}" for c in result["caveats"]))
    if result.get("assumptions"):
        st.caption("Assumptions")
        st.json(result["assumptions"], expanded=False)


def render_calculation(turn):
    if not turn["tool_calls"]:
        return
    with st.expander("How this was calculated"):
        if turn.get("model"):
            st.caption(f"Model: {turn['model']} · topic check: {turn['topic']['method'] if turn['topic'] else '-'}")
        for n, call in enumerate(turn["tool_calls"], 1):
            st.markdown(f"**{n}. {TOOL_LABELS.get(call['name'], call['name'])}** "
                        f"<small>(`{call['name']}` {json.dumps(call['args'])} → {call['result'].get('status')})</small>",
                        unsafe_allow_html=True)
            render_tool_result(call["name"], call["result"])
            st.divider()


def render_feedback(i, turn):
    saved = st.session_state.feedback_saved
    if i in saved:
        st.caption("Thanks, feedback saved.")
        return
    rating = st.feedback("thumbs", key=f"fb_{i}")
    if rating is None:
        return
    comment = st.text_input("Optional comment", key=f"fbc_{i}")
    if st.button("Send feedback", key=f"fbs_{i}"):
        save_feedback({
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": turn["question"], "answer": turn["answer"],
            "rating": "up" if rating == 1 else "down", "comment": comment,
            "tools": [{"name": c["name"], "args": c["args"]} for c in turn["tool_calls"]],
            "needs_review": turn["needs_review"], "model": turn.get("model"),
        })
        saved.add(i)
        st.rerun()


def render_assistant(i, turn, is_last):
    if turn["pending"]:
        if is_last:
            render_confirmation(i, turn)
        else:
            st.caption("Not run: the analyst asked a different question instead.")
        return
    if turn["error"]:
        st.error(turn["answer"])
        return
    if turn["off_topic"]:
        st.info(turn["answer"])
        return
    if turn["needs_review"]:
        st.warning("**Needs analyst review:** " + "; ".join(turn["review_reasons"]))
    if turn["manual_override"] or turn["analyst_adjusted"]:
        st.info("Manual override applied (analyst-adjusted weights or excluded airports).")
    st.markdown(turn["answer"])

    clar = turn.get("clarification")
    if clar and clar.get("options") and is_last:
        st.caption("Choose one:")
        cols = st.columns(len(clar["options"]))
        for col, opt in zip(cols, clar["options"]):
            if col.button(f"{opt['iata']} – {opt['name']}", key=f"clar_{i}_{opt['iata']}"):
                st.session_state.queued = f"I mean {opt['name']} ({opt['iata']})."
                st.rerun()

    render_chart(turn)
    render_calculation(turn)
    render_feedback(i, turn)


# ---------------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------------
def main():
    init_state()
    if not config.GEMINI_API_KEYS:
        st.error("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key (see README).")
        st.stop()

    overrides, confirm = sidebar()
    agent = get_agent()
    agent.analyst_overrides = overrides
    agent.confirm_ranking = confirm

    st.header("US Airport Terminal Expansion Analyst")
    st.caption("Ask which airports have demand exceeding capacity. Every number comes from deterministic "
               "calculations on BTS data; open “How this was calculated” under any answer.")

    messages = st.session_state.messages
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(i, msg["turn"], is_last=(i == len(messages) - 1))

    question = st.chat_input("Ask about US airports...") or st.session_state.pop("queued", None)
    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Analyzing..."):
            try:
                turn = agent.ask(question)
                turn["model"] = agent.model_used
            except Exception as exc:   # never show a stack trace to the analyst
                turn = {"question": question, "answer": f"Something went wrong: {exc}", "error": str(exc),
                        "tool_calls": [], "pending": None, "off_topic": False}
        messages.append({"role": "assistant", "turn": turn})
        st.rerun()


main()
