"""UI tests with Streamlit's AppTest. The real Agent is replaced by a fake, so no Gemini calls are made."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent as agent_module
import config
import tools

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeAgent:
    """Returns canned turns built from real tool results on the fixture DB."""
    def __init__(self, pending_first=False):
        self.analyst_overrides, self.confirm_ranking, self.model_used = {}, True, "fake-model"
        self.pending_first = pending_first
        self.resumed_with = "not called"

    def _turn(self, question, **extra):
        turn = agent_module._new_turn(question)
        turn.update(extra)
        return turn

    def ask(self, question):
        if self.pending_first:
            pending = agent_module.preview_assumptions("rank_airports", {"region_or_states": "New England"})
            return self._turn(question, pending=pending)
        result = tools.rank_airports("New England", min_enplanements=0)
        return self._turn(question, answer="BIG ranks first.", needs_review=result["needs_review"],
                          review_reasons=result["review_reasons"],
                          tool_calls=[{"name": "rank_airports", "args": {"region_or_states": "New England"},
                                       "result": result}])

    def resume(self, args_override=None):
        self.resumed_with = args_override
        self.pending_first = False
        return self.ask("Which airports in New England should expand?")

    def reset(self):
        pass


@pytest.fixture
def app(fixture_db, monkeypatch, tmp_path):
    tools.set_connection(fixture_db)
    monkeypatch.setattr(config, "GEMINI_API_KEYS", ["test-key"])
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    yield
    tools.set_connection(None)


def start(fake):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["agent"] = fake
    at.run()
    assert not at.exception
    return at


def test_app_loads_with_sidebar(app):
    at = start(FakeAgent())
    assert "2025-08 to 2026-07" in at.sidebar.caption[0].value
    assert len(at.sidebar.slider) == len(config.SCORE_WEIGHTS["investment"])


def test_question_shows_answer_and_calculation(app):
    at = start(FakeAgent())
    at.chat_input[0].set_value("Which airports in New England should expand?").run()
    assert not at.exception
    assert any("BIG ranks first." in m.value for m in at.markdown)
    assert "How this was calculated" in [e.label for e in at.expander]
    assert len(at.dataframe) >= 2      # metrics table + score breakdown


def test_confirmation_card_then_run(app):
    fake = FakeAgent(pending_first=True)
    at = start(fake)
    at.chat_input[0].set_value("Which airports in New England should expand?").run()
    assert any("check these assumptions" in m.value for m in at.markdown)
    assert any("What I'll do:" in m.value and "New England" in m.value for m in at.markdown)
    at.button(key="run_1").click().run()
    assert not at.exception
    assert fake.resumed_with is None   # ran with the shown assumptions
    assert any("BIG ranks first." in m.value for m in at.markdown)


def test_example_button_sends_question(app):
    at = start(FakeAgent())
    at.sidebar.button(key="example_0").click().run()
    assert not at.exception
    assert at.session_state["messages"][0]["content"].startswith("Which airports in New England")
