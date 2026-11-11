import pytest
from fastapi.testclient import TestClient

from app.agent import DECLINE, safe_eval
from app.main import app


def test_healthz():
    assert TestClient(app).get("/healthz").json() == {"status": "ok"}


def test_ask_validates_input():
    assert TestClient(app).post("/ask", json={"question": ""}).status_code == 422


def test_ask_grounded_calls_llm(index_dir, fake_converse):
    r = TestClient(app).post("/ask", json={"question": "how many PTO days per year?"})
    assert r.status_code == 200
    body = r.json()
    assert "20 days" in body["answer"] and body["sources"]
    assert len(fake_converse) == 1
    assert "handbook.md" in fake_converse[0]["system"]  # context was injected


def test_ask_ungrounded_declines_without_llm(index_dir, fake_converse):
    r = TestClient(app).post("/ask", json={"question": "zzzz qqqq"})
    assert r.json()["answer"] == DECLINE
    assert fake_converse == []  # no tokens spent


def test_agent_tool_loop(index_dir, monkeypatch):
    from app import agent

    script = [
        {"output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "t1", "name": "calc", "input": {"expression": "60*5"}}}]}},
         "stopReason": "tool_use", "usage": {"inputTokens": 5, "outputTokens": 5}},
        {"output": {"message": {"role": "assistant", "content": [{"text": "300"}]}},
         "stopReason": "end_turn", "usage": {"inputTokens": 5, "outputTokens": 1}},
    ]

    async def fake(messages, system, tools=None):
        return script.pop(0)

    monkeypatch.setattr(agent, "converse", fake)
    out = TestClient(app).post("/agent", json={"question": "5 days of meals?"}).json()
    assert out["answer"] == "300"
    assert out["tool_calls"] == [{"tool": "calc", "input": {"expression": "60*5"}}]
    assert out["usage"]["inputTokens"] == 10


def test_safe_eval():
    assert safe_eval("60*5 + 800") == 1100
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('x')")
