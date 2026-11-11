"""No AWS in tests: hashed bag-of-words 'embeddings' + a fake Converse."""
import hashlib
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from app.retrieval import tokenize

DIMS = 256


def fake_vec(text: str) -> list[float]:
    v = np.zeros(DIMS, dtype="float32")
    for t in tokenize(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIMS] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    import ingest

    chunks = []
    for p in sorted((ROOT / "data").glob("*.md")):
        for i, c in enumerate(ingest.chunk(p.read_text())):
            chunks.append({"id": f"{p.stem}-{i}", "content": c, "source": p.name})
    vecs = np.asarray([fake_vec(c["content"]) for c in chunks], dtype="float32")
    ingest.build_index(chunks, vecs, tmp_path)

    from app import retrieval
    from app.config import get_settings

    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    retrieval.get_index.cache_clear()

    async def fake_embed(text):
        return fake_vec(text)

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    yield tmp_path
    get_settings.cache_clear()
    retrieval.get_index.cache_clear()


@pytest.fixture
def fake_converse(monkeypatch):
    """Replaces app.agent.converse; returns canned text and records calls."""
    calls = []

    async def fake(messages, system, tools=None):
        calls.append({"messages": messages, "system": system, "tools": tools})
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "PTO is 20 days [handbook.md]"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

    from app import agent

    monkeypatch.setattr(agent, "converse", fake)
    return calls
