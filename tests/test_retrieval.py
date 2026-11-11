import ingest

from app.retrieval import hybrid_search, rrf


def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    assert scores[1] > scores[2] and scores[3] > scores[4]
    assert max(scores, key=scores.get) in (1, 3)


def test_chunk_covers_all_text():
    text = "\n\n".join(f"para {i} " + "x" * 200 for i in range(5))
    chunks = ingest.chunk(text)
    assert len(chunks) > 1 and all("para" in c for c in chunks)


async def test_hybrid_finds_pto(index_dir):
    hits = await hybrid_search("how many PTO days per year")
    assert hits[0]["source"] == "handbook.md"
    assert "PTO" in hits[0]["content"]
    assert hits[0]["cosine"] > 0.2


async def test_bm25_leg_catches_exact_keyword(index_dir):
    hits = await hybrid_search("phishing", k=1)
    assert "phishing" in hits[0]["content"]
