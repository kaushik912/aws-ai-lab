"""Build the local index: ./data/*.md -> chunk -> Titan embed -> ./index/{faiss.index,chunks.json}.

Usage: python scripts/ingest.py     (needs AWS creds + Bedrock access to Titan Embed v2)
Cost: handbook sample ~ 1-2K tokens => well under $0.01. Rerun only when data/ changes.
"""
import json
import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.config import get_settings
from app.llm import embed_sync

ROOT = pathlib.Path(__file__).resolve().parent.parent


def chunk(text: str, size: int = 450, overlap: int = 60) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to ~size chars, carry overlap."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > size:
            out.append(cur.strip())
            cur = cur[-overlap:] + "\n\n" + para
        else:
            cur += "\n\n" + para
    if cur.strip():
        out.append(cur.strip())
    return out


def build_index(chunks: list[dict], vectors: np.ndarray, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; fine to ~100K chunks
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=1))


def main() -> None:
    s = get_settings()
    chunks = []
    for path in sorted((ROOT / "data").glob("*.md")):
        for i, c in enumerate(chunk(path.read_text())):
            chunks.append({"id": f"{path.stem}-{i}", "content": c, "source": path.name})
    if not chunks:
        sys.exit("no docs in ./data")
    vectors = np.asarray([embed_sync(c["content"]) for c in chunks], dtype="float32")
    build_index(chunks, vectors, ROOT / s.index_dir)
    print(f"indexed {len(chunks)} chunks, dims={vectors.shape[1]} -> {s.index_dir}/")


if __name__ == "__main__":
    main()
