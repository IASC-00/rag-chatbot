"""2026-10-18: swapping the sample documents left the old ones answering questions,
because ingest only ever upserted."""

import ingest
import query


def test_ingest_drops_chunks_from_removed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(query, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(query, "_collection", None)
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(ingest, "DOCS_DIR", docs)
    (docs / "old.txt").write_text("Old programming notes about REST APIs.")
    (docs / "new.txt").write_text("Refunds are given the same day with a receipt.")
    ingest.ingest()
    (docs / "old.txt").unlink()
    ingest.ingest()
    sources = {m["source"] for m in query.get_collection().get(include=["metadatas"])["metadatas"]}
    assert sources == {"new.txt"}


def test_emptying_docs_clears_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(query, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(query, "_collection", None)
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(ingest, "DOCS_DIR", docs)
    (docs / "old.txt").write_text("Old notes.")
    ingest.ingest()
    (docs / "old.txt").unlink()
    assert ingest.ingest() == {"documents": 0, "chunks": 0}
    assert query.get_collection().count() == 0
