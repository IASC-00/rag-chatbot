"""
Ingest pipeline: load .txt docs → chunk → embed → store in ChromaDB.
Usage: python ingest.py
"""

import os
import glob
from pathlib import Path
from query import get_collection, embed_texts

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 500
OVERLAP = 50


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_document(doc, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split document into overlapping character-level chunks."""
    text = doc["text"]
    source = doc["source"]
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"source": source, "text": text[start:end], "chunk": idx})
        if end == len(text):
            break
        start += chunk_size - overlap
        idx += 1
    return chunks


def ingest():
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("  No .txt files found in docs/")
        return {"documents": 0, "chunks": 0}
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} documents → {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    idxs = [c["chunk"] for c in chunks]
    ids = [f"{s}__chunk{i}" for s, i in zip(sources, idxs)]
    metadatas = [{"source": s, "chunk": i} for s, i in zip(sources, idxs)]

    print("Embedding and storing in ChromaDB...")
    embeddings = embed_texts(texts)
    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    # Upsert alone kept chunks from files that were removed from docs/.
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if stale:
        collection.delete(ids=stale)
        print(f"  Removed {len(stale)} chunks from documents no longer in docs/.")
    print(f"  Done. {collection.count()} total chunks in store.")
    return {"documents": len(docs), "chunks": len(chunks)}


if __name__ == "__main__":
    ingest()
