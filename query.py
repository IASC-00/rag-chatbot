"""
RAG query pipeline.

Embeddings: ChromaDB DefaultEmbeddingFunction (local ONNX, no API key needed).
LLM synthesis: OpenRouter free tier (requires OPENROUTER_API_KEY in .env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH", str(Path.home() / ".rag-chatbot" / "chroma")
)
COLLECTION = "rag_docs"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "")

_FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "arcee-ai/trinity-large-preview:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]
TOP_K = 3

_embedding_fn = DefaultEmbeddingFunction()
_client = None
_collection = None


def get_collection():
    """Return (or create) the persistent ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=_embedding_fn,
    )
    return _collection


def embed_texts(texts):
    """Embed a list of strings using ChromaDB's local DefaultEmbeddingFunction."""
    return _embedding_fn(texts)


def retrieve(query_text, n_results=TOP_K):
    """
    Embed the query and retrieve the top-k most similar chunks from ChromaDB.
    Returns list of {text, source, distance} dicts.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            }
        )
    return chunks


def build_prompt(question, context_chunks):
    """Build the RAG prompt from question + retrieved chunks."""
    if not context_chunks:
        return f"Answer this question as best you can:\n\n{question}"

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Always cite the source file(s) you used at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""


def call_llm(prompt):
    """Call OpenRouter chat completion, trying fallback models on rate-limit. Returns the answer string."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "[No OPENROUTER_API_KEY set — add it to .env to enable LLM synthesis]"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    models = [CHAT_MODEL] + _FALLBACK_MODELS if CHAT_MODEL else _FALLBACK_MODELS
    last_err = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
    raise last_err


def answer(question, context_chunks=None):
    """
    Full answer synthesis: build prompt from retrieved context, call LLM.
    Returns {answer, sources} dict.
    """
    if context_chunks is None:
        context_chunks = retrieve(question)
    prompt = build_prompt(question, context_chunks)
    text = call_llm(prompt)
    sources = list({c["source"] for c in context_chunks})
    return {"answer": text, "sources": sources}


def pipe(question):
    """Full RAG pipeline: retrieve → answer. Returns {answer, sources}."""
    chunks = retrieve(question)
    return answer(question, chunks)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are REST API best practices?"
    result = pipe(q)
    print(f"\nAnswer: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
