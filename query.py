"""
RAG query pipeline.

Embeddings: ChromaDB DefaultEmbeddingFunction (local ONNX, no API key needed).
LLM synthesis: Groq free tier first (GROQ_API_KEY), then OpenRouter free models
(OPENROUTER_API_KEY). If every model is busy, the matching passages are returned as is.
"""

import logging
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
log = logging.getLogger(__name__)
# The OpenRouter key has no spend limit, so only ":free" slugs are ever sent unless
# ALLOW_PAID_MODELS=1 is set on purpose. OpenRouter retires free slugs without
# notice: on 2026-09-23 all four previous hard-coded fallbacks were gone and the
# primary was rate-limited upstream, so every chat failed. The list now lives in
# the environment (CHAT_FALLBACK_MODELS, comma-separated); re-check it against
# https://openrouter.ai/api/v1/models when the demo errors.
_ALLOW_PAID = os.environ.get("ALLOW_PAID_MODELS", "") == "1"
_DEFAULT_FALLBACKS = "google/gemma-4-31b-it:free,nvidia/nemotron-3-super-120b-a12b:free"


def _usable(model):
    return bool(model) and (_ALLOW_PAID or model.endswith(":free"))


CHAT_MODEL = os.environ.get("CHAT_MODEL", "").strip()
_FALLBACK_MODELS = [
    m.strip()
    for m in (os.environ.get("CHAT_FALLBACK_MODELS") or _DEFAULT_FALLBACKS).split(",")
    if _usable(m.strip())
]
# Groq's free tier (no card on the account) allows 1,000 requests a day per model,
# checked from the server's response headers on 2026-10-18. Tried before OpenRouter,
# whose free models stop at 50 requests a day. GROQ_MODELS overrides the list.
_DEFAULT_GROQ = "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b"
GROQ_MODELS = [
    m.strip() for m in (os.environ.get("GROQ_MODELS") or _DEFAULT_GROQ).split(",") if m.strip()
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


def _providers():
    """(base_url, api_key, models) for every provider with a key, cheapest-limit first."""
    out = []
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and GROQ_MODELS:
        out.append(("https://api.groq.com/openai/v1", groq_key, GROQ_MODELS))
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        primary = [CHAT_MODEL] if _usable(CHAT_MODEL) else []
        models = primary + [m for m in _FALLBACK_MODELS if m not in primary]
        if models:
            out.append(("https://openrouter.ai/api/v1", or_key, models))
    return out


def call_llm(prompt):
    """Ask each configured model in turn until one answers. Returns the answer string."""
    providers = _providers()
    if not providers:
        raise RuntimeError("No usable chat model configured (GROQ_API_KEY, or OPENROUTER_API_KEY + :free)")

    from openai import OpenAI

    last_err = None
    for base_url, api_key, models in providers:
        # One gunicorn worker serves every visitor, so a hung upstream call must not
        # hold it for the SDK default of 10 minutes.
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)
        for model in models:
            extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    extra_body=extra or None,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
                last_err = RuntimeError(f"{model} returned an empty answer")
            except Exception as e:
                last_err = e
    raise last_err


BUSY_PREFIX = (
    "The AI part is busy right now, so here are the passages from the documents"
    " that best match your question:"
)


def passages_answer(context_chunks):
    """What a visitor gets when every model is busy: the matching passages, verbatim."""
    if not context_chunks:
        return (
            "The AI part is busy right now and nothing in the documents matched that question."
            " Try again in a minute."
        )
    parts = [f"From {c['source']}:\n{c['text'].strip()}" for c in context_chunks]
    return BUSY_PREFIX + "\n\n" + "\n\n".join(parts)


def answer(question, context_chunks=None):
    """
    Full answer synthesis: build prompt from retrieved context, call LLM.
    Returns {answer, sources} dict.
    """
    if context_chunks is None:
        context_chunks = retrieve(question)
    prompt = build_prompt(question, context_chunks)
    sources = sorted({c["source"] for c in context_chunks})
    try:
        return {"answer": call_llm(prompt), "sources": sources}
    except Exception:
        log.exception("every chat model failed; answering with the passages")
        return {"answer": passages_answer(context_chunks), "sources": sources, "fallback": True}


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
