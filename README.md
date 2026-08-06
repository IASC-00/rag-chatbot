# RAG Knowledge Base Chatbot

End-to-end RAG pipeline — ingest documents, embed locally with ONNX, retrieve semantically, synthesise answers with source citations. No external embedding API required.

**Live demo**: [rag-chatbot-4n44.onrender.com](https://rag-chatbot-4n44.onrender.com)

**Stack**: Python · Flask · ChromaDB · all-MiniLM-L6-v2 (ONNX) · OpenRouter

---

## Features

- Local ONNX embeddings — no OpenAI embedding API, runs offline
- ChromaDB vector store with persistent storage
- Multi-model LLM fallback chain handles OpenRouter rate limits
- Source citations returned with every answer
- `/ingest` endpoint to reload documents without restarting

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY
python3 ingest.py      # chunk + embed docs from docs/
python3 app.py         # API on :5050
```

## How it works

1. `ingest.py` chunks `.txt` files from `docs/` (500 chars, 50 char overlap), embeds with `all-MiniLM-L6-v2`, stores vectors in ChromaDB
2. On query: embed the question, retrieve top-3 chunks by cosine similarity, build a grounded prompt
3. LLM synthesises an answer and cites the source files

## API

```
POST /chat    { "query": "your question" }
POST /ingest  re-run the ingest pipeline
GET  /health
```

---

[iswain.dev](https://iswain.dev)

## /chat access and cost controls

`/chat` calls a paid model on every request, so the endpoint has two controls.
Setting neither preserves the open public-demo behaviour.

| Variable | Default | Effect |
|---|---|---|
| `CHAT_TOKEN` | unset | When set, `/chat` requires a matching `X-Chat-Token` header. Unset = open, as before. |
| `CHAT_MAX_REQUESTS` | `30` | Requests allowed per source address per window. Always enforced. |
| `CHAT_WINDOW_SECONDS` | `3600` | Length of that window. |

`/ingest` is separately gated by `INGEST_TOKEN` and is disabled (503) when unset.

**Limiter scope:** in-memory and single-process — state resets on restart and it
does not coordinate across workers. Fine for the single-process demo; a
multi-worker deployment needs shared state. Floods distributed across many
source addresses are not limited.
