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
