# RAG Knowledge Base Chatbot

Document Q&A with semantic search and source citations. Ask questions, get answers grounded in your documents.

**Stack**: Python · Flask · ChromaDB (local ONNX embeddings) · OpenRouter free LLM

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY
python3 ingest.py     # load docs from docs/ into ChromaDB
python3 app.py        # start API on :5050
```

## API

```
POST /chat    { "query": "your question" }  →  { "query": "...", "response": { "answer": "...", "sources": [...] } }
POST /ingest  →  re-run ingest pipeline
GET  /health  →  { "ok": true }
```

## How it works

1. `ingest.py` chunks `.txt` files from `docs/` (500 chars, 50 overlap), embeds with `all-MiniLM-L6-v2`, stores in ChromaDB
2. `query.py` embeds the question, retrieves top-3 chunks, builds a grounded prompt, calls the LLM
3. LLM synthesises an answer citing source files

**Live demo**: [link]
