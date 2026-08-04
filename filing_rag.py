# -*- coding: utf-8 -*-
"""filing_rag.py — runtime retrieval over the SEC 10-K/10-Q vector index.

Loads the index built by sec_filings_pipeline.py ONCE (process-cached) and serves
semantic top-k chunks for a ticker. Used by the Per-Ticker Debate to ground
Trading/Risk/Critic verdicts in the company's actual filings.

  get_filing_context(ticker, query, k) → [{form, date, text, score}, ...]

Fast: vector search is numpy cosine over the (usually <100k) chunk matrix (~ms).
Embedding the query uses the same local bge-small (ONNX). Degrades gracefully to
[] if the index or fastembed is unavailable (so the swarm never breaks).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

RAG_DIR = Path(".sec_rag")
EMB_PATH = RAG_DIR / "index.npy"
META_PATH = RAG_DIR / "meta.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_EMBS: Optional[np.ndarray] = None
_META: Optional[list] = None
_TICKER_ROWS: Optional[dict] = None   # ticker → row indices
_MODEL = None
_LOADED = False
_AVAILABLE = False


def _ensure_loaded() -> bool:
    """Lazy-load index + embedding model once. Returns True if RAG is usable."""
    global _EMBS, _META, _TICKER_ROWS, _MODEL, _LOADED, _AVAILABLE
    if _LOADED:
        return _AVAILABLE
    _LOADED = True
    try:
        if not (EMB_PATH.exists() and META_PATH.exists()):
            return False
        _EMBS = np.load(EMB_PATH)
        _META = json.loads(META_PATH.read_text())
        if _EMBS.shape[0] == 0 or not _META:
            return False
        # normalize once for cosine via dot product
        norms = np.linalg.norm(_EMBS, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _EMBS = _EMBS / norms
        _TICKER_ROWS = {}
        for i, m in enumerate(_META):
            _TICKER_ROWS.setdefault(m.get("ticker", "").upper(), []).append(i)
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(EMBED_MODEL)
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def get_filing_context(ticker: str, query: str, k: int = 4) -> list[dict]:
    """Semantic top-k filing chunks for `ticker` matching `query`. [] if unavailable."""
    if not _ensure_loaded():
        return []
    rows = _TICKER_ROWS.get((ticker or "").upper())
    if not rows:
        return []
    try:
        q = next(iter(_MODEL.embed([query])))
        q = np.asarray(q, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sub = _EMBS[rows]                      # (n_ticker_chunks, 384), already normalized
        sims = sub @ q                          # cosine
        order = np.argsort(-sims)[:k]
        out = []
        for j in order:
            m = _META[rows[int(j)]]
            out.append({"form": m.get("form"), "date": m.get("date"),
                        "score": round(float(sims[int(j)]), 3),
                        "text": m.get("text", "")})
        return out
    except Exception:
        return []


def format_filing_block(ticker: str, query: str, k: int = 3, max_chars: int = 1400) -> str:
    """Compact prompt block of retrieved filing context (empty string if none)."""
    hits = get_filing_context(ticker, query, k)
    if not hits:
        return ""
    lines = [f"📄 {ticker} SEC 공시 발췌 (10-K/Q, 의미검색):"]
    budget = max_chars
    for h in hits:
        snippet = (h["text"] or "")[:600].strip()
        if budget <= 0:
            break
        snippet = snippet[:budget]
        budget -= len(snippet)
        lines.append(f"  [{h['form']} {h['date']} · sim {h['score']}] {snippet}")
    return "\n".join(lines)


def index_status() -> dict:
    """Quick status for diagnostics."""
    if not (EMB_PATH.exists() and META_PATH.exists()):
        return {"available": False, "reason": "no index — run sec_filings_pipeline.py"}
    try:
        meta = json.loads(META_PATH.read_text())
        tickers = sorted({m.get("ticker") for m in meta})
        return {"available": True, "chunks": len(meta), "tickers": len(tickers),
                "ticker_list": tickers[:50]}
    except Exception as e:
        return {"available": False, "reason": str(e)[:100]}
