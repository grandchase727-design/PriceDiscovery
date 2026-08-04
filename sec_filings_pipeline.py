# -*- coding: utf-8 -*-
"""sec_filings_pipeline.py — SEC EDGAR 10-K / 10-Q / 8-K ingestion → chunk → embed → store.

OFFLINE pipeline (run quarterly). Builds a local vector index of 10-K/10-Q filing
text for US large-caps so the Per-Ticker Debate (phase5) can retrieve a pick's
actual risk factors / MD&A at debate time.

  EDGAR (free, no API key) → clean HTML → chunk → bge-small embed (local, ONNX) →
  .sec_rag/index.npy (embeddings) + .sec_rag/meta.json (chunk text + ticker/form/date)

Run:
  python3 sec_filings_pipeline.py                 # default: US stocks in universe
  python3 sec_filings_pipeline.py AAPL NVDA MSFT  # specific tickers
  python3 sec_filings_pipeline.py --skip-existing # incremental (skip already-indexed accessions)

Dependencies: fastembed (ONNX bge-small), numpy, requests.  No torch, no API key.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

# SEC requires a descriptive User-Agent (company/app + email).
SEC_HEADERS = {"User-Agent": "PriceDiscovery research grandchase727@gmail.com",
               "Accept-Encoding": "gzip, deflate"}
RAG_DIR = Path(".sec_rag")
EMB_PATH = RAG_DIR / "index.npy"
META_PATH = RAG_DIR / "meta.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX via fastembed
CHUNK_CHARS = 3000          # ~800 tokens per chunk
CHUNK_OVERLAP = 300
FORMS = ("10-K", "10-Q", "8-K")
# latest annual + 2 recent quarterlies + 8 recent material-event reports (8-K:
# earnings releases, M&A, guidance updates, exec changes — event-driven, recent quarter).
MAX_PER_FORM = {"10-K": 1, "10-Q": 2, "8-K": 8}
RATE_DELAY = 0.25          # SEC politeness (≤10 req/s)
_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


# ─────────────────────────────────────────────────────────────────
def _log(m): print(m, flush=True)


def _ticker_cik_map() -> dict:
    """ticker(upper) → zero-padded 10-digit CIK."""
    r = requests.get(_TICKER_CIK_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


def _recent_filings(cik: str) -> list[dict]:
    """Return recent filings metadata for a CIK (form, accession, primaryDocument, date)."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    rec = r.json().get("filings", {}).get("recent", {})
    forms = rec.get("form", []); accs = rec.get("accessionNumber", [])
    docs = rec.get("primaryDocument", []); dates = rec.get("filingDate", [])
    out = []
    for f, a, d, dt in zip(forms, accs, docs, dates):
        if f in FORMS:
            out.append({"form": f, "accession": a, "doc": d, "date": dt})
    return out


def _download_filing_text(cik: str, accession: str, doc: str) -> str:
    """Download a filing's primary document and strip to plain text."""
    acc_nodash = accession.replace("-", "")
    cik_int = str(int(cik))   # path uses un-padded CIK
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    html = r.text
    # strip scripts/styles, tags → text
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"&#160;|&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html); html = re.sub(r"&#39;|&apos;", "'", html)
    text = re.sub(r"[ \t\xa0]+", " ", html)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _chunk(text: str) -> list[str]:
    """Split into ~CHUNK_CHARS chunks with overlap. Skip boilerplate-heavy short pieces."""
    chunks = []
    i, n = 0, len(text)
    while i < n:
        piece = text[i:i + CHUNK_CHARS].strip()
        if len(piece) > 250:   # keep substantive pieces (incl. short 8-K bodies)
            chunks.append(piece)
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def _load_index() -> tuple[np.ndarray, list[dict]]:
    if EMB_PATH.exists() and META_PATH.exists():
        embs = np.load(EMB_PATH)
        meta = json.loads(META_PATH.read_text())
        return embs, meta
    return np.zeros((0, 384), dtype=np.float32), []


def _save_index(embs: np.ndarray, meta: list[dict]):
    RAG_DIR.mkdir(exist_ok=True)
    np.save(EMB_PATH, embs.astype(np.float32))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False))


def _default_tickers() -> list[str]:
    """US stocks from the system universe (no foreign suffix) — the 10-K/Q-eligible set."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import price_discovery as pd
        ts = set()
        for cat in pd.STOCK_UNIVERSE.values():
            for t in cat.get("tickers", {}):
                if "." not in t and t.isalpha():   # US listing (no .KS/.T suffix)
                    ts.add(t.upper())
        return sorted(ts)
    except Exception as e:
        _log(f"⚠ universe load failed ({e}); using a small default set")
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


def ingest(tickers: list[str] = None, skip_existing: bool = False):
    tickers = tickers or _default_tickers()
    _log(f"📥 SEC ingest: {len(tickers)} tickers, forms={FORMS}")
    from fastembed import TextEmbedding
    model = TextEmbedding(EMBED_MODEL)
    embs, meta = _load_index()
    done_acc = {m["accession"] for m in meta} if skip_existing else set()
    embs_list = [embs] if embs.shape[0] else []

    try:
        cik_map = _ticker_cik_map()
    except Exception as e:
        _log(f"✗ ticker→CIK map failed: {e}"); return

    n_files = n_chunks = 0
    for ti, tk in enumerate(tickers, 1):
        cik = cik_map.get(tk.upper())
        if not cik:
            _log(f"  [{ti}/{len(tickers)}] {tk}: CIK 없음 (US 미상장?) — skip"); continue
        try:
            filings = _recent_filings(cik); time.sleep(RATE_DELAY)
        except Exception as e:
            _log(f"  [{ti}/{len(tickers)}] {tk}: 제출목록 실패 {str(e)[:50]}"); continue
        # latest N per form
        picked, per = [], {f: 0 for f in FORMS}
        for fl in filings:
            if per[fl["form"]] < MAX_PER_FORM[fl["form"]]:
                picked.append(fl); per[fl["form"]] += 1
        new_chunks_meta = []
        for fl in picked:
            if fl["accession"] in done_acc:
                continue
            try:
                txt = _download_filing_text(cik, fl["accession"], fl["doc"]); time.sleep(RATE_DELAY)
            except Exception as e:
                _log(f"      {tk} {fl['form']} {fl['date']}: 다운로드 실패 {str(e)[:40]}"); continue
            for ch in _chunk(txt):
                new_chunks_meta.append({"ticker": tk.upper(), "form": fl["form"],
                                         "date": fl["date"], "accession": fl["accession"],
                                         "text": ch})
            n_files += 1
        if new_chunks_meta:
            vecs = list(model.embed([m["text"] for m in new_chunks_meta]))
            embs_list.append(np.array(vecs, dtype=np.float32))
            meta.extend(new_chunks_meta)
            n_chunks += len(new_chunks_meta)
            _log(f"  [{ti}/{len(tickers)}] {tk}: +{len(new_chunks_meta)} chunks ({len(picked)} filings)")
        # periodic save
        if ti % 10 == 0 and embs_list:
            _save_index(np.vstack(embs_list), meta); embs_list = [np.vstack(embs_list)]

    if embs_list:
        _save_index(np.vstack(embs_list), meta)
    _log(f"✅ SEC ingest 완료: {n_files} filings, {n_chunks} new chunks, total {len(meta)} chunks")
    _log(f"   index: {EMB_PATH} ({EMB_PATH.stat().st_size//1024//1024 if EMB_PATH.exists() else 0}MB)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = "--skip-existing" in sys.argv
    ingest(args or None, skip_existing=skip)
