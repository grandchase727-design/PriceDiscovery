#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""daily_pipeline.py — Daily automation pipeline.

Executed by launchd at 07:00 KST daily.

Pipeline:
  1. Live Scan      — price + technicals (~1-2 min)
  2. Agent Scan     — Market Leaders Swarm + Backtest (~12-15 min)
  3. Final List     — 3-Agent voting (PM + Trading + Risk)
  4. PDF generation — Korean report
  5. Email send     — to configured recipient

Logs: /tmp/daily_pipeline_<YYYY-MM-DD>.log
"""
from __future__ import annotations

import json
import sys
import os
import time
import subprocess
from datetime import datetime
from pathlib import Path


# Always run from project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# Log file
LOG_PATH = Path(f"/tmp/daily_pipeline_{datetime.now().strftime('%Y-%m-%d')}.log")


def log(msg: str):
    """Log to stdout and file."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step_live_scan() -> dict:
    """Step 1: Live Scan."""
    log("=" * 60)
    log("[1/5] Live Scan 시작")
    log("=" * 60)
    try:
        from price_discovery import run_scan
        result = run_scan(include_stocks=True, use_realtime=False)
        n = len(result.get("results", [])) if isinstance(result, dict) else 0
        log(f"✓ Live Scan 완료 — {n}개 ticker scored")
        return {"ok": True, "n_tickers": n}
    except Exception as e:
        log(f"✗ Live Scan 실패: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_swarm() -> dict:
    """Step 2a: Market Leaders Swarm."""
    log("=" * 60)
    log("[2a/5] Market Leaders Swarm 시작")
    log("=" * 60)
    try:
        from agents.market_leaders_swarm import run_swarm
        payload = run_swarm()
        pm = (payload or {}).get("phase5_pm", {})
        n_picks = sum(
            len(pm.get("horizons", {}).get(h, {}).get(b, []) or [])
            for h in ("tactical", "core", "strategic")
            for b in ("long_stocks", "long_etfs", "short_stocks", "short_etfs")
        )
        log(f"✓ Swarm 완료 — {n_picks} picks generated")
        return {"ok": True, "n_picks": n_picks}
    except Exception as e:
        log(f"✗ Swarm 실패: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_backtest() -> dict:
    """Step 2b: Backtest."""
    log("=" * 60)
    log("[2b/5] Backtest 시작")
    log("=" * 60)
    try:
        result = subprocess.run(
            ["python3", "backtest/run.py"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log(f"✗ Backtest 실패: {result.stderr[:300]}")
            return {"ok": False, "error": result.stderr[:200]}
        log(f"✓ Backtest 완료")
        return {"ok": True}
    except Exception as e:
        log(f"✗ Backtest 예외: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_final_list() -> dict:
    """Step 3: Build Final List + Commentary."""
    log("=" * 60)
    log("[3/5] Final List 생성")
    log("=" * 60)
    try:
        from agents.final_list import build_final_lists
        data = build_final_lists()
        log(f"✓ Final List 완료 — 매수 {len(data.get('buy_list',[]))}개 · 매도 {len(data.get('sell_list',[]))}개")
        return {"ok": True, "data": data}
    except Exception as e:
        log(f"✗ Final List 실패: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_pdf(final_data: dict) -> dict:
    """Step 4: Generate PDF. Saves to /tmp AND reports/ for git commit."""
    log("=" * 60)
    log("[4/5] PDF 생성")
    log("=" * 60)
    try:
        from scripts.pdf_generator import build_daily_report_pdf
        sw_cache = {}
        cp = Path(".market_leaders_swarm_cache.json")
        if cp.exists():
            sw_cache = json.loads(cp.read_text(encoding="utf-8"))
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = f"/tmp/price_discovery_daily_{date_str}.pdf"
        path = build_daily_report_pdf(final_data, swarm_cache=sw_cache, output_path=out_path)
        # Also save to reports/ for GitHub commit
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        repo_pdf = reports_dir / f"price_discovery_daily_{date_str}.pdf"
        import shutil
        shutil.copy(path, repo_pdf)
        size_kb = Path(path).stat().st_size / 1024
        log(f"✓ PDF 생성 완료 — {path} ({size_kb:.1f} KB)")
        log(f"  Also copied to: {repo_pdf}")
        return {"ok": True, "path": path, "repo_path": str(repo_pdf)}
    except Exception as e:
        log(f"✗ PDF 실패: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_commit_push() -> dict:
    """Step 6: Commit + push PDF + cache to GitHub.

    This is the BACKUP mechanism — if Mac is off tomorrow, GitHub Actions
    will pick up this committed PDF and send it via Telegram.
    """
    log("=" * 60)
    log("[6/6] GitHub commit + push")
    log("=" * 60)
    try:
        import subprocess
        # Stage PDF + relevant cache (keep large pkl files OUT of repo via .gitignore)
        subprocess.run(["git", "add", "reports/"], cwd=PROJECT_ROOT, check=True)
        # Check if anything to commit
        diff = subprocess.run(["git", "diff", "--cached", "--name-only"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True)
        if not diff.stdout.strip():
            log("⚠ Nothing to commit (no changes)")
            return {"ok": True, "skipped": True}

        date_str = datetime.now().strftime("%Y-%m-%d")
        msg = f"Daily report {date_str} (auto)"
        subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
        push = subprocess.run(["git", "push", "origin", "main"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        if push.returncode != 0:
            log(f"⚠ Push 실패: {push.stderr[:200]}")
            return {"ok": False, "error": push.stderr[:200]}
        log(f"✓ Committed + pushed to GitHub")
        return {"ok": True}
    except Exception as e:
        log(f"⚠ Git commit/push 예외: {e}")
        return {"ok": False, "error": str(e)[:200]}


def step_telegram(pdf_path: str, final_data: dict) -> dict:
    """Step 5: Send via Telegram Bot."""
    log("=" * 60)
    log("[5/5] Telegram 발송")
    log("=" * 60)
    try:
        from scripts.telegram_sender import send_pdf_with_summary
        date_str = datetime.now().strftime("%Y-%m-%d")
        buy_list = final_data.get("buy_list", [])
        sell_list = final_data.get("sell_list", [])

        # Top 3 ★★★ tickers
        top_buy = ", ".join(r["ticker"] for r in buy_list if r.get("stars") == 3)[:80]
        top_sell = ", ".join(r["ticker"] for r in sell_list if r.get("stars") == 3)[:80]

        result = send_pdf_with_summary(
            pdf_path=pdf_path,
            summary={
                "date": date_str,
                "n_buy":  len(buy_list),
                "n_sell": len(sell_list),
                "top_buy_tickers":  top_buy or "—",
                "top_sell_tickers": top_sell or "—",
            },
        )
        if result.get("ok"):
            log(f"✓ Telegram 발송 완료 — chat_id={result.get('chat_id')} · {result.get('size_kb',0):.1f}KB")
        else:
            log(f"✗ Telegram 발송 실패: {result.get('error')}")
        return result
    except Exception as e:
        log(f"✗ Telegram 예외: {e}")
        return {"ok": False, "error": str(e)[:200]}


def main():
    log("")
    log("=" * 60)
    log(f"🤖 Daily Pipeline 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M KST')}")
    log("=" * 60)

    t0 = time.time()
    summary = {"steps": {}}

    # Step 1: Live Scan
    r = step_live_scan()
    summary["steps"]["live_scan"] = r
    if not r["ok"]:
        log("⚠ Live Scan 실패 — 계속 진행 (기존 cache 사용)")

    # Step 2a: Swarm
    r = step_swarm()
    summary["steps"]["swarm"] = r
    swarm_ok = r["ok"]

    # Step 2b: Backtest
    r = step_backtest()
    summary["steps"]["backtest"] = r

    # Step 3: Final List
    if not swarm_ok:
        log("⚠ Swarm 실패 — Final List는 기존 cache로 생성")
    r = step_final_list()
    summary["steps"]["final_list"] = r
    if not r["ok"]:
        log("✗ Final List 생성 실패 — 종료")
        return 1
    final_data = r["data"]

    # Step 4: PDF
    r = step_pdf(final_data)
    summary["steps"]["pdf"] = r
    if not r["ok"]:
        log("✗ PDF 생성 실패 — 종료")
        return 1
    pdf_path = r["path"]

    # Step 5: Telegram
    r = step_telegram(pdf_path, final_data)
    summary["steps"]["telegram"] = r

    # Step 6: Commit + push PDF to GitHub (backup for Mac-off days)
    r = step_commit_push()
    summary["steps"]["git_push"] = r

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"🎯 Daily Pipeline 완료 — {elapsed/60:.1f}분")
    log("=" * 60)

    # Save summary JSON
    sm_path = Path(f"/tmp/daily_pipeline_summary_{datetime.now().strftime('%Y-%m-%d')}.json")
    sm_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
    log(f"Summary: {sm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
