# -*- coding: utf-8 -*-
"""agents/swarm/subprocess_runner.py — Hardened `claude -p` subprocess wrapper.

Extracted from agents/market_leaders_swarm.py during Option B refactor.

================================================================================
PURPOSE
================================================================================

Session-safety guards for invoking `claude -p` from within Python:

  - Global lock → only ONE concurrent subprocess at a time (Max plan constraint)
  - Pre-flight zombie reaper → sweep stale `claude` processes every 5 calls
  - Process group kill on timeout → no orphan children survive
  - Retries with exponential backoff (60/120/180s for login, 8/16/32s for transient)
  - JSON extraction from claude output (markdown fenced or raw)

================================================================================
PUBLIC API
================================================================================

run_claude(prompt, timeout=180, retries=2) -> dict
  Main subprocess executor. Returns parsed JSON dict from claude output.
  Raises RuntimeError on definitive failure.

reap_zombie_claude_processes(max_age_sec=300) -> int
  Manually trigger zombie cleanup. Returns count killed.

find_claude() -> Optional[str]
  Locate `claude` binary on PATH.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading as _th
import time
from typing import Optional


# ─── Global state ──────────────────────────────────────────────────
_CLAUDE_SUBPROCESS_LOCK = _th.Lock()
_CLAUDE_CALL_COUNT = 0


# ─── Helpers ──────────────────────────────────────────────────────
def find_claude() -> Optional[str]:
    """Locate `claude` binary on PATH."""
    return shutil.which("claude")


def extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from text (markdown fenced or raw)."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


def reap_zombie_claude_processes(max_age_sec: int = 300) -> int:
    """Sweep stale 'claude' processes that are NOT this Python process's parent chain.

    Targets bare 'claude' command (no path) processes started by previous swarm
    runs that timed out and lost child handles. Spares the user's VSCode session.
    Returns count of processes killed.
    """
    try:
        my_ppid = os.getppid()
        out = subprocess.run(
            ["ps", "-axo", "pid,ppid,etime,command"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return 0

    killed = 0
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0]); ppid = int(parts[1])
        except ValueError:
            continue
        cmd = parts[3]
        if pid == os.getpid() or pid == my_ppid:
            continue
        if cmd.strip() == "claude" or cmd.strip().startswith("claude "):
            etime = parts[2]
            if "-" in etime or ":" in etime:
                try:
                    if "-" in etime:
                        days, rest = etime.split("-", 1)
                        h, m, s = rest.split(":")
                        secs = int(days)*86400 + int(h)*3600 + int(m)*60 + int(s)
                    else:
                        parts_t = etime.split(":")
                        if len(parts_t) == 2:
                            secs = int(parts_t[0])*60 + int(parts_t[1])
                        else:
                            secs = int(parts_t[0])*3600 + int(parts_t[1])*60 + int(parts_t[2])
                    if secs > max_age_sec:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            killed += 1
                        except (ProcessLookupError, PermissionError):
                            pass
                except (ValueError, IndexError):
                    pass
    return killed


def run_claude(prompt: str, timeout: int = 180, retries: int = 2) -> dict:
    """Invoke `claude -p` with retries on ANY transient failure mode.

    HARDENED:
      - Global lock → guarantees ONE concurrent subprocess at a time
      - Pre-flight zombie reaper → removes stale `claude` processes >5min old
      - Process group kill on timeout → no orphan children survive
    """
    global _CLAUDE_CALL_COUNT
    bin_path = find_claude()
    if not bin_path:
        raise RuntimeError("claude CLI not on PATH")

    with _CLAUDE_SUBPROCESS_LOCK:
        _CLAUDE_CALL_COUNT += 1
        call_n = _CLAUDE_CALL_COUNT

    if call_n % 5 == 1:    # every 5th call
        n_killed = reap_zombie_claude_processes(max_age_sec=300)
        if n_killed:
            time.sleep(2)

    last = ""
    for attempt in range(retries + 1):
        with _CLAUDE_SUBPROCESS_LOCK:
            proc = None
            try:
                proc = subprocess.Popen(
                    [bin_path, "-p",
                     "--allowedTools", "WebSearch", "WebFetch"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    proc.wait(timeout=5)
                    last = f"subprocess timeout after {timeout}s (killed proc group)"
                    proc = None
                    if attempt < retries:
                        time.sleep(8 * (2 ** attempt))
                        continue
                    raise RuntimeError(last)
            except Exception:
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                raise

        # Normalize Popen output
        returncode = proc.returncode if proc else 1
        out = (stdout if proc else "").strip()
        stderr_text = (stderr if proc else "").strip()

        # Detect transient API errors in stdout
        transient_stdout = (
            "overloaded_error" in out
            or "529" in out[:200]
            or "rate_limit" in out.lower()
            or "API Error:" in out
            or "Not logged in" in out
            or "Please run /login" in out
            or "session expired" in out.lower()
            or "concurrent session" in out.lower()
        )
        login_error = (
            "Not logged in" in out
            or "Please run /login" in out
            or "session expired" in out.lower()
            or "concurrent session" in out.lower()
        )
        login_backoff = lambda att: 60 + 60 * att   # 60s, 120s, 180s

        if returncode != 0:
            last = f"claude exit {returncode}: stdout={out[:150]} stderr={stderr_text[:150]}"
            if attempt < retries:
                time.sleep(login_backoff(attempt) if login_error else 8 * (2 ** attempt))
                continue
            raise RuntimeError(last)

        if transient_stdout:
            last = f"anthropic api error: {out[:300]}"
            if attempt < retries:
                time.sleep(login_backoff(attempt) if login_error else 8 * (2 ** attempt))
                continue
            raise RuntimeError(last)

        parsed = extract_json(out)
        if not parsed:
            last = f"no JSON in output: {out[:300]}"
            if attempt < retries:
                time.sleep(4 * (2 ** attempt))
                continue
            raise RuntimeError(last)

        return parsed

    raise RuntimeError(last or "max retries exceeded")
