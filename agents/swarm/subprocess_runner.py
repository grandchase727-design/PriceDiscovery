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
# Bounded concurrency for claude -p subprocesses. Independent fan-out stages
# (phase1's 5 analysts, phase3's 2 synthesis, phase5's 3 horizons, pt_debate
# batches) can now run CONCURRENTLY — identical output, lower wall-clock.
# Capped (default 3) to stay under Max-plan concurrent-session limits — the
# historical logout happened at 6 simultaneous sessions. Tune via env var.
#   CLAUDE_MAX_CONCURRENCY=1  → fully serial (legacy safe behavior)
#   CLAUDE_MAX_CONCURRENCY=3  → default (3x speedup on fan-out, no quality change)
try:
    # Default 2 (parallel) — the earlier starvation/hang root causes are FIXED:
    #   (a) acquire now has a timeout (no indefinite block on a stalled slot),
    #   (b) action_selector output cut 80→32 (no mega-output stall),
    #   (c) tranche no longer uses claude (was the hang source).
    # Concurrency runs IDENTICAL calls concurrently → output unchanged (no quality
    # impact), only wall-clock drops. Tune: CLAUDE_MAX_CONCURRENCY=1 (serial) / 3.
    _CLAUDE_MAX_CONCURRENCY = max(1, int(os.environ.get("CLAUDE_MAX_CONCURRENCY", "2")))
except (TypeError, ValueError):
    _CLAUDE_MAX_CONCURRENCY = 1
# Max seconds to wait for a free slot before failing the call.
# 7쿼리 기준 shard_timeout=300s × retries=1 = 최대 300s per slot.
# 650 > 300s × 2 슬롯 — stall 감지 빠르게.
# Must comfortably EXCEED the longest single-call timeout (phase1 unified = 1200s)
# so a queued call never spuriously fails while two long websearch calls hold both slots.
_SEM_ACQUIRE_TIMEOUT = 1300
_CLAUDE_SEMAPHORE = _th.BoundedSemaphore(_CLAUDE_MAX_CONCURRENCY)   # caps concurrent subprocesses
_COUNTER_LOCK = _th.Lock()                                          # protects the call counter only
_CLAUDE_CALL_COUNT = 0

# Registry of `claude -p` subprocess PIDs THIS python process currently owns.
# The zombie reaper consults it so an in-flight sibling call is NEVER mistaken for
# a stale orphan and killed (root cause of the 2026-07-05 exit-143 regression: once
# WebSearch pushed a legit call past the reaper's age threshold, the reaper SIGTERM'd
# the still-working subprocess). Populated right after Popen, cleared in finally.
_ACTIVE_SUBPROCESS_PIDS: set = set()
_ACTIVE_PIDS_LOCK = _th.Lock()

# When set, run_claude refuses to spawn NEW subprocesses. shutdown_and_kill_all() sets it
# (briefly) at the end of a swarm run so a background thread that outlived run_swarm — e.g.
# a news_narrative future whose .result(timeout=…) already gave up — cannot keep spawning
# claude or block process exit. Auto-cleared after the sweep so later, legitimate callers
# (e.g. final_list_commentary) are unaffected.
_SHUTDOWN = _th.Event()


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
    """Sweep TRUE-ORPHAN 'claude' processes left behind by a previous, now-dead run.

    An orphan is a bare 'claude' process that has been reparented to init (ppid == 1)
    because the python process that spawned it died — these leak sessions/resources
    and are safe to kill. A process whose parent is still ALIVE (ppid != 1) is, by
    definition, an in-flight call owned by some live process, so it is NEVER reaped
    here (that job belongs to the owner's own communicate()-timeout / SIGKILL path).

    Additional guards (defense in depth): our own live children are excluded both by
    the ppid==1 rule (their parent — us — is alive) AND by the _ACTIVE_SUBPROCESS_PIDS
    registry, so a sibling call can never be mistaken for an orphan.

    Returns count of processes killed.
    """
    try:
        my_pid = os.getpid()
        my_ppid = os.getppid()
        with _ACTIVE_PIDS_LOCK:
            active = set(_ACTIVE_SUBPROCESS_PIDS)
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
        # Never touch: self, our parent, our registered live children, or any process
        # whose parent is still alive (ppid != 1 → not an orphan → someone's live call).
        if pid == my_pid or pid == my_ppid or pid in active:
            continue
        if ppid != 1:
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


def reset_shutdown() -> None:
    """Clear the shutdown flag. Called at the START of a swarm run so a stale flag from a
    prior run (or an interrupted sweep) never blocks legitimate calls."""
    _SHUTDOWN.clear()


def shutdown_and_kill_all(settle_sec: float = 3.0) -> int:
    """Terminate all `claude` subprocesses THIS process owns and briefly block new spawns.

    Call from a swarm run's finally: once run_swarm has returned, any claude still running is
    a leaked/background call (e.g. a timed-out news_narrative future). SIGKILL each owned
    process group so its blocking run_claude returns immediately (retries=0 → the thread ends),
    reap freshly-orphaned strays, hold the no-spawn flag for settle_sec so any thread mid-retry
    observes it and stops, then CLEAR the flag so later callers (final_list_commentary) work.

    Returns the number of owned subprocesses signalled.
    """
    _SHUTDOWN.set()
    n = 0

    def _kill_owned() -> int:
        with _ACTIVE_PIDS_LOCK:
            pids = list(_ACTIVE_SUBPROCESS_PIDS)
        k = 0
        for pid in pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                k += 1
            except (ProcessLookupError, PermissionError):
                pass
        try:
            reap_zombie_claude_processes(max_age_sec=0)
        except Exception:
            pass
        return k

    try:
        # Two-pass: a thread that passed the flag-check just before .set() may Popen + register
        # AFTER the first snapshot (race). The flag blocks any NEW spawn during settle_sec, so a
        # second kill after the window catches the straggler now that it is in the registry.
        n += _kill_owned()
        time.sleep(max(0.0, settle_sec))
        n += _kill_owned()
    finally:
        _SHUTDOWN.clear()
    return n


def run_claude(prompt: str, timeout: int = 180, retries: int = 2) -> dict:
    """Invoke `claude -p` with retries on ANY transient failure mode.

    HARDENED:
      - Bounded semaphore → ≤ _CLAUDE_MAX_CONCURRENCY (default 3) subprocesses at once
      - Pre-flight zombie reaper → removes stale bare-`claude` processes >5min old
        (path-launched siblings are spared, so concurrency is safe)
      - Process group kill on timeout → no orphan children survive
    """
    global _CLAUDE_CALL_COUNT
    if _SHUTDOWN.is_set():
        raise RuntimeError("subprocess runner shutting down — refusing new claude spawn")
    bin_path = find_claude()
    if not bin_path:
        raise RuntimeError("claude CLI not on PATH")

    with _COUNTER_LOCK:
        _CLAUDE_CALL_COUNT += 1
        call_n = _CLAUDE_CALL_COUNT

    if call_n % 5 == 1:    # every 5th call
        n_killed = reap_zombie_claude_processes(max_age_sec=300)
        if n_killed:
            time.sleep(2)

    last = ""
    for attempt in range(retries + 1):
        if _SHUTDOWN.is_set():
            raise RuntimeError("subprocess runner shutting down — aborting claude retry")
        # Bounded concurrency with ACQUIRE-TIMEOUT — if a sibling subprocess stalls
        # holding a slot, fail gracefully (sweep zombies) instead of hanging forever.
        if not _CLAUDE_SEMAPHORE.acquire(timeout=_SEM_ACQUIRE_TIMEOUT):
            reap_zombie_claude_processes(max_age_sec=120)
            last = "semaphore acquire timeout (slot starvation — swept zombies)"
            if attempt < retries:
                continue
            raise RuntimeError(last)
        _child_pid = None
        try:
            proc = None
            _prompt_file = None
            try:
                # 큰 프롬프트는 stdin pipe buffer deadlock 방지를 위해 임시파일로 전달
                import tempfile
                _prompt_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                )
                _prompt_file.write(prompt)
                _prompt_file.flush()
                _prompt_file.close()
                proc = subprocess.Popen(
                    [bin_path, "-p",
                     "--allowedTools", "WebSearch", "WebFetch"],
                    stdin=open(_prompt_file.name, "r", encoding="utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                # Register this live child so the reaper never mistakes it for an orphan.
                _child_pid = proc.pid
                with _ACTIVE_PIDS_LOCK:
                    _ACTIVE_SUBPROCESS_PIDS.add(_child_pid)
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass  # SIGKILL sent; OS will reap the process eventually
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
        finally:
            _CLAUDE_SEMAPHORE.release()
            if _child_pid is not None:
                with _ACTIVE_PIDS_LOCK:
                    _ACTIVE_SUBPROCESS_PIDS.discard(_child_pid)
            if _prompt_file is not None:
                try: os.unlink(_prompt_file.name)
                except Exception: pass

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
