#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────
# cleanup_claude_sessions.sh — Manual stale Claude process killer
# ─────────────────────────────────────────────────────────────────────
# Run this when you see "Not logged in" errors or session disconnects.
#
# Max plan allows ~2 concurrent CLI sessions. Stale processes from
# previous swarm runs (subprocess timeouts that left zombies) + old
# VSCode sessions accumulate and eat session quota.
#
# Strategy:
#   1) List all claude processes
#   2) Identify the CURRENT VSCode session (spare it)
#   3) Kill everything else that's been running >5min
# ─────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "Claude session cleanup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. List all claude processes
echo "Currently running claude processes:"
ps -axo pid,etime,command | grep -E "claude" | grep -v grep | grep -v cleanup_claude
echo ""

# 2. Identify current VSCode session (most recently started full-path claude)
CURRENT=$(ps -axo pid,etime,command | grep "anthropic.claude-code" | grep "vscode" | grep -v grep | sort -k2 | head -1 | awk '{print $1}')

if [ -n "$CURRENT" ]; then
  echo "Current VSCode session (PROTECTED): PID $CURRENT"
else
  echo "No current VSCode session detected"
fi
echo ""

# 3. Kill stale VSCode sessions older than 12 hours
echo "Killing VSCode sessions older than 12 hours..."
ps -axo pid,etime,command | grep "anthropic.claude-code" | grep -v grep | while read pid etime cmd; do
  if [ "$pid" != "$CURRENT" ]; then
    # etime format: DD-HH:MM:SS or HH:MM:SS or MM:SS
    if [[ "$etime" == *-* ]]; then
      # Has days → definitely > 12h
      echo "  Killing stale session PID $pid (elapsed $etime)"
      kill $pid 2>/dev/null
    elif [[ "$etime" == *:*:* ]]; then
      # HH:MM:SS — check if hours >= 12
      hours=$(echo "$etime" | cut -d: -f1)
      if [ "$hours" -ge 12 ] 2>/dev/null; then
        echo "  Killing stale session PID $pid (elapsed $etime)"
        kill $pid 2>/dev/null
      fi
    fi
  fi
done
echo ""

# 4. Kill bare 'claude' subprocesses (no path = orphans from swarm runs)
#    ONLY if elapsed > 5 minutes — VSCode/CLI use short-lived bare `claude` calls
#    for auth status checks; killing those would cause more disruption.
echo "Killing orphaned bare 'claude' subprocesses (only if older than 5 min)..."
ps -axo pid,etime,command | grep -E "^[ ]*[0-9]+[ ]+[0-9:-]+[ ]+claude[ ]*$" | grep -v grep | while read pid etime cmd; do
  # Calculate elapsed seconds
  if [[ "$etime" == *-* ]]; then
    # Has days → definitely > 5 min
    elapsed_sec=999999
  elif [[ "$etime" == *:*:* ]]; then
    # HH:MM:SS
    h=$(echo "$etime" | cut -d: -f1)
    m=$(echo "$etime" | cut -d: -f2)
    s=$(echo "$etime" | cut -d: -f3)
    elapsed_sec=$((h*3600 + m*60 + s))
  else
    # MM:SS
    m=$(echo "$etime" | cut -d: -f1)
    s=$(echo "$etime" | cut -d: -f2)
    elapsed_sec=$((m*60 + s))
  fi
  if [ "$elapsed_sec" -ge 300 ] 2>/dev/null; then
    echo "  Killing orphan PID $pid (elapsed $etime)"
    kill $pid 2>/dev/null
  else
    echo "  Skipping PID $pid (elapsed $etime — too recent, may be auth helper)"
  fi
done
echo ""

sleep 1

echo "═══════════════════════════════════════════════════════════════"
echo "Remaining claude processes:"
echo "═══════════════════════════════════════════════════════════════"
ps -axo pid,etime,command | grep -E "claude" | grep -v grep | grep -v cleanup_claude
echo ""
echo "✓ Done. Run again if you still see logout errors."
