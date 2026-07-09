#!/usr/bin/env bash
set -euo pipefail

# Portable MCP service manager.
# Defaults assume this script is placed at the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "$PROJECT_ROOT")}" 
MCP_DIR="${MCP_DIR:-$PROJECT_ROOT/FileSystem-MCP-for-GPT-main/FileSystem-MCP-for-GPT-main}"
LOG_FILE="${LOG_FILE:-$MCP_DIR/mcp.log}"
PID_FILE="${PID_FILE:-$MCP_DIR/mcp.pid}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
  if is_running; then
    echo "MCP already running: PID $(cat "$PID_FILE")"
    exit 0
  fi
  if [ ! -f "$MCP_DIR/fileSystemMCP.py" ]; then
    echo "fileSystemMCP.py not found under MCP_DIR=$MCP_DIR" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  cd "$MCP_DIR"
  nohup "$PYTHON_BIN" fileSystemMCP.py "$WORKSPACE_ROOT" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  if is_running; then
    echo "MCP started: PID $(cat "$PID_FILE")"
    echo "Workspace root: $WORKSPACE_ROOT"
    echo "MCP dir: $MCP_DIR"
    echo "Log: $LOG_FILE"
  else
    echo "MCP failed to start; see $LOG_FILE" >&2
    exit 1
  fi
}

stop() {
  if is_running; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
  echo "MCP stopped"
}

status() {
  if is_running; then
    echo "MCP running: PID $(cat "$PID_FILE")"
    echo "Workspace root: $WORKSPACE_ROOT"
    echo "MCP dir: $MCP_DIR"
  else
    echo "MCP not running"
    exit 1
  fi
}

logs() {
  touch "$LOG_FILE"
  tail -n "${2:-100}" -f "$LOG_FILE"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs "$@" ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs [lines]}"; exit 2 ;;
esac
