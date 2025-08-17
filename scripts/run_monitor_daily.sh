#!/usr/bin/env bash
set -euo pipefail

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present (OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.)
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo ".venv not found at $PROJECT_ROOT/.venv. Create it and install requirements.txt." >&2
  exit 1
fi

# Parameters
DAYS="${DAYS:-45}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Logs
LOG_DIR="${PROJECT_ROOT}/monitor_output/logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# Run monitor (Telegram will auto-send if TELEGRAM_* envs are set)
python monitor.py run --days "$DAYS" --telegram ${EXTRA_ARGS} >"${LOG_DIR}/run_${TS}.log" 2>&1
