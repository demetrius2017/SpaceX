#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
python - <<'PY'
from telegram_bot import daily_broadcast

if __name__ == '__main__':
    daily_broadcast()
PY
