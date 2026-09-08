#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo 'Create .venv and install requirements.txt first; see README.md.' >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT_ROOT/.venv/bin/python" -m ghost_in_the_deck.app "$@"
