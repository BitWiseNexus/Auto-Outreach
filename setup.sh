#!/usr/bin/env bash
# AutoOutreach setup (macOS / Linux / Git Bash)
# Usage:  bash setup.sh
set -euo pipefail

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example -- open it and fill in your keys."
fi

echo
echo "Done. Next:"
echo "  source venv/bin/activate"
echo "  python main.py --dry-run --limit 3"
