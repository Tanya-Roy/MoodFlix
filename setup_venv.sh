#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Virtual environment created and dependencies installed."
echo "Run the app with: streamlit run app.py"
