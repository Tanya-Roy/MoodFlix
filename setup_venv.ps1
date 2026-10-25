$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "`nVirtual environment created and dependencies installed."
Write-Host "Run the app with: streamlit run app.py"
