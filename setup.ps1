# AutoOutreach setup (Windows / PowerShell)
# Usage:  .\setup.ps1
$ErrorActionPreference = "Stop"

python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example -- open it and fill in your keys."
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python main.py --dry-run --limit 3"
