# Commit everything and push to GitHub. Safe to paste repeatedly.
#
#   .\scripts\push.ps1                  # commit + push with a default message
#   .\scripts\push.ps1 -m "fixed pour"  # commit + push with your own message
#   .\scripts\push.ps1 -Public          # ALSO flip the repo to public (for submission)
param(
    [string]$m = "",
    [switch]$Public
)
# "Continue", not "Stop": git writes ordinary progress and CRLF notices to
# stderr, and PowerShell turns any stderr from a native command into a
# terminating error under "Stop". Exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"

function Invoke-Checked {
    param([string]$What, [scriptblock]$Do)
    & $Do 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

# Work from the repo root regardless of where this was launched from.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "repo: $root" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $root ".git"))) {
    throw "no .git here -- run this from inside the apparecchiato repo"
}

# Never push something that does not pass its own tests.
Write-Host "running tests..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed -- not pushing" }

if (-not $m) { $m = "wip: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }

Invoke-Checked "git add" { git add -A }

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "nothing to commit" -ForegroundColor Yellow
} else {
    Invoke-Checked "git commit" {
        git -c user.name="Gabriele Desimini" -c user.email="dabi.ai.eng@gmail.com" `
            commit -q -m $m
    }
    Write-Host "committed: $m" -ForegroundColor Green
}

Invoke-Checked "git push" { git push -u origin main }
Write-Host "pushed to $(git remote get-url origin)" -ForegroundColor Green

if ($Public) {
    Invoke-Checked "gh repo edit" {
        gh repo edit --visibility public --accept-visibility-change-consequences
    }
    Write-Host "repo is now PUBLIC" -ForegroundColor Yellow
}
