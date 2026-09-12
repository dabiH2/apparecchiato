<#
    Apparecchiato -- one-command day-one setup (Windows PowerShell).

      1. finds the repo root (the folder containing pyproject.toml)
      2. copies it out of OneDrive to C:\Users\<you>\dev\apparecchiato
      3. creates a virtual environment there
      4. installs the simulation dependencies (not torch/OpenVINO -- those are day 3)
      5. runs the test suite
      6. runs the diagnostic on the drawer step

    Why it moves the project: OneDrive syncs every file it sees. A venv is tens of
    thousands of small files, git wants its own .git directory, and OneDrive can
    quietly serve a stale copy of a file you just edited -- which is maddening to
    debug. Working in C:\Users\<you>\dev keeps OneDrive as the backup.

    Usage -- works from anywhere, including a detached copy of this script:
        powershell -ExecutionPolicy Bypass -File .\scripts\dayone.ps1

    Set up in place without moving:
        powershell -ExecutionPolicy Bypass -File .\scripts\dayone.ps1 -InPlace

    Point it at a specific repo:
        powershell -ExecutionPolicy Bypass -File .\dayone.ps1 -Source "C:\path\to\apparecchiato"
#>
param(
    [switch]$InPlace,
    [string]$Source,
    [string]$Dest = "$env:USERPROFILE\dev\apparecchiato"
)

$ErrorActionPreference = "Stop"

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

function Find-RepoRoot([string]$start) {
    # The repo root is the directory holding pyproject.toml. Walk up first, then
    # look downward -- so running this from a detached copy of the script (say a
    # downloads folder beside the repo) still finds it instead of silently
    # copying the wrong tree, which is exactly what went wrong the first time.
    $d = $start
    while ($d) {
        if (Test-Path (Join-Path $d 'pyproject.toml')) { return $d }
        $parent = Split-Path -Parent $d
        if ($parent -eq $d) { break }
        $d = $parent
    }
    foreach ($base in @($start, (Split-Path -Parent $start))) {
        if (-not $base -or -not (Test-Path $base)) { continue }
        $hit = Get-ChildItem -Path $base -Filter pyproject.toml -Recurse -Depth 2 `
                             -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.DirectoryName }
    }
    return $null
}

$src = if ($Source) { (Resolve-Path $Source).Path } else { Find-RepoRoot $PSScriptRoot }
if (-not $src) {
    throw "Could not find the repo root (no pyproject.toml near $PSScriptRoot). Pass -Source <path>."
}
Write-Host "Repo root: $src" -ForegroundColor DarkGray

if ($InPlace) {
    $work = $src
    Step 1 "Working in place: $work"
} else {
    Step 1 "Copying project to $Dest (excluding .venv, __pycache__, out, .git)"
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    robocopy $src $Dest /E /XD .venv __pycache__ out .git .pytest_cache /XF *.pyc /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }
    $work = $Dest
    Write-Host "    done. The OneDrive copy stays as your backup." -ForegroundColor Green
}

Set-Location $work

# Verify the copy really is the repo before going further. Without this check a
# mis-rooted copy sails on and only fails four steps later with a confusing
# "can't open file scripts\verify_env.py".
$required = @('pyproject.toml', 'scripts\verify_env.py', 'scripts\diagnose.py', 'apparecchiato\kinematics.py')
$missing = $required | Where-Object { -not (Test-Path (Join-Path $work $_)) }
if ($missing) {
    throw "The copy at $work is missing: $($missing -join ', '). Wrong source folder?"
}

$py = Join-Path $work ".venv\Scripts\python.exe"

Step 2 "Creating virtual environment"
if (-not (Test-Path $py)) { python -m venv .venv }
& $py -m pip install --upgrade pip --quiet
& $py --version

Step 3 "Installing simulation dependencies (a few minutes on first run)"
& $py -m pip install --quiet mujoco numpy imageio imageio-ffmpeg pillow pytest
& $py -c "import mujoco, numpy; print('    mujoco', mujoco.__version__, '| numpy', numpy.__version__)"

Step 4 "Environment check"
& $py scripts\verify_env.py

Step 5 "Test suite (expect 163 passed)"
& $py -m pytest -q

Step 6 "Diagnosing step 1 -- the drawer"
& $py scripts\diagnose.py --seed 0 --steps 1

Write-Host @"

--------------------------------------------------------------------
Next:
  Run one full episode          .\.venv\Scripts\python.exe scripts\run_episode.py --seed 0
  Diagnose a specific step      .\.venv\Scripts\python.exe scripts\diagnose.py --seed 0 --steps 2
  Walk every step, don't stop   .\.venv\Scripts\python.exe scripts\diagnose.py --seed 0 --keep-going
  Look at the scene by hand     .\.venv\Scripts\python.exe scripts\run_episode.py --seed 0 --plan-only --dump-mjcf out\scene.xml
                                .\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=out\scene.xml

Read the numbers, not the vibes: site->target distance says whether the arm
got there, joint error says whether it is sagging, contacts say what is in
the way. Working directory is now:
  $work
--------------------------------------------------------------------
"@ -ForegroundColor Yellow
