<#
    Watch Apparecchiato run -- three ways to see what the robot is doing.

        .\scripts\watch.ps1                 # live viewer, seed 0, whole episode
        .\scripts\watch.ps1 -Seed 3         # a different randomised table
        .\scripts\watch.ps1 -Record         # write out\seed0.mp4 instead of watching
        .\scripts\watch.ps1 -Scene          # just open the scene, no policy, drag it around
        .\scripts\watch.ps1 -Diagnose 2     # numbers for one step, with the grasp probe

    In the live viewer: left-drag orbits, right-drag pans, scroll zooms.
    Double-click a body to select it, Ctrl+drag to push it around mid-episode
    (a fast way to check the policy is not just replaying a fixed trajectory).
    Press Tab for the control panel, Space to pause. Close the window to stop.
#>
param(
    [int]$Seed = 0,
    [switch]$Record,
    [switch]$Scene,
    [int]$Diagnose = 0,
    [double]$Speed = 1.0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv found. Run .\scripts\dayone.ps1 first." }
New-Item -ItemType Directory -Force -Path (Join-Path $repo "out") | Out-Null

if ($Diagnose -gt 0) {
    & $py scripts\diagnose.py --seed $Seed --steps $Diagnose
    exit $LASTEXITCODE
}

if ($Scene) {
    # Static scene: no policy, just the world. Good for eyeballing geometry --
    # is the knob where you think, is anything spawned inside the furniture.
    & $py scripts\run_episode.py --seed $Seed --plan-only --dump-mjcf out\scene.xml
    & $py -m mujoco.viewer --mjcf="$repo\out\scene.xml"
    exit $LASTEXITCODE
}

if ($Record) {
    $out = "out\seed$Seed.mp4"
    & $py scripts\run_episode.py --seed $Seed --record $out
    if (Test-Path $out) {
        Write-Host "`nwrote $out" -ForegroundColor Green
        Start-Process (Resolve-Path $out)   # opens in your default video player
    }
    exit $LASTEXITCODE
}

& $py scripts\run_episode.py --seed $Seed --view --slow $Speed
