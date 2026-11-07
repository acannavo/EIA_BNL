# ==========================================================================
#  env.ps1 -- environment for EIA_BNL (PowerShell, Windows)
#
#  Dot-source it from the repository folder (note the leading dot + space):
#
#      . .\env.ps1              set variables, activate .venv
#      . .\env.ps1 -Install     first time / after a dependency change:
#                               create .venv and install eiakit + dev tools
#      . .\env.ps1 -Install -Python "C:\path\to\python.exe"
#                               use a specific Python (>= 3.11) to create .venv
#
#  What it does
#    * sets EIA_DATA_ROOT, EIA_PROCESSED_ROOT, EIA_STUDIES_ROOT (+ MPLBACKEND,
#      PYTHONUTF8) for this terminal session;
#    * writes the same variables to .env, which VS Code uses for Run/Debug and
#      the test explorer (python.envFile);
#    * creates processed\ and studies\ if missing (data\ is never written);
#    * activates the virtual environment .venv in the repository.
#
#  Local overrides: create env.local.ps1 (git-ignored) next to this file, e.g.
#      $EIA_BASE = 'D:\EIA\NewEIASoftware'
#
#  If PowerShell refuses to run scripts, once:
#      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ==========================================================================

param(
    [switch]$Install,
    [string]$Python = ''
)

$RepoRoot = $PSScriptRoot

# ---- base folder of data / processed / studies ---------------------------
$EIA_BASE = 'C:\Users\acannavo\OneDrive - Brookhaven National Laboratory\NewEIASoftware'
$LocalFile = Join-Path $RepoRoot 'env.local.ps1'
if (Test-Path $LocalFile) { . $LocalFile }

$env:EIA_DATA_ROOT      = Join-Path $EIA_BASE 'data'
$env:EIA_PROCESSED_ROOT = Join-Path $EIA_BASE 'processed'
$env:EIA_STUDIES_ROOT   = Join-Path $EIA_BASE 'studies'
$env:MPLBACKEND         = 'QtAgg'      # interactive matplotlib windows (peak/baseline/CBS pickers)
$env:PYTHONUTF8         = '1'          # UTF-8 everywhere (µ, ², ... in file headers)

if (-not (Test-Path $env:EIA_DATA_ROOT)) {
    Write-Warning "EIA_DATA_ROOT does not exist: $env:EIA_DATA_ROOT"
}
foreach ($d in @($env:EIA_PROCESSED_ROOT, $env:EIA_STUDIES_ROOT)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# ---- .env for VS Code (Run/Debug, tests) ---------------------------------
@(
    "# written by env.ps1 -- do not edit, edit env.ps1 / env.local.ps1 instead"
    "EIA_DATA_ROOT=$env:EIA_DATA_ROOT"
    "EIA_PROCESSED_ROOT=$env:EIA_PROCESSED_ROOT"
    "EIA_STUDIES_ROOT=$env:EIA_STUDIES_ROOT"
    "MPLBACKEND=$env:MPLBACKEND"
    "PYTHONUTF8=$env:PYTHONUTF8"
) | Set-Content -Path (Join-Path $RepoRoot '.env') -Encoding UTF8

# ---- virtual environment ---------------------------------------------------
$Venv   = Join-Path $RepoRoot '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'

if ($Install -and -not (Test-Path $VenvPy)) {
    if ($Python) {
        $cmd = @($Python)
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $cmd = @('py', '-3.12')          # Python launcher; falls back below if 3.12 is absent
        & py -3.12 --version 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { $cmd = @('py', '-3') }
    } else {
        $cmd = @('python')
    }
    Write-Host "Creating virtual environment with: $($cmd -join ' ')"
    if ($cmd.Count -gt 1) { & $cmd[0] $cmd[1..($cmd.Count - 1)] -m venv $Venv }
    else                  { & $cmd[0] -m venv $Venv }
}

if (Test-Path $VenvPy) {
    . (Join-Path $Venv 'Scripts\Activate.ps1')
    if ($Install) {
        python -m pip install --upgrade pip
        python -m pip install -e "$RepoRoot[dev]"
    }
} else {
    Write-Warning ".venv not found. Run  . .\env.ps1 -Install  once to create it."
}

Write-Host ""
Write-Host "EIA_BNL environment" -ForegroundColor Cyan
Write-Host "  data      : $env:EIA_DATA_ROOT"
Write-Host "  processed : $env:EIA_PROCESSED_ROOT"
Write-Host "  studies   : $env:EIA_STUDIES_ROOT"
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "  python    : $((Get-Command python).Source)"
}
Write-Host "  check with: eiakit info"
