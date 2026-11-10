# Step 11 / 12 on the analysis PC (all FC traces available).
# Run from the repository folder in the "EIA PowerShell" terminal:
#     git pull
#     .\tools\pc_step11_12.ps1
# Writes the changelog entries of N3/N5/N6/N14 and N8, measures the options that are not
# adopted (N7 FC grid/peak, I2 area calibration, FC start/stop), makes the method studies
# and the study figures. Takes about 20-30 minutes. Paste the summary at the end.

$ErrorActionPreference = "Continue"
$study = "PRL_AlNb"
$log = Join-Path $env:EIA_STUDIES_ROOT "$study\pc_step11_12.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
"pc_step11_12 started $(Get-Date)" | Out-File $log

function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title" -ForegroundColor Cyan
    "`n=== $title" | Out-File $log -Append
    & $cmd 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Write-Host "    (exit code $LASTEXITCODE)" -ForegroundColor Yellow }
}

Step "install (PySide6 pin, new entry points)" { pip install -q -e . }
Step "pipeline with the current defaults" { eiakit run }

Step "N3/N5/N6/N14: changelog note" {
    eiakit note $study "N3/N5/N6/N14: checked, no paper number changes" --text ("N3 the fast/slow labels of runs.toml agree with the arrival times in every z of every run (audit check). N5 the reconstruction takes the ion mass from the constants profile. N6 all fit files declare '# Signal: per-ion' and the header is read. N14 (fit stored under another voltage: 20250611 p30/p35, 20250606 p250/p250.01) is removed completely by N1.")
}

Step "N8: m/z overlaps (adopted default nearest_line)" {
    eiakit compare $study calibration.overlaps keep_both nearest_line --changelog "N8: one row per peak in overlapping m/z lines" --text ("untitled28 fits every (m, z) of the mass list in a +/-2 % window, so close lines (O3+/Al5+, C3+/Al7+, O+/Nb6+, C+/Nb8+, O2+/Nb12+, C3+/O4+) could be fitted on the same peak and counted twice in the summed EIA yield. Now every row is assigned to its nearest predicted line (line scale from the target rows of the file, TOF ~ sqrt(m/z)); one row per peak (unresolvable lines: target species, then lower z). Rows dropped: 20250513 18, 20250602 1, 20250606 1, 20250623 8, 20250611 and 20250617 0; 2026_1103: 89 (CFY x 1.078).")
}

Step "N7: FC on the native grid, smoothed peak (measured, not adopted)" {
    eiakit snapshot $study --label N7-legacy-grid
    eiakit --set calibration.fc_grid_points=native --set calibration.fc_peak=smoothed run
    eiakit --set calibration.fc_grid_points=native --set calibration.fc_peak=smoothed snapshot $study --label N7-native-smoothed
    eiakit diff $study N7-legacy-grid N7-native-smoothed
    eiakit --set calibration.fc_grid_points=native run
    eiakit --set calibration.fc_grid_points=native snapshot $study --label N7-native-max
    eiakit diff $study N7-legacy-grid N7-native-max
}

Step "I2: area calibration (measured, not adopted)" {
    eiakit --set calibration.method=area run
    eiakit --set calibration.method=area snapshot $study --label I2-area
    eiakit diff $study N7-legacy-grid I2-area
}

Step "FC start / both (measured, not adopted)" {
    eiakit --set calibration.fc_source=both run
    eiakit --set calibration.fc_source=both snapshot $study --label FC-both
    eiakit diff $study N7-legacy-grid FC-both
}

Step "back to the defaults" { eiakit run }

Step "method studies (Step 12)" { eiakit methods $study --boot 30 }
Step "study figures" { eiakit study $study }

Step "summary" {
    eiakit status
    $m = Join-Path $env:EIA_STUDIES_ROOT "$study\methods"
    Write-Host "`n--- calibration (I2) ---"; Get-Content (Join-Path $m "calibration.csv")
    Write-Host "`n--- FC (start / stop / both, peak time) ---"; Get-Content (Join-Path $m "fc.csv")
    Write-Host "`nchangelog: $(Join-Path $env:EIA_STUDIES_ROOT "$study\changelog.md")"
    Write-Host "methods:   $(Join-Path $m 'methods.html')"
    Write-Host "full log:  $log"
}
