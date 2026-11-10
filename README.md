# EIA_BNL

Analysis software for Energy Ion Analyzer (EIA) measurements of the charge-state
distribution of laser-produced plasmas: SEM peak extraction, Faraday-cup
calibration, multi-component CBS fitting, and cross-run comparisons
(velocities, ion yields, average charge state z̄(E), velocity collapse).

The package is `eiakit`. It replaces a set of stand-alone scripts. It is
developed in three phases: first rebuild the analysis on the existing (legacy)
results, then check that new and old results agree, then apply the known
corrections one at a time.

## Repository content

Code only. Raw data, processed results and figures are **not** in the repository
(see `.gitignore`); their location is set by environment variables.

```
src/eiakit/        package
db/                database of the experimental conditions (TOML, version-controlled)
tests/             pytest tests
env.ps1            environment script (PowerShell)
pyproject.toml     package metadata and dependencies
.vscode/           VS Code settings, run/debug configurations, recommended extensions
```

## Folders outside the repository

| Variable | Default | Content |
|---|---|---|
| `EIA_DATA_ROOT` | `...\NewEIASoftware\data` | raw data and legacy results, **read-only** |
| `EIA_PROCESSED_ROOT` | `...\NewEIASoftware\processed` | everything the software writes per run |
| `EIA_STUDIES_ROOT` | `...\NewEIASoftware\studies` | cross-run comparisons |
| `EIA_FITS_ROOT` (optional) | `...\NewEIASoftware\fits` (next to `processed`) | fits made in the window: `<run>\cbs_fit.json`, `<run>\peaks\p<U>.txt`, `<run>\fc_baseline.json`; never deleted by the pipeline, back them up |

Data layout, one folder per run (`20250513`, `2026_1103 Nb analysis`, ...):

```
<run>\0_FocalPosition | 0_FocalSearch
<run>\1_FC_Start          Faraday-cup traces before the scan
<run>\2_SEM               raw SEM spectra, one CSV per deflector voltage and shot
<run>\2_SEM\Output         legacy peak table per deflector voltage
<run>\2_SEM\Output_detail
<run>\3_FC_Stop           Faraday-cup traces after the scan
<run>\EIA-Results         legacy checkpoint and CBS fit files
```

Processed layout, one folder per run and stage, each with a `manifest.json` (what the
stage read - db values, input files with sha256, code, upstream stage - and wrote):

```
processed\<run>\00_legacy_import   audit of the legacy results
processed\<run>\10_peaks           untitled28 replay
processed\<run>\20_calibration     untitled27: spectra.h5 (calibrated spectra), calibration.json
processed\<run>\30_cbs             CBS stage on the current calibration: cbs_areas.csv, replay_*
processed\<run>\40_report          report.html: spectra, CBS decomposition per z, zbar, tables
```

Study outputs (`eiakit study NAME`): `studies\<NAME>\<part>\` with PDF + PNG figures (PRL
style), one interactive HTML page (zoom, click a legend entry to hide it, double-click to
show only it), CSV tables and a manifest of the run outputs used.

Never edit these by hand: `eiakit status` notices it and marks the stage stale.

Folders whose name is a canonical name plus `~`, `_`, `_0`, `_b`, ... (for example
`Output~`, `EIA-Results_`) or that contain `wrong` are temporary copies with wrong
results and are ignored. Other folders (`Miscellaneous`, `LowStateCharge`,
`4_MultiShotSameSpot`, ...) are listed but not used.

### The eiakit window (`eiakit gui`)

| tab | what it does |
|---|---|
| Runs | every run with the state of its stages (hover a cell: why it is stale); Run selected / all, open the run report, start the CBS fit store |
| Studies | figures of a study, method studies, snapshots and diffs; double-click a page to open it |
| Settings | the switches of `db\pipeline.toml`; Compare one setting (`eiakit compare`) |
| CBS fitter | per charge state on the current calibration: click the peak, Fit (kT scan), kT from FWHM, add / remove labelled components (fast, slow, ...), windows; Save writes `fits\<run>\cbs_fit.json` (old version kept), then Run updates the cbs and report stages |
| Peaks | the raw SEM shots of one voltage: refit every line at the file voltage, move a window (click), drop a line; Save replaces that voltage's Output rows in the calibration |
| FC baseline | the FC average and its pre-trigger region; pick the pedestal per run (used with `fc_baseline = "picked"`) |

Long actions run `eiakit ...` in the background; the output appears in the log at the
bottom. Without a fit store the legacy `CBS_fit_results` file is used, so nothing changes
until something is saved (`eiakit fit-import` copies the legacy fits into the store: all
paper numbers identical).

### Corrections to the legacy analysis (Step 11)

Each correction is a switch in `db\pipeline.toml`, off by default (= legacy numbers), except the adopted ones: I1 (`codata`), N1 (`last_per_mz`), N8 (`nearest_line`). To see
what one moves:

```powershell
eiakit compare PRL_AlNb calibration.output_duplicates all last_per_mz --changelog "N1: ..."
```

`compare` runs the pipeline and snapshots the study with the first value, then with the
second (no file to edit), prints the diff, appends the changelog entry and finally reruns
with `db\pipeline.toml` as it is. Any setting can also be overridden for one command:
`eiakit --set corrections.mass_constants=legacy run`. By hand: `eiakit snapshot`, `eiakit diff`.

| switch | values | effect |
|---|---|---|
| `[corrections] mass_constants` | `legacy` / `codata` | I1: CODATA amu, ion masses and e in CBS and analysis; CBS kT, V0 rescaled, curves unchanged |
| `[calibration] output_duplicates` | `all` / `last_per_mz` / `last_save` | N1: rows of repeated saves in `Output\p<U>.txt`: all (legacy), the newest fit of each charge state, or only the last save |
| `[calibration] overlaps` | `keep_both` / `nearest_line` | N8: a peak fitted as two (m, z) (e.g. O⁺ on Nb⁶⁺, C³⁺/O⁴⁺) counted twice in the summed yield, or each row assigned to its nearest predicted line |
| `[calibration] fc_grid_points`, `fc_peak` | `1000` / `native`; `max` / `smoothed` | N7: FC average on 1000 points (40 ns: peak −1.4 % on 20250611) or the scope sampling; peak = max, or max after a moving average of `fc_smooth_s` |
| `[calibration] method` | `peak` / `area` | I2: CFY from the peaks (legacy) or the areas over the common time range (`eiakit methods`) |
| `[analysis] zbar_fill`, `zbar_mask`, `zbar_min_frac`, `zbar_clip_negative` | see `db\pipeline.toml` | I4 / I5: zero fill outside the measured ranges, noise floor (`eiakit methods`) |
| `[calibration] fc_missing` | `error` / `legacy_cfy` / `legacy_fc_peak` | run without FC traces: stop, use the legacy CFY, or the legacy FC peak (CFY then follows EIA-side corrections exactly) |

## Database (`db/`)

All experiment-dependent values live here; the code contains no paths and no
physical constants.

| File | Content |
|---|---|
| `constants.toml` | physical constants by profile: `corrected` (CODATA 2018, default) and the `legacy_*` profiles that reproduce exactly what each legacy script used |
| `species.toml` | ion species: mass [amu], mass labels used in the legacy files |
| `instruments.toml` | EIA (flight path, analyzer radii, time zero), Faraday cup (distance, aperture, load), acquisition file formats |
| `runs.toml` | one entry per run, **keyed by the data folder name** (`20250623`, `2026_1103 Nb analysis`): target, laser, instruments, file format, legacy settings |
| `pipeline.toml` | default parameters of the processing stages |
| `literature.toml` | literature data used in the comparison figures |
| `studies/*.toml` | cross-run comparisons (runs, layout, exclusions) |

Values that are assumptions are listed in `provisional` (e.g. the FC aperture of
the 2026_1103 run). When the real value is known, change it in the database and
re-run the calibration and the stages after it.

`EIA_DB_ROOT` can point to another database folder (default: `db/` of the repository).

## Setup (Windows, VS Code)

Requirements: Python ≥ 3.11 (3.12 recommended), Git, VS Code.

```powershell
git clone https://github.com/acannavo/EIA_BNL.git C:\Users\acannavo\code\EIA_BNL
cd C:\Users\acannavo\code\EIA_BNL
. .\env.ps1 -Install      # first time: creates .venv and installs eiakit + dev tools
eiakit info               # versions, paths, run folders found
pytest                    # tests (set $env:EIA_SLOW=1 to include the slow real-data replays)
```

Every new terminal: `. .\env.ps1` (the leading dot matters). In VS Code the default
terminal profile "EIA PowerShell" does this automatically, and Run/Debug uses the
`.env` file written by the script.

To use another Python: `. .\env.ps1 -Install -Python "C:\path\to\python.exe"`.
To change the base folder on another machine, create `env.local.ps1` (git-ignored)
containing `$EIA_BASE = 'D:\...\NewEIASoftware'`.

If PowerShell refuses to run the script, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

## Commands

```
eiakit info                          environment, paths, run folders
eiakit checksum [--root DIR] [--out FILE] [--skip-variants]
                                     SHA-256 manifest of a tree (default: data root,
                                     written to processed\_manifests)
eiakit checksum-compare A.csv B.csv  compare two manifests
eiakit runs [--all]                  runs and their conditions (--all: also runs not analysed)
eiakit db-check                      validate the database
eiakit files KEY [--list]            raw files picked up / skipped by the run's file format
eiakit audit [KEY ...] [--study NAME]
                                     audit of the legacy results -> processed\KEY\00_legacy_import
eiakit peaks KEY [--U V ...] [--max-clicks N] [--deep] [--write-state] [--search] [-v]
                                     peak stage (untitled28): refits each voltage from the raw
                                     SEM files, recovers the GUI state that was not saved (U
                                     arrow clicks, averaging grid; --deep: LOF clicks, Corr) and
                                     compares with Output_detail -> processed\KEY\10_peaks;
                                     --write-state stores the recovered state in
                                     db\legacy_peaks\KEY.toml (versioned), reused afterwards
eiakit calibrate KEY [--fc start|stop|both] [--baseline auto|pretrigger|V] [--legacy-cfy]
                                     calibration check with other FC choices, compared with
                                     the legacy checkpoint -> processed\KEY\20_calibration_check
eiakit cbs [KEY ...]                 CBS stage (cbs_manual_fitter_v12): recomputes areas,
                                     fractions, fitted peak / FWHM and chi2 from the stored
                                     (kT, V0, A, window) and the checkpoint, compares with the
                                     CBS fit / areas files -> processed\KEY\30_cbs
eiakit status [KEY ...] [-v]         state of every stage of every run: ok / STALE (with the
                                     reasons: db values, input files, code, upstream) / -- / FAILED
eiakit run [KEY ...] [--stage S ...] [--force] [--dry-run]
                                     runs what is missing or stale, in dependency order
eiakit study NAME [--part P ...] [--source pipeline|legacy] [--html-only] [--status]
                                     cross-run figures and tables of a study, parts:
                                     powerlaw        zbar fits, Sheil spectra, per-z spectra grids
                                     cbs             peak velocity, picks, ions / FWHM, fit summary
                                     reconstruction  CBS decomposition + residuals of chosen z
                                     collapse        u2/u1 = z/z* (curve peaks and CBS fits),
                                                     energy scaling, literature, peak check
eiakit snapshot NAME --label L [--source pipeline|legacy] [--note TEXT]
                                     every paper number of the study (z̄ fits, Sheil yields, CBS
                                     kT / V0 / ions, velocities, z*, energy fits) to
                                     studies\<NAME>\snapshots\L.csv (+ .json: switches, git)
eiakit diff NAME A B [--changelog TITLE] [--text TEXT]
                                     what changed between two snapshots (diffs\A__B.csv);
                                     --changelog appends the entry to studies\<NAME>\changelog.md
eiakit compare NAME KEY A B [--changelog TITLE] [--text TEXT]
                                     effect of one setting of db\pipeline.toml (see below)
eiakit methods NAME [--only P ...] [--boot N]
                                     method studies (Step 12) -> studies\<NAME>\methods\
                                     (methods.html): zbar (I4 zero fill, I5 noise floor),
                                     calibration (I2 peak vs area, EIA/FC overlay),
                                     sensitivity (flight length, time zero), fc (start/stop,
                                     shot-to-shot FC), shots (N12 bootstrap over shots)
eiakit --set SECTION.KEY=VALUE <command>
                                     override a setting for one command only
eiakit gui                           the eiakit window (see below)
eiakit fit-import [KEY ...] [--force]
                                     start the CBS fit store from the legacy fit files
eiakit import-exclusions FILE [--study PRL_AlNb]
                                     convert a legacy ExclusionList.txt to the study's [exclude] table
```

Checking that `data\` is an exact copy of the original tree:

```powershell
$orig = "C:\Users\acannavo\OneDrive - Brookhaven National Laboratory\NewIEA"
eiakit checksum --root "$orig\20250506_EIA_Mid_Mass" --out orig_2025.csv
eiakit checksum --root "$orig\2026_1103 Nb analysis"  --out orig_2026.csv
eiakit checksum --out copy.csv                                   # whole data root
eiakit checksum --root "$env:EIA_DATA_ROOT\2026_1103 Nb analysis" --out copy_2026.csv
eiakit checksum-compare orig_2025.csv copy.csv       # "only in B" = the 2026 folder
eiakit checksum-compare orig_2026.csv copy_2026.csv
```
