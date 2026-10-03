param(
    [switch]$Reproduce,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create virtual environment. Install Python 3.12 and retry.' }
    & $taskPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $taskPython -c 'import flask, sklearn, matplotlib, pytest, PIL'
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$taskArtifactRoot = $PSScriptRoot
if ($Reproduce) {
    $taskArtifactRoot = Join-Path $PSScriptRoot 'runs\reproduction'
}
if ($Reproduce -or -not (Test-Path -LiteralPath (Join-Path $taskArtifactRoot 'models\digit_pipeline.joblib'))) {
    if ((Test-Path -LiteralPath (Join-Path $taskArtifactRoot 'reports\evaluation.json')) -or (Test-Path -LiteralPath (Join-Path $taskArtifactRoot 'data\test_access.json'))) {
        throw 'Existing final test results are protected. See README for a separate reproducibility directory or audited reruns.'
    }
    & $taskPython -m src.data --root $taskArtifactRoot
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
    & $taskPython -m src.train --root $taskArtifactRoot
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
    & $taskPython -m src.evaluate --root $taskArtifactRoot
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
}
if ($Check) {
    & $taskPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    exit 0
}
Write-Host 'DigitLab: http://127.0.0.1:5000 (Ctrl+C to stop)'
& $taskPython -m app --root $taskArtifactRoot
