param(
    [string[]]$Position = @("all"),
    [int]$Steps = 50,
    [int]$Trials = 2,
    [ValidateSet("fixed", "agent")]
    [string]$StopMode = "agent",
    [int]$MinStepsBeforeStop = 8,
    [double]$PostStepDelay = 3.0,
    [string]$Segment = "",
    [switch]$Debug,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $scriptRoot

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $repoRoot
}

if (-not $PythonExe) {
    $preferredPython = "C:\Users\vishv\miniconda3\envs\rl_proj\python.exe"
    if (Test-Path $preferredPython) {
        $PythonExe = $preferredPython
    } else {
        $PythonExe = "python"
    }
}

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $PythonExe"
}

if (-not $env:TINKER_API_KEY -and -not (Test-Path ".env")) {
    throw "TINKER_API_KEY is not set and no .env file was found."
}

if (-not $env:TINKER_SUBPROCESS_SAMPLING) {
    $env:TINKER_SUBPROCESS_SAMPLING = "1"
}

$resultsDir = "results\manual_test"
$plotDir = Join-Path $resultsDir "plots"
$summaryPath = Join-Path $resultsDir "summary.csv"
$stepsPath = Join-Path $resultsDir "steps.csv"

$evalArgs = @(
    ".\evaluate_qwen3_comparison.py",
    "--position"
) + $Position + @(
    "--steps", $Steps,
    "--trials", $Trials,
    "--stop-mode", $StopMode,
    "--min-steps-before-stop", $MinStepsBeforeStop,
    "--post-step-delay", $PostStepDelay
)

if ($Segment) {
    $evalArgs += @("--segment", $Segment)
}
if ($Debug) {
    $evalArgs += "--debug"
}

Write-Host "Running Qwen3 comparison eval..."
& $PythonExe @evalArgs
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed."
}

Write-Host "Summarizing run group..."
& $PythonExe ".\summarize.py" "--results-dir" $resultsDir
if ($LASTEXITCODE -ne 0) {
    throw "Summarization failed."
}

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
Copy-Item "results\summary.csv" $summaryPath -Force
Copy-Item "results\steps.csv" $stepsPath -Force

if (Test-Path $plotDir) {
    Remove-Item -Recurse -Force $plotDir
}
New-Item -ItemType Directory -Path $plotDir -Force | Out-Null

Write-Host "Generating comparison plots..."
& $PythonExe ".\plot_results.py" "--summary" $summaryPath "--steps" $stepsPath "--metric" "best_z_on_nerve_gained" "--out-dir" $plotDir
if ($LASTEXITCODE -ne 0) {
    throw "Plot generation failed for best_z_on_nerve_gained."
}

$extraMetrics = @(
    "z_gained",
    "steps_visible",
    "steps_uncertain",
    "steps_not_visible"
)
foreach ($metric in $extraMetrics) {
    & $PythonExe ".\plot_results.py" "--summary" $summaryPath "--steps" $stepsPath "--metric" $metric "--out-dir" $plotDir "--no-steps"
    if ($LASTEXITCODE -ne 0) {
        throw "Plot generation failed for metric '$metric'."
    }
}

Write-Host ""
Write-Host "Run complete."
Write-Host "Results dir: $resultsDir"
Write-Host "Summary:     $summaryPath"
Write-Host "Steps:       $stepsPath"
Write-Host "Plots:       $plotDir"
