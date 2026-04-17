param(
    [string]$ArchiveName = "",
    [switch]$IncludeData,
    [switch]$IncludeDocxOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($ArchiveName)) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ArchiveName = "thyroid_ai_results_$Timestamp.zip"
}

$StageDir = Join-Path $ProjectRoot "_pack_results"
if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Path $StageDir | Out-Null

New-Item -ItemType Directory -Path (Join-Path $StageDir "evaluation") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageDir "reports") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageDir "models") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageDir "reports\workflow_library") -Force | Out-Null

if ($IncludeDocxOnly) {
    Copy-Item reports\*.docx (Join-Path $StageDir "reports") -Force -ErrorAction SilentlyContinue
    if (Test-Path reports\workflow_library) {
        Copy-Item reports\workflow_library\*.docx (Join-Path $StageDir "reports\workflow_library") -Force -ErrorAction SilentlyContinue
    }
} else {
    Copy-Item evaluation\* (Join-Path $StageDir "evaluation") -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item reports\* (Join-Path $StageDir "reports") -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item models\*.pth (Join-Path $StageDir "models") -Force -ErrorAction SilentlyContinue
    Copy-Item models\*.pkl (Join-Path $StageDir "models") -Force -ErrorAction SilentlyContinue
    if (Test-Path models\backbone_benchmark) {
        Copy-Item models\backbone_benchmark (Join-Path $StageDir "models\backbone_benchmark") -Recurse -Force
    }
}

Copy-Item configs\config.yaml $StageDir -Force
Copy-Item reports\project_structure.txt $StageDir -Force -ErrorAction SilentlyContinue
Copy-Item scripts\setup_and_run.ps1 $StageDir -Force -ErrorAction SilentlyContinue
Copy-Item scripts\collect_results.ps1 $StageDir -Force -ErrorAction SilentlyContinue

if ($IncludeData) {
    New-Item -ItemType Directory -Path (Join-Path $StageDir "data") | Out-Null
    Copy-Item data\cleaned_analysis_dataset.xlsx (Join-Path $StageDir "data") -Force -ErrorAction SilentlyContinue
    Copy-Item data\thyroid_research_dataset_publishable_v2.xlsx (Join-Path $StageDir "data") -Force -ErrorAction SilentlyContinue
}

$ArchivePath = Join-Path $ProjectRoot $ArchiveName
if (Test-Path $ArchivePath) {
    Remove-Item -Force $ArchivePath
}

Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ArchivePath -Force
Remove-Item -Recurse -Force $StageDir

Write-Host "Created archive: $ArchivePath"
