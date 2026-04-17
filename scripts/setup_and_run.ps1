param(
    [string]$PythonExe = "python",
    [string]$VenvDir = ".venv",
    [int]$Epochs = 40,
    [int]$BatchSize = 4,
    [int]$BackboneEpochs = 30,
    [switch]$SkipVenv,
    [switch]$SkipInstall,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$ConfigPath = Join-Path $ProjectRoot "configs\config.yaml"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host "==> $Name"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

function Get-ConfigModelType {
    $content = Get-Content $ConfigPath -Raw
    $match = [regex]::Match($content, '(?m)^(\s*model_type:\s*)(\S+)\s*$')
    if (-not $match.Success) {
        throw "Could not find model_type in $ConfigPath"
    }
    return $match.Groups[2].Value
}

function Set-ConfigModelType {
    param(
        [string]$ModelType
    )

    $content = Get-Content $ConfigPath -Raw
    $updated = [regex]::Replace($content, '(?m)^(\s*model_type:\s*)(\S+)\s*$', "`$1$ModelType")
    if ($updated -eq $content) {
        throw "Failed to update model_type in $ConfigPath"
    }
    Set-Content -Path $ConfigPath -Value $updated -Encoding UTF8
}

Write-Host "Project root: $ProjectRoot"
$OriginalModelType = Get-ConfigModelType

if (-not $SkipVenv) {
    if (-not (Test-Path $VenvDir)) {
        Invoke-Step "Create virtual environment" { & $PythonExe -m venv $VenvDir }
    }
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    . $ActivateScript
}

if (-not $SkipInstall) {
    Invoke-Step "Upgrade pip" { python -m pip install --upgrade pip }
    Invoke-Step "Install torch/torchvision" { pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 }
    Invoke-Step "Install project requirements" { pip install -r requirements.txt }
}

Write-Host "Checking CUDA availability..."
Invoke-Step "CUDA check" { python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" }

$GpuArgs = @()
if ($CpuOnly) {
    $GpuArgs += "--cpu-only"
}

Invoke-Step "Clinical analysis" { python evaluation/run_thyroid_analysis.py }
Invoke-Step "Clinical DOCX report" { python reports/generate_clinical_analysis_docx.py }

Invoke-Step "Multimodal baseline pipeline" { python training/run_multimodal_thyroid_pipeline.py --epochs $Epochs --batch-size $BatchSize @GpuArgs }
Invoke-Step "Multimodal DOCX report" { python reports/generate_multimodal_docx_report.py }

Invoke-Step "Patient-level fusion model" { python training/run_patient_level_fusion.py --epochs $Epochs --batch-size $BatchSize @GpuArgs }
Invoke-Step "Fusion DOCX report" { python reports/generate_fusion_sci_docx.py }

Invoke-Step "Backbone benchmark" { python training/run_backbone_benchmark.py --epochs $BackboneEpochs --batch-size $BatchSize @GpuArgs }
Invoke-Step "Backbone benchmark DOCX report" { python reports/generate_backbone_benchmark_docx.py }

try {
    Set-ConfigModelType "final_multimodal"
    Invoke-Step "Final multimodal model (baseline)" { python training/run_final_multimodal_model.py --epochs $Epochs --batch-size $BatchSize @GpuArgs }

    Set-ConfigModelType "weak_supervised_reasoning"
    Invoke-Step "Final multimodal model (weak supervised reasoning)" { python training/run_final_multimodal_model.py --epochs $Epochs --batch-size $BatchSize @GpuArgs }

    Invoke-Step "Workflow overview DOCX" { python reports/generate_workflow_overview_docx.py }
}
finally {
    Set-ConfigModelType $OriginalModelType
}

Write-Host "Pipeline completed."
Write-Host "Key outputs:"
Write-Host "  evaluation/"
Write-Host "  reports/"
Write-Host "  models/"
Write-Host "  reports/workflow_library/"
