# Manual Run Guide

This file records the recommended command-line workflow for running the thyroid AI project step by step, as well as the one-click PowerShell workflow.

## Option 1. Run Step by Step

Run the following commands from the project root:

```powershell
python evaluation/run_thyroid_analysis.py
python reports/generate_clinical_analysis_docx.py

python training/run_multimodal_thyroid_pipeline.py --epochs 40 --batch-size 4
python reports/generate_multimodal_docx_report.py

python training/run_patient_level_fusion.py --epochs 40 --batch-size 4
python reports/generate_fusion_sci_docx.py

python training/run_backbone_benchmark.py --epochs 30 --batch-size 4
python reports/generate_backbone_benchmark_docx.py

python training/run_final_multimodal_model.py --epochs 40 --batch-size 4
python reports/generate_final_multimodal_docx.py
```

## Option 2. One-click Pipeline

If the environment is already prepared, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_and_run.ps1 -SkipVenv -SkipInstall -Epochs 40 -BatchSize 4 -BackboneEpochs 30
```

This script will:

- run the clinical analysis
- run the multimodal baseline pipeline
- run the patient-level fusion pipeline
- run the backbone benchmark
- run both final models:
  - `final_multimodal`
  - `weak_supervised_reasoning`
- generate the workflow overview document

## Option 3. Collect Final Results

After the pipeline finishes, package the outputs with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\collect_results.ps1
```

## Key Output Directories

- `evaluation/`
- `reports/`
- `models/`
- `reports/workflow_library/`

## Notes

- Run all commands from the project root directory.
- The one-click script automatically switches between the two final multimodal model types and restores the original config afterward.
- The workflow library folder is intended for long-term comparison of workflow notes, diagrams, and summary documents.
