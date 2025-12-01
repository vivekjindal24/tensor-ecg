================================================================================
ECG PREPROCESSING AUTOMATION - COMPLETE GUIDE
================================================================================
Generated: 2025-12-01

PROJECT STATUS
--------------------------------------------------------------------------------
✓ Environment validated (Python 3.11, PyTorch 2.9.1+cpu)
✓ Virtual environment active (.venv1)
✓ Free disk space: 740 GB
✓ Smoke test passed (500 files, 5.7 rec/s)
✓ Model compatibility verified
✓ Git repository initialized (branch: automated/preprocess-fix)

SCRIPTS CREATED
--------------------------------------------------------------------------------
1. validate_mapping.py         - Validate unified label mapping CSV
2. preprocess_streaming.py     - Idempotent streaming preprocessing
3. verify_smoke_test.py        - Verify preprocessing outputs
4. model_smoke_test.py         - Test model forward pass
5. improve_mapping.py          - Suggest label mapping improvements
6. apply_mapping_improvements.py - Apply suggested improvements
7. run_full_automation.py      - Complete automation orchestrator

FILES GENERATED
--------------------------------------------------------------------------------
Config:
  .gitignore                   - Git ignore rules

Logs:
  logs/preprocess_automation.log         - Real-time processing log
  logs/preprocess_report_smoke.txt       - Smoke test report
  logs/unmapped_sample.csv               - Sample of unmapped records

Outputs (from smoke test):
  artifacts/processed/records/*.npy      - Signal data (500 files)
  artifacts/processed/records/*.label    - Labels (500 files)
  artifacts/processed/records/*.meta.json - Metadata (500 files)
  artifacts/processed/manifest.jsonl     - Manifest
  artifacts/processed/splits.json        - Train/val/test splits
  artifacts/processed/label_map.json     - Label mapping
  artifacts/processed/labels.npy         - Label array
  artifacts/processed/progress.json      - Progress checkpoint
  artifacts/processed/checkpoints/checkpoint_skel.pth - Model checkpoint

USAGE INSTRUCTIONS
================================================================================

OPTION A: AUTOMATED ORCHESTRATOR (RECOMMENDED)
--------------------------------------------------------------------------------
Run the complete automation with interactive prompts:

    cd D:\ecg-research
    .\.venv1\Scripts\python.exe scripts\run_full_automation.py

This will:
  1. Validate mapping
  2. Optionally improve mapping coverage
  3. Run preprocessing (with mode selection)
  4. Verify outputs
  5. Test model compatibility
  6. Generate final report

OPTION B: MANUAL STEP-BY-STEP
--------------------------------------------------------------------------------

Step 1: Validate mapping
    .\.venv1\Scripts\python.exe scripts\validate_mapping.py

Step 2: (Optional) Improve mapping
    .\.venv1\Scripts\python.exe scripts\improve_mapping.py
    # Review: logs/mapping_improvements_suggested.csv
    .\.venv1\Scripts\python.exe scripts\apply_mapping_improvements.py

Step 3: Run preprocessing
    # Smoke test (500 files)
    $env:ECG_PREPROCESS_LIMIT=500
    .\.venv1\Scripts\python.exe scripts\preprocess_streaming.py
    
    # Medium test (5,000 files, ~15 min)
    $env:ECG_PREPROCESS_LIMIT=5000
    .\.venv1\Scripts\python.exe scripts\preprocess_streaming.py
    
    # Full run (all files, ~7 hours)
    .\.venv1\Scripts\python.exe scripts\preprocess_streaming.py

Step 4: Verify outputs
    .\.venv1\Scripts\python.exe scripts\verify_smoke_test.py

Step 5: Test model
    .\.venv1\Scripts\python.exe scripts\model_smoke_test.py

OPTION C: DIRECT PREPROCESSING ONLY
--------------------------------------------------------------------------------
If you just want to run preprocessing:

    cd D:\ecg-research
    .\.venv1\Scripts\Activate.ps1
    
    # Set limit (optional, omit for full run)
    $env:ECG_PREPROCESS_LIMIT=5000
    
    # Run
    python scripts\preprocess_streaming.py

MONITORING & DEBUGGING
================================================================================

Monitor progress in real-time:
    Get-Content logs\preprocess_automation.log -Wait -Tail 20

Check processed count:
    (Get-ChildItem artifacts\processed\records\*.npy).Count

View progress checkpoint:
    Get-Content artifacts\processed\progress.json | ConvertFrom-Json

Resume interrupted run:
    # Just run the script again - it will skip processed files
    .\.venv1\Scripts\python.exe scripts\preprocess_streaming.py

LABEL MAPPING COVERAGE
================================================================================

Current status:
  - Total records: 84,556
  - Mapped: 28,578 (33.8%)
    • NORM: 19,286 (22.8%)
    • MI: 3,941 (4.7%)
    • AF: 2,771 (3.3%)
    • BBB: 2,580 (3.1%)
  - Unmapped (→ OTHER): 55,978 (66.2%)

To improve coverage:
  1. Run: python scripts\improve_mapping.py
  2. Review: logs\mapping_improvements_suggested.csv
  3. Apply: python scripts\apply_mapping_improvements.py

EXPECTED OUTPUTS
================================================================================

After full preprocessing, you will have:

Directory structure:
  artifacts/
    processed/
      records/                    # ~152,000 .npy files
      manifest.jsonl              # Complete record manifest
      splits.json                 # Train/val/test splits (80/10/10)
      label_map.json              # Label encoding
      labels.npy                  # Label array
      progress.json               # Resume checkpoint
      checkpoints/
        checkpoint_skel.pth       # Model checkpoint

File counts by dataset:
  - Chapman_Shaoxing: ~45,152
  - ptb-xl: ~21,799
  - CinC2017: ~17,056
  - PTB_Diagnostic: ~549
  - Plus additional .mat files

Storage requirements:
  - ~3.2 GB for processed records
  - Additional ~500 MB for manifests and metadata

PERFORMANCE NOTES
================================================================================

Processing speed: ~5.7 records/second (smoke test, CPU-only)

Factors affecting speed:
  ✓ Disk I/O (SSD recommended)
  ✓ CPU speed (single-threaded operations)
  ✓ Dataset location (local vs network)
  ✓ Antivirus real-time scanning (may slow file operations)

Optimization tips:
  - Use SSD for artifacts/ directory
  - Disable real-time antivirus scanning temporarily
  - Close unnecessary applications
  - Run overnight for full dataset

IDEMPOTENCY & RESUMABILITY
================================================================================

The preprocessing is designed to be safe and resumable:

✓ Skips existing files       - Checks for .npy, .label, and .meta.json
✓ Progress checkpoints        - Saved every 1000 records
✓ Append-only manifest        - No data loss on interruption
✓ No dataset modification     - Source files never touched

To restart from scratch:
    Remove-Item artifacts\processed\records\* -Force
    Remove-Item artifacts\processed\manifest.jsonl
    Remove-Item artifacts\processed\progress.json

NEXT STEPS AFTER PREPROCESSING
================================================================================

1. Review outputs:
     - Check logs\preprocess_report.txt
     - Verify splits.json counts
     - Inspect sample .npy files

2. Open notebook:
     jupyter notebook notebooks\ecg_tensor_pipeline.ipynb

3. Train model:
     - Use splits from splits.json
     - Load records lazily from .npy files
     - Monitor with MLflow (optional)

4. Evaluate:
     - Test set predictions
     - Confusion matrix
     - Per-class metrics
     - ROC/PR curves

GIT WORKFLOW
================================================================================

Current branch: automated/preprocess-fix

View changes:
    git status
    git log --oneline

Commit additional changes:
    git add <files>
    git commit -m "your message"

To merge to main (after validation):
    git checkout main
    git merge automated/preprocess-fix

TROUBLESHOOTING
================================================================================

Issue: Out of memory
Solution: Script uses streaming - should not happen. Check other applications.

Issue: Slow processing (<3 rec/s)
Solution: Check disk space, antivirus, move to SSD.

Issue: Missing wfdb module
Solution: pip install wfdb

Issue: "File not found" errors during processing
Solution: Check Dataset/ folder structure, verify paths in logs.

Issue: All records labeled as OTHER
Solution: Improve mapping with improve_mapping.py script.

Issue: Interrupted processing
Solution: Just run again - will resume from checkpoint.

CONTACT & SUPPORT
================================================================================

For issues or questions:
  1. Check logs\preprocess_automation.log
  2. Review this guide
  3. Examine error messages in terminal
  4. Verify dataset file structure

Logs location:
  - Main log: logs\preprocess_automation.log
  - Report: logs\preprocess_report.txt
  - Mapping validation: logs\unmapped_sample.csv

================================================================================
END OF GUIDE
================================================================================

QUICK START COMMAND:
    cd D:\ecg-research
    .\.venv1\Scripts\python.exe scripts\run_full_automation.py

