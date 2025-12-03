# 🚀 ECG Training Quick Start

**Last Updated:** December 3, 2025  
**Notebook:** `notebooks/master_pipeline.ipynb`

---

## 30-Second Start

```powershell
# 1. Open notebook
jupyter notebook notebooks/master_pipeline.ipynb

# 2. Run cells 1-6 (setup, preprocessing if needed)

# 3. Run "QUICK SMOKE RUN" cell (2 min test)

# 4. If smoke test passes ✓, uncomment and run "FULL PRODUCTION TRAINING" cell
```

---

## Quick Commands

### Interactive (in notebook)
```python
# Smoke test (2 min, 256 samples)
# → Run cell 11: "QUICK SMOKE RUN"

# Medium test (15-30 min, 5k samples)  
# → Uncomment and run cell 12: "MEDIUM RUN"

# Full production (1-4 hours, all data)
# → Uncomment and run cell 13: "FULL PRODUCTION TRAINING"
```

### Headless (PowerShell)
```powershell
# Full pipeline, no timeout
jupyter nbconvert --to notebook --execute notebooks/master_pipeline.ipynb --ExecutePreprocessor.timeout=-1 --output logs/run_$(Get-Date -Format 'yyyyMMdd_HHmmss').ipynb
```

---

## Configuration (Edit Cell 1)

```python
# Quick adjustments
BATCH_SIZE_GPU = 64        # Lower if GPU OOM (32, 16, 8)
EPOCHS = 20                # Reduce for faster tests (5, 10)
GRAD_ACCUM_STEPS = 1       # Increase if GPU memory limited (2, 4)
ECG_PREPROCESS_LIMIT = 0   # 0=all, >0=limit for quick runs

# Advanced
WEIGHT_DECAY = 1e-4        # Regularization strength
SCHEDULER_TYPE = 'cosine'  # 'cosine' or 'step'
EARLY_STOP_PATIENCE = 0    # 0=disabled, >0=early stop after N epochs
MIXED_PRECISION = True     # Auto-enabled on CUDA
DRY_RUN = False            # True=run 10 steps only (quick test)
```

---

## Expected Runtimes

| Mode | Samples | CPU | GPU (L4) |
|------|---------|-----|----------|
| Smoke | 256 | 2 min | 1 min |
| Medium | 5,000 | 45 min | 15 min |
| Full | 50,000+ | 4-8 hrs | 1-2 hrs |

---

## Output Locations

```
artifacts/processed/checkpoints/
  └─ best_model.pth                    ← Your trained model

artifacts/figures/
  ├─ training_curves_*.png             ← Loss, accuracy, F1
  ├─ confusion_matrix_*.png            ← Confusion matrix
  ├─ roc_curves_*.png                  ← ROC curves
  └─ precision_recall_curves_*.png     ← PR curves

logs/
  └─ smoke_test_results.json           ← Smoke test summary
```

---

## Troubleshooting (Top 3)

### 1. GPU Out of Memory
```python
# In config cell, change:
BATCH_SIZE_GPU = 16          # Was 64
GRAD_ACCUM_STEPS = 4         # Was 1
```

### 2. Slow on CPU
```python
# Run smaller test first:
ECG_PREPROCESS_LIMIT = 5000  # Instead of 0 (all)
EPOCHS = 5                   # Instead of 20
```

### 3. Low Validation F1
```python
# Check class balance first:
# Run pre-flight checklist cell (Section 9)

# Then try:
EPOCHS = 30                  # More training
LR = 5e-4                    # Lower learning rate
```

---

## Pre-Flight Checklist

Before full training, run this cell (Section 9):
```python
run_preflight_checks()
```

Should see:
- ✓ Dataset directory found
- ✓ Unified mapping CSV found
- ✓ Manifest found (N processed records)
- ✓ Splits found (train/val/test)
- ✓ GPU available (or ⚠ CPU fallback)
- ✓ Sufficient disk space

---

## After Training

```python
# Load best model
model = load_best_model(model)

# Check test metrics
import json
with open('artifacts/processed/checkpoints/test_metrics.json') as f:
    print(json.dumps(json.load(f), indent=2))

# Run inference on new ECG
signal_tensor = torch.randn(1, 5000)  # Your ECG signal
pred_class, probs = predict_sample(model, signal_tensor)
print(f"Predicted: {INT_TO_LABEL[pred_class]} with {probs[pred_class]:.2f} confidence")
```

---

## Environment Variables (Optional)

```powershell
# Custom config via env vars
$env:ECG_PREPROCESS_LIMIT = "5000"
$env:ECG_EPOCHS = "10"
$env:ECG_BATCH_SIZE = "32"
$env:ECG_LR = "0.0005"
$env:ECG_SEED = "42"

# Then run notebook
jupyter nbconvert --to notebook --execute notebooks/master_pipeline.ipynb --ExecutePreprocessor.timeout=-1 --output logs/custom_run.ipynb
```

---

## Next Steps

1. ✅ Run smoke test (verify pipeline works)
2. ✅ Run full training (uncomment cell 13)
3. ✅ Review metrics (`test_metrics.json`)
4. ✅ Inspect plots (`artifacts/figures/`)
5. ✅ Export model (ONNX, TorchScript)
6. ✅ Deploy for inference

---

## Key Files

| File | Purpose |
|------|---------|
| `notebooks/master_pipeline.ipynb` | Main notebook (run this) |
| `PRODUCTION_TRAINING_UPDATE.md` | Detailed documentation |
| `QUICKSTART_TRAINING.md` | This file (quick reference) |
| `logs/unified_label_mapping.csv` | Label mapping (required) |
| `artifacts/processed/splits.json` | Train/val/test splits |

---

## Need Help?

1. **Notebook Section 14:** Full runbook with troubleshooting
2. **PRODUCTION_TRAINING_UPDATE.md:** Detailed feature docs
3. **GitHub Issues:** https://github.com/vivekjindal24/tensor-ecg/issues

---

**Ready to train!** Start with smoke test, then proceed to full training when ready.

---

*Quick reference v1.0 - December 3, 2025*

