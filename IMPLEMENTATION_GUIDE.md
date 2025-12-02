# ECG Tensor Pipeline - Implementation Complete

## Generated Files

### Python Modules (src/)
1. **`__init__.py`** - Package initialization with module exports
2. **`utils.py`** - Utility functions (JSON I/O, seeding, path handling, device detection)
3. **`preprocessing.py`** - Streaming preprocessing with unified label mapping
4. **`dataloaders.py`** - PyTorch Dataset and DataLoader creation with lazy loading
5. **`model.py`** - 1D ResNet architecture for ECG classification
6. **`training.py`** - Training loop with mixed precision and checkpointing
7. **`eval.py`** - Evaluation metrics and visualization

### Notebook
- **`notebooks/ecg_tensor_pipeline.ipynb`** - Complete runnable pipeline

### Configuration
- **`requirements.txt`** - Updated with essential dependencies

## Key Features

### 1. Robust Path Handling
- Multiple path variant matching for cross-dataset compatibility
- Handles Windows/Unix path separators
- Special handling for CinC dataset naming conventions

### 2. Streaming Preprocessing
- Processes files one at a time (memory efficient)
- Supports WFDB (.hea/.dat) and MAT formats
- Automatic resampling to TARGET_FS (500 Hz)
- Z-score normalization
- Pad/truncate to fixed length (5000 samples = 10 seconds)
- Unified label mapping from CSV
- Stratified train/val/test splits (80/10/10)

### 3. Lazy Loading
- Individual compressed .npz files per record
- PyTorch Dataset reads on-demand
- Optional lightweight augmentation (Gaussian noise)
- Weighted sampler for class imbalance

### 4. Model Architecture
- 1D ResNet with residual blocks
- Batch normalization and GELU activation
- Adaptive global pooling
- ~1.5M trainable parameters (configurable)

### 5. Training Features
- Automatic mixed precision (AMP) on CUDA
- Gradient accumulation support
- Cosine annealing LR scheduler
- Best and latest checkpointing
- JSON training log
- Progress bars with tqdm

### 6. Evaluation
- Confusion matrix
- Per-class precision/recall/F1
- Macro/micro/weighted F1 scores
- Bar plot visualization
- Classification report

## Notebook Usage

### Cell-by-Cell Execution

**Cell 1 - Environment Setup**
- Rehydrates paths (ROOT, DATASET_DIR, PROCESSED_DIR, etc.)
- Creates directories if missing
- Imports src modules
- Sets up logging

**Cell 2 - Configuration**
- Sets TARGET_FS=500, TARGET_SAMPLES=5000
- Defines LABEL_ORDER=['MI','AF','BBB','NORM','OTHER']
- Detects CUDA and prints GPU info
- Sets hyperparameters (batch size, learning rate, epochs)
- Sets random seed for reproducibility

**Cell 3 - Check Dataset**
- Verifies unified_label_mapping.csv exists
- Lists dataset subdirectories

**Cell 4 - Smoke Test Preprocessing**
- Processes first 10 files for quick testing
- Saves to artifacts/processed/smoke_test/
- Verifies NPZ files created

**Cell 5 - Full Preprocessing**
- Optional: Set RUN_FULL_PREPROCESSING=True
- Processes all files in dataset
- Saves to artifacts/processed/
- Creates splits.json, label_map.json, labels.npy

**Cell 6 - Create DataLoaders**
- Loads splits.json
- Creates train/val/test DataLoaders
- Tests batch loading
- Prints batch shapes

**Cell 7 - Model Instantiation**
- Creates ECGResNet1D model
- Moves to device (CUDA/CPU)
- Prints parameter count
- Shows architecture

**Cell 8 - Training**
- Creates Trainer instance
- Runs training loop with progress bars
- Saves checkpoints (best.pth, latest.pth)
- Logs metrics to JSON

**Cell 9 - Plot Training History**
- Plots loss and accuracy curves
- Saves to figures/training_history.png

**Cell 10 - Evaluation**
- Loads best checkpoint
- Evaluates on test set
- Computes all metrics
- Saves metrics JSON and F1 plot

**Cell 11 - Pipeline Summary**
- Prints complete pipeline statistics
- Shows output locations

## Running the Pipeline

### Quick Start (Smoke Test)
```python
# Run cells 1-9 with default settings
# Uses smoke test (10 files) for quick verification
```

### Full Pipeline
```python
# Cell 5: Set RUN_FULL_PREPROCESSING = True
# Cell 6: Set USE_SMOKE_TEST = False
# Cell 2: Adjust EPOCHS as needed (50+ for production)
# Run all cells
```

### Out-of-Order Execution
Each cell rehydrates environment variables, so you can:
- Re-run configuration cell to change hyperparameters
- Re-run training with different settings
- Re-run evaluation without retraining

## Output Structure

```
artifacts/
  processed/
    smoke_test/          # Smoke test outputs
      records/           # Individual .npz files
      splits.json        # Train/val/test splits
      label_map.json     # Label encoding
      labels.npy         # All labels array
    records/             # Full dataset (if full preprocessing run)
    splits.json
    label_map.json
    labels.npy
    test_metrics.json    # Evaluation results
  models/
    best.pth             # Best model checkpoint
    latest.pth           # Latest checkpoint
  training_log.json      # Training history
figures/
  training_history.png   # Loss/accuracy plots
  test_f1_scores.png     # Per-class F1 bar chart
logs/
  pipeline.log           # Execution log
  unified_label_mapping.csv  # Label mapping (required input)
```

## Unified Label Mapping CSV Format

Required columns:
- `dataset`: Dataset name (e.g., 'ptb-xl', 'PTB_Diagnostic')
- `record_id`: Relative path to record within dataset
- `mapped_label`: Target label ('MI', 'AF', 'BBB', 'NORM', 'OTHER')

Example:
```csv
dataset,record_id,mapped_label
ptb-xl,records100/00001_lr,NORM
PTB_Diagnostic,patient001/s0014lre,MI
CinC2017,A00001,AF
```

## Error Handling

### Missing unified_label_mapping.csv
- Warning printed
- All records mapped to 'OTHER'
- Pipeline continues

### File Read Failures
- Individual failures logged but skipped
- Processing continues with remaining files
- Failure count reported in summary

### Unexpected Sampling Rates
- Automatic resampling to TARGET_FS
- Warning logged if significant deviation

## Performance Optimization

### For GPU Training
- Mixed precision (AMP) enabled automatically on CUDA
- pin_memory=True for faster CPU-GPU transfer
- Increase num_workers (4-8) for data loading
- Increase batch_size based on GPU memory

### For CPU Training
- Reduce batch_size (8-16)
- Reduce num_workers (2-4)
- Consider reducing model base_channels (32 instead of 64)

### For Large Datasets
- Preprocessing is already streaming (no memory issues)
- Can process datasets with millions of records
- Splits are created after all files processed

## Troubleshooting

### Import Errors
- Verify all src/*.py files exist
- Check ROOT path is set correctly
- Ensure Python 3.10+ is used

### CUDA Out of Memory
- Reduce BATCH_SIZE (try 16, 8, or 4)
- Reduce model base_channels (32)
- Enable gradient accumulation (grad_accum_steps=2)

### No Files Found
- Check DATASET_DIR path
- Verify datasets are in correct subdirectories
- Check for .hea or .mat files in subdirectories

### Label Mapping Issues
- Verify unified_label_mapping.csv format
- Check dataset names match directory names
- Use robust_path_variants for debugging

## Next Steps

1. **Prepare unified_label_mapping.csv**
   - Map all records from your datasets to unified labels
   - Place in logs/ directory

2. **Run smoke test**
   - Verify preprocessing works (cells 1-6)
   - Check output files in artifacts/processed/smoke_test/

3. **Run full preprocessing**
   - Set RUN_FULL_PREPROCESSING=True
   - May take 10-60 minutes depending on dataset size

4. **Train model**
   - Start with EPOCHS=10 for testing
   - Increase to 50-100 for production
   - Monitor validation F1 score

5. **Evaluate and iterate**
   - Check per-class performance
   - Adjust label mapping if needed
   - Try different hyperparameters

## Code Quality Features

- Type hints on all functions
- Comprehensive docstrings
- Logging instead of print in modules
- Error handling with descriptive messages
- Progress bars for long operations
- Checkpointing for interrupted training
- Reproducible with seed setting

## Dependencies

All required packages in requirements.txt:
- numpy, pandas, scipy, matplotlib
- wfdb (WFDB format support)
- torch, torchvision (deep learning)
- scikit-learn (metrics)
- tqdm (progress bars)
- jupyter, jupyterlab (notebook)

Install with:
```bash
pip install -r requirements.txt
```

## Validation

Run the smoke test to verify:
1. Files can be read from dataset
2. Preprocessing produces valid .npz files
3. DataLoaders can load batches
4. Model forward pass works
5. Training loop runs without errors
6. Evaluation produces metrics

All validation happens in cells 1-9 with smoke test data.

## Final Confirmation

When preprocessing completes successfully, you'll see:
```
Preprocessing with unified labels completed successfully.
```

When training completes successfully, you'll see:
```
Training complete!
  Best validation F1: X.XXXX
  Checkpoints saved to: artifacts/models
```

When evaluation completes successfully, you'll see:
```
Evaluation complete!
  Metrics saved to: artifacts/processed/test_metrics.json
  F1 plot saved to: figures/test_f1_scores.png
```

---

**Implementation Complete**: All modules and notebook are ready for execution.

