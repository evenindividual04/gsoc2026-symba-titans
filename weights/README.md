# Trained Model Checkpoints

This directory contains all trained model weights, training logs, ablation study checkpoints, and generated plots.

## 📦 Download Pre-trained Weights

**Note:** If checkpoint files (`.pt`) are missing due to size constraints, they can be:
1. **Downloaded** from external storage (if provided)
2. **Regenerated** by running notebooks with `SKIP_TRAINING = False`

## 🎯 Best Model Checkpoints

### Baseline Transformer

| Model | File | Parameters | Test Performance |
|---|---|---|---|
| QED Baseline | `baseline_qed_best_ckpt.pt` | 1,437,696 | 88.89% EM (32/36) |
| QCD Baseline | `baseline_qcd_best_ckpt.pt` | 1,437,696 | 79.17% EM (19/24) |

**Architecture:** Standard encoder-decoder Transformer with Pre-LN
- d_model=128, nhead=4, 3 encoder layers, 3 decoder layers
- dim_feedforward=512, dropout=0.1
- Sinusoidal positional encoding (max_pos=2,200)

**Training:**
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- LR schedule: Linear warmup (10%) + cosine decay
- QED: 148 epochs (early stopped), batch_size=16
- QCD: 750 epochs, batch_size=4 with accum_steps=4

### Titans Memory-as-Layer (MAL)

| Model | File | Parameters | Test Performance |
|---|---|---|---|
| QED Titans | `titans_mal_qed_best_ckpt.pt` | 1,587,459 | 83.33% EM (30/36) |
| QCD Titans | `titans_mal_qcd_best_ckpt.pt` | 1,587,459 | 83.33% EM (20/24) |

**Architecture:** Baseline + MAL memory module
- Memory module: 2-layer MLP (d_model → 64 → d_model)
- Surprise-weighted updates with momentum (β=0.7)
- Memory LR: 0.005 (16.67× faster than backbone)
- +149,763 parameters over baseline (+10.4% overhead)

**Training:**
- Same optimizer/schedule as baseline
- QED: 150 epochs, batch_size=16
- QCD: 750 epochs (extended for memory convergence)

**Key Improvement:** QCD token accuracy 81.67% → 97.06% (+15.39pp)

## 📊 Training Logs

| File | Model | Description |
|---|---|---|
| `baseline_qed_training_log.csv` | Baseline QED | Per-epoch train/val loss, accuracy, exact match |
| `baseline_qcd_training_log.csv` | Baseline QCD | 750 epochs of training metrics |
| `titans_qed_training_log.csv` | Titans QED | MAL training progression |
| `titans_qcd_training_log.csv` | Titans QCD | Extended training showing memory convergence |

**Columns:** `epoch`, `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_em`, `val_em`, `learning_rate`

## 🔬 Ablation Study Checkpoints

### Training Duration Ablation (QCD)

| Checkpoint | Epochs | Purpose |
|---|---|---|
| `transformer_qcd_150ep_standard.pt` | 150 | Baseline at early stop point |
| `transformer_qcd_250ep_standard.pt` | 250 | Mid-training checkpoint |
| `transformer_qcd_750ep_cosine.pt` | 750 | Extended training with cosine LR |
| `titans_mal_qcd_150ep_standard.pt` | 150 | MAL at baseline stopping point |
| `titans_mal_qcd_250ep_standard.pt` | 250 | MAL mid-training |
| `titans_mal_qcd_750ep_cosine.pt` | 750 | MAL fully converged |

**Finding:** 750 epochs required for memory module to learn effective write patterns. Early stopping at 150 epochs terminates before memory stabilizes.

### Learning Rate Schedule Ablation

| Checkpoint | Schedule | Test EM |
|---|---|---|
| `transformer_qcd_750ep_standard.pt` | Standard decay | 79.17% |
| `transformer_qcd_750ep_cosine.pt` | Cosine annealing | 79.17% |
| `titans_mal_qcd_750ep_standard.pt` | Standard decay | 83.33% |
| `titans_mal_qcd_750ep_cosine.pt` | Cosine annealing | 83.33% |

**Finding:** Both schedules achieve similar final performance. Standard decay slightly more stable for long training (750 epochs).

## 📈 Generated Plots

Located in `plots/` subdirectory:

### Dataset Analysis
- `vocab_breakdown.png` — Vocabulary composition (192 tokens across 11 categories)
- `token_length_distributions.png` — QED vs QCD sequence length histograms

### Baseline Analysis
- `baseline_error_qed.png` — Error type breakdown for QED test set
- `baseline_error_qcd.png` — Error type breakdown showing 62.5% length errors
- `baseline_length_scatter.png` — Predicted vs. true QCD sequence lengths
- `baseline_ablation_learning_curves.png` — Model size (Small vs. Medium) and training duration ablations

### Titans Analysis
- `titans_qcd_config_comparison.png` — Memory dimension ablation study
- `transformer_qcd_ext_learning_curve.png` — Baseline extended training (150→750 epochs)
- `titans_qcd_ext_learning_curve.png` — MAL extended training showing memory convergence

## 🔄 Reproducing Training

### Quick Start (Use Pre-trained Weights)

Set in notebooks:
```python
SKIP_TRAINING = True  # Default
```

Models will load from checkpoint files and evaluate on test sets.

### Retrain from Scratch

Set in notebooks:
```python
SKIP_TRAINING = False
```

**Training Time Estimates:**
- QED Baseline: ~8 minutes (T4 GPU)
- QCD Baseline: ~45 minutes (T4 GPU, 750 epochs)
- QED Titans: ~10 minutes (T4 GPU)
- QCD Titans: ~55 minutes (T4 GPU, 750 epochs)

**Device Support:** Auto-detects `cuda` → `mps` (Apple Silicon) → `cpu`

## 📝 Results Summary Files

| File | Description |
|---|---|---|
| `transformer_results.json` | Baseline test metrics (exact match, token accuracy, CI) |
| `titans_results.json` | Titans MAL test metrics with comparison to baseline |

## 🗂️ Checkpoint Loading

All checkpoints can be loaded with PyTorch:

```python
import torch

# Load checkpoint
checkpoint = torch.load('baseline_qed_best_ckpt.pt', map_location='cpu')

# Extract components
model_state = checkpoint['model_state_dict']
optimizer_state = checkpoint['optimizer_state_dict']
epoch = checkpoint['epoch']
val_loss = checkpoint['val_loss']

# Load into model
model.load_state_dict(model_state)
```

**Checkpoint Contents:**
- `model_state_dict` — Model parameters
- `optimizer_state_dict` — Optimizer state (AdamW)
- `epoch` — Training epoch when saved
- `val_loss` — Validation loss at checkpoint
- `val_accuracy` — Validation token accuracy
- `val_exact_match` — Validation exact match %

## 📚 References

- Training implementation: `notebooks/notebook2_transformer_baseline.ipynb`, `notebook3_titans_mal.ipynb`
- Architecture details: See `ARCHITECTURE.md`
- Full results analysis: See `RESULTS.md`
- LaTeX report: `latex-report/report.pdf` (§6 Results, §7 Error Analysis)
