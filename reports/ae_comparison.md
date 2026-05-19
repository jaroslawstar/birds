# Encoder-Decoder Embedding Comparison — CUB-200

Architecture: ResNet-50 (fine-tuned) → projection head → decoder (denoising AE).
MLP classifier trained on frozen embeddings for 30 epochs.

## Results

| Embedding | Dim | Top-1 (%) | Top-5 (%) | Best Epoch |
|-----------|-----|-----------|-----------|------------|
| 512       | 512 |     59.48 |     86.42 |         19 |
| 256       | 256 |     58.82 |     85.64 |         15 |
| pca       | 446 |     59.98 |     86.16 |         22 |

## Delta vs 512-d baseline

| Embedding | Delta Top-1 | Delta Top-5 |
|-----------|-------------|-------------|
| 256       | -0.66        | -0.78        |
| pca       | +0.50        | -0.26        |

## PCA Details

```
n_components: 446
explained_variance: 95.0068%
source_encoder: ae_512
variance_threshold: 0.95
```

## Per-experiment artifacts

Each variant writes to `reports/exp_mlp_<tag>/`:
- `train_log.csv`, `per_class_accuracy.csv`, `confusion_matrix.png`, `eval_summary.txt`