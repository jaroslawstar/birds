# Encoder-Decoder Embedding Comparison — CUB-200

Architecture: ResNet-50 (fine-tuned) → projection head → decoder (denoising AE).
MLP classifier trained on frozen embeddings for 30 epochs.

## Results

| Embedding | Dim | Top-1 (%) | Top-5 (%) | Best Epoch |
|-----------|-----|-----------|-----------|------------|
| 512       | 512 |     19.04 |     46.43 |         26 |
| 256       | 256 |     38.07 |     69.95 |         30 |
| pca       | 448 |     59.63 |     86.14 |         18 |

## Delta vs 512-d baseline

| Embedding | Delta Top-1 | Delta Top-5 |
|-----------|-------------|-------------|
| 256       | +19.04        | +23.52        |
| pca       | +40.59        | +39.71        |

## PCA Details

```
n_components: 448
explained_variance: 95.0093%
source_encoder: ae_512
variance_threshold: 0.95
```

## Per-experiment artifacts

Each variant writes to `reports/exp_mlp_<tag>/`:
- `train_log.csv`, `per_class_accuracy.csv`, `confusion_matrix.png`, `eval_summary.txt`