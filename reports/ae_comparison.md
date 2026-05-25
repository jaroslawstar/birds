# Encoder-Decoder Embedding Comparison — CUB-200

Joint loss: alpha*MSE + (1-alpha)*CrossEntropy  |  MLP classifier trained on frozen embeddings for 30 epochs.

## Results

| Embedding      | Augmented | Dim | Top-1 (%) | Top-5 (%) | Best Epoch |
|----------------|-----------|-----|-----------|-----------|------------|
| 512            | Yes       | 512 |     59.58 |     85.90 |         28 |
| 512_noaug      | No        | 512 |     62.29 |     88.45 |          6 |
| 256            | Yes       | 256 |     58.11 |     86.38 |          5 |
| 256_noaug      | No        | 256 |     60.75 |     87.54 |          7 |
| pca            | —         | 446 |     59.73 |     86.83 |         16 |

## Augmentation Effect (aug − noaug)

| Dim | Delta Top-1 | Delta Top-5 |
|-----|-------------|-------------|
| 512 | -2.71        | -2.55        |
| 256 | -2.64        | -1.16        |

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