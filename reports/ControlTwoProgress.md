# Sprint 0 Progress Note — CUB-200 Transfer Learning Baseline

**Date:** 2026-05-07  
**Model:** ResNet-50 (linear probe — frozen backbone)  
**Dataset:** CUB-200-2011

---

## 1. Dataset Summary

| Split | Images | Classes | Min / Max per class |
|-------|--------|---------|---------------------|
| Train | 5,394  | 200 | 26 / 27 |
| Val   | 600    | 200 | 3 / 3   |
| Test  | 5,794  | 200 | 11 / 30 |

The official CUB-200-2011 dataset contains **11,788 images** across **200 fine-grained bird species**.
The official `train_test_split.txt` split was used without modification. A 10% validation set was
carved out of the training partition via stratified sampling (seed=42), yielding exactly 3 images per
class in validation and roughly 27 per class in training — very balanced.

---

## 2. Preprocessing and Augmentation

| Stage | Train | Val / Test |
|-------|-------|------------|
| Resize | Shorter edge → 256 px | Shorter edge → 256 px |
| Crop | Random 224×224 | Center 224×224 |
| Flip | Random horizontal flip | — |
| Normalize | ImageNet mean/std | ImageNet mean/std |

ImageNet normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].  
No colour jitter, rotation, or erasing in this sprint — richer augmentation is planned for Week 1.

---

## 3. Model and Training Configuration

**Architecture:** ResNet-50 with ImageNet-pretrained weights (`IMAGENET1K_V1`).  
The final FC layer (2048 → 1000) was replaced with a new linear layer (2048 → 200).  
All backbone parameters were **frozen**; only the 200-class head (2048×200 + 200 = 410,200 params)
was trained (linear-probe baseline).

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| LR schedule | Cosine annealing (T\_max = 10) |
| Loss | Cross-entropy |

Training ran on CUDA in ~13 minutes (~78 s/epoch).  
Best checkpoint selected by validation accuracy.

**Training curve:**

| Epoch | Train Loss | Train Acc | Val Acc | LR |
|-------|-----------|-----------|---------|-----|
| 1  | 4.351 | 14.8% | 37.0% | 9.76e-4 |
| 2  | 2.379 | 45.9% | 49.8% | 9.05e-4 |
| 3  | 1.655 | 61.7% | 51.8% | 7.94e-4 |
| 4  | 1.332 | 68.3% | 57.0% | 6.55e-4 |
| 5  | 1.094 | 74.2% | 55.8% | 5.00e-4 |
| 6  | 0.951 | 79.2% | 61.2% | 3.45e-4 |
| 7  | 0.824 | 83.5% | 62.0% | 2.06e-4 |
| 8  | 0.766 | 85.5% | 63.8% | 9.55e-5 |
| 9  | 0.731 | 86.8% | 64.0% | 2.45e-5 |
| 10 | 0.695 | 87.7% | **64.3%** | 0.00e+0 |

---

## 4. Final Test-Set Results

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | **63.57%** |
| **Top-5 Accuracy** | **89.13%** |

This is in line with the expected linear-probe ceiling for ResNet-50 on CUB-200 (~60–65% top-1
is typical in the literature for frozen ImageNet features).

**Best-performing classes (top-1 = 100%):**
- 101. White Pelican (20/20)
- 018. Spotted Catbird (15/15)
- 110. Geococcyx / Roadrunner (30/30)

**Worst-performing classes (top-1 < 25%):**
- 144. Common Tern — 20.0% (6/30)
- 043. Yellow-bellied Flycatcher — 20.7% (6/29)
- 142. Black Tern — 23.3% (7/30)

Full per-class breakdown: `reports/per_class_accuracy.csv`

---

## 5. Observations from Confusion Matrix and Sample Predictions

**1. Tern / Gull confusion cluster.**  
The confusion matrix shows a dense off-diagonal block among the tern and gull species (Common Tern,
Black Tern, Elegant Tern, California Gull). These birds share very similar plumage — pale grey back,
white underparts, black cap — and differ mainly in bill shape and size detail. A frozen linear probe
over global average-pooled features cannot capture these local discriminative cues. This is the
single biggest source of errors and is the primary motivation for full fine-tuning in Week 1.

**2. Flycatcher / Pewee confusion cluster.**  
Sayornis (phoebe), Western Wood-Pewee, and Yellow-bellied Flycatcher all score below 27%. These
small, plain brown–olive passerines are notoriously hard even for expert ornithologists, as the
species differ mainly in subtle vocalization and habitat context, neither of which the image
features capture. Even full fine-tuning is unlikely to push these classes above 60–70% without
part-based attention.

**3. Visually distinctive birds score near-perfectly.**  
Classes like White Pelican (large, entirely white), Roadrunner (distinctive long tail and crest),
Blue Jay (bold blue/white/black pattern), and Brown Pelican (large pouch) score ≥93% because
their global colour and shape statistics are sufficient for a linear classifier. The gap between
the best (~100%) and worst (~20%) classes — a 5× spread — confirms that the bottleneck is
fine-grained local features, not the backbone representation quality.

---

## 6. Planned Next Steps (Week 1)

1. **Unfreeze backbone and fine-tune end-to-end.** Set `model.freeze_backbone: false` in
   `configs/config.yaml`. Use differential learning rates: ~1e-4 for the backbone layers and
   ~1e-3 for the head. Expected gain: +10–15 pp top-1 accuracy.

2. **Richer augmentation.** Add `ColorJitter`, `RandomRotation(±15°)`, and `RandomErasing` to the
   training pipeline. These regularise the model and improve generalisation on the confusable
   species.

3. **Longer training schedule.** Fine-tuning typically needs 20–30 epochs with a warm-up phase.
   Switch to a linear warm-up (5 epochs) followed by cosine annealing.

4. **Stronger backbone.** Experiment with ResNet-101, EfficientNet-B4, or a Vision Transformer
   (ViT-B/16) to establish an upper-bound baseline before designing domain-specific modules.

5. **Part-based or attention module.** The confusion matrix clearly shows that local features
   (bill shape, wing pattern) are decisive for the hardest classes. A CBAM attention module or a
   bounding-box crop (CUB provides bounding boxes) could directly address this.
