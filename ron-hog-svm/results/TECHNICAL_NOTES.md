# HOG + Linear SVM — Technical Notes (Slot A, classical)

For Q&A prep: everything below is implemented from scratch in
`ron-hog-svm/hog_svm.py` (NumPy only). Be able to explain every step.

## HOG descriptor (per image, 32x32 RGB -> 1764 numbers)

1. **Grayscale**: `0.299R + 0.587G + 0.114B` (luma weights).
2. **Gradients**: central differences, `gx = img[x+1] - img[x-1]`,
   same for `gy`. Borders stay zero.
3. **Magnitude/orientation**: `mag = hypot(gx, gy)`,
   `ori = arctan2(gy, gx) mod 180` — *unsigned*: an edge and its
   photographic negative count as the same orientation.
4. **Cell histograms**: image split into 4x4 px cells (8x8 grid =
   64 cells). Each pixel votes its magnitude into 9 orientation bins
   (20 deg each) with **soft voting** — the vote is split linearly
   between the two nearest bins (this is the Dalal & Triggs trilinear
   interpolation, without the spatial part).
5. **Block normalization (L2-Hys)**: 2x2-cell blocks, stride 1 cell
   -> 7x7 = 49 blocks. Each 36-dim block vector is L2-normalized,
   clipped at 0.2, and renormalized. This kills contrast/lighting
   variation while keeping every value in [-1, 1].
6. Concatenate all blocks: 49 x 36 = **1764 features**.

Why 4x4 cells instead of the classic 8x8: on 32x32 images, 8x8 cells
give only a 4x4 spatial grid — too coarse. The choice was made honestly
on the 5,000-image validation split (canonical 45k/5k/10k split shared by
all three slots): 52.82% (4x4) vs 48.08% (8x8). The final model was then
refit on the full 50k training set and evaluated once on the test set:
**52.06%**.

## Linear SVM (one-vs-rest, SGD on the hinge loss)

- 10 binary classifiers; label for class c is +1 if the image is class
  c, else -1.
- Objective per class: `sum max(0, 1 - y*(x.w + b)) + (reg/2)*||w||^2`.
- **Subgradient** (used when the margin is violated, i.e. `y*(x.w+b) < 1`):
  `dW = reg*W - mean(x_outer(y))`, `db = -mean(y)`. Otherwise only the
  `reg*W` shrinkage applies.
- Minibatch SGD: batch 256, lr 0.1 with x0.92 decay per epoch,
  25 epochs, reg=1e-3. Features standardized (zero mean, unit variance)
  — required, or the hinge loss sees wildly different scales.
- Predict: `argmax` over the 10 decision scores.

## Results (CIFAR-10 test, 10k images)

| Metric | Value |
|---|---|
| Top-1 accuracy | **52.06%** |
| Macro F1 | **0.5116** |
| Runtime | **0.13 ms/image** (CPU) |
| Params | 0 M (classical, per brief; 17,650 learned weights actually) |
| Training time | **0.8 min** |

## Honest analysis (for the report's discussion section)

HOG is a **rigid template matcher**: it records where edges sit on a
fixed spatial grid. It wins on classes with stable silhouettes —
truck (F1 0.66), automobile (0.66), ship (0.59), airplane (0.59) —
and collapses on **deformable animals** whose pose varies: cat (0.30),
bird (0.38), dog (0.38). The confusion matrix shows dog -> cat (31%)
and deer -> frog/horse: when the edge layout doesn't match the rigid
template, the classifier has no fallback. This is exactly the weakness
learned features (slots B and C) are expected to fix — and the
strength is speed: 0.13 ms/image with no GPU and under a minute of
training.

## Files

- `ron-hog-svm/hog_svm.py` — the module (HOG + SVM + metrics, no sklearn)
- `ron-hog-svm/sanity_hog.py` — 10 synthetic-edge checks (all pass)
- `ron-hog-svm/train_hog_svm.py` — full training/eval pipeline
- `ron-hog-svm/results/` — metrics.json, confusion_matrix.png/.npy,
  model.npz, scaler.npz, sanity_report.txt, training_log.txt
