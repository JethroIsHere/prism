# Color Histogram + k-NN — Technical Notes (Slot D, classical)

For Q&A prep: everything below is implemented from scratch in
`ron-color-knn/color_knn.py` (NumPy only). Be able to explain every step.

## The idea

The weakest possible honest baseline: ignore shape entirely and classify
by color palette alone. If this does well, the dataset is color-separable
and the other methods are overkill. It does not do well — which is the
point of the slot.

## Feature: 24-dim color histogram (per image)

1. Each image is 32x32 RGB. For each channel (R, G, B) separately, count
   how many pixels fall into each of **8 bins** (0–31, 32–63, ..., 224–255).
2. Concatenate the three 8-bin histograms: **24 numbers**.
3. **L1-normalize** each histogram (divide by the total count, 1024
   pixels), so it is a distribution that sums to 1 per channel — image
   size and brightness no longer matter, only the color mix.

Why 8 bins and not 256: with 256 bins the histogram is mostly zeros
on a 32x32 image and every image looks different; 8 bins is coarse
enough that similar palettes match.

## Distance: chi-squared

Comparing two histograms `h` and `g`:

```
d(h, g) = sum over bins of (h - g)^2 / (h + g + 1e-10)
```

Chi-squared punishes differences in bins where both histograms have
mass and ignores bins where both are near zero — it behaves better than
Euclidean distance on distributions. The `1e-10` guards against a
divide-by-zero on empty bins.

The distance matrix is computed in **blocks** (1,000 test images at a
time against all 45,000 training images), so it never holds the full
10k x 45k matrix in memory.

## Classifier: k-NN majority vote

- Store every training histogram (45,000 x 24) — the stored data *is*
  the model, which is why `model.npz` is 4.7 MB.
- For a new image, find the k training images with the smallest
  chi-squared distance and take the **majority vote** of their labels.
- k was chosen honestly on the 5,000-image validation split (canonical
  45k/5k/10k split shared by all five slots): one shared neighbor
  search, then k swept over {1, 3, 5, 10, 20} -> **k = 20** (val
  32.96%). Larger k wins here because color features are noisy — a
  bigger vote smooths out single nearest neighbors that happen to share
  a palette by accident.
- "Training" is just extracting the training histograms: **0.18 min**.

## Results (CIFAR-10 test, 10k images)

| Metric | Value |
|---|---|
| Top-1 accuracy | **33.40%** |
| Macro F1 | **0.3289** |
| Runtime | **14.2 ms/image** (CPU) |
| Params | 0 M (classical, per brief) |
| Training time | **0.18 min** |

Per-class F1: airplane 0.46, frog 0.41, ship 0.38, truck 0.37,
automobile 0.37, deer 0.32, horse 0.28, bird 0.26, dog 0.24, cat 0.20.

## Honest analysis (for the report's discussion section)

Color alone is a **weak signal**: 33.40% beats random guessing (10%) by
a lot, but it is 19 points below HOG (52.06%) — that gap is exactly what
shape adds. The confusions prove it: truck -> automobile (30%),
ship -> automobile (20%), deer -> frog (25%), bird -> frog (18%).
These classes share palettes (grey road, blue sky, green background)
and color has no way to tell them apart. The only classes it does
passably on are the ones with a distinctive palette — airplane (0.46,
blue sky) and frog (0.41, green). Cat is the worst class here too
(0.20), as it is for every method in this project.

The other cost is speed: 14.2 ms/image, the slowest of the five
methods, because every prediction compares against all 45,000 stored
training vectors. No training, no learned weights — just a lookup.

## Files

- `ron-color-knn/color_knn.py` — the module (histograms + chi-squared k-NN, no sklearn)
- `ron-color-knn/color_knn_predict.py` — ColorKNNClassifier inference API (for Kurt's system)
- `ron-color-knn/train_color_knn.py` — k sweep on val + test evaluation
- `ron-color-knn/train_color_knn.ipynb` — executed notebook (all outputs visible)
- `ron-color-knn/results/` — metrics.json, confusion_matrix.png/.npy,
  model.npz, training_log.txt
