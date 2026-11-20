# EfficientNet-B0 Fine-Tuning — Technical Notes (Slot E, pretrained)

For Q&A prep: everything below is implemented in
`ron-efficientnet/efficientnet_finetune.py` (PyTorch). Be able to explain
every step.

## The model

**EfficientNet-B0** (Tan & Le, 2019) — the smallest of the EfficientNet
family, designed by compound scaling: depth, width, and input
resolution are scaled together in a fixed ratio instead of one at a
time. The network is built from **MBConv blocks** (mobile inverted
bottleneck: expand -> depthwise conv -> squeeze-and-excitation channel
attention -> project), which is why it gets good accuracy per
parameter: **4.02M parameters** vs ResNet-18's 11.18M.

## Fine-tuning recipe (mirrors the ResNet-18 recipe, Slot C)

- Start from **ImageNet-pretrained weights**; replace the 1000-way
  classifier head with a 10-way linear head for CIFAR-10. The whole
  network is fine-tuned (no frozen backbone).
- Optimizer: **Adam, lr 1e-4, weight decay 1e-4**, cosine schedule.
- **12 epochs**, batch 64, random horizontal flip augmentation,
  ImageNet normalization.
- Split: canonical 45k train / 5k val / 10k test, shared by all five
  slots. Best-val checkpoint is saved and evaluated on test once.

## Input size: 32x32, and why (read this before the Q&A)

The notebook keeps the **224x224** recipe (the standard for
EfficientNet, and what a Colab GPU run should use). The run that
produced these numbers used **32x32** — CIFAR-native resolution —
because it ran on a CPU-only machine, and 224px was measured at
28.68 s/step (batch 32), which is **~134 hours for 12 epochs**.
At 32x32 the same run costs ~15 min/epoch.

This is recorded honestly in `metrics.json` (`"input_size": 32`), and
it means the ResNet-18 comparison (96.30% at 224px) is **not a pure
architecture comparison** — resolution is part of the gap. Do not
present 79.88% vs 96.30% as "EfficientNet is worse than ResNet";
present it as "EfficientNet-B0 at 32x32 on CPU vs ResNet-18 at 224x224
on GPU".

## The run (it survived two machine replacements)

- First attempt died ~45 min in when the VM was replaced, with nothing
  saved (results were only written at the end). Fix before restarting:
  **per-epoch checkpointing** (`results/checkpoint.pt`, atomic write),
  storing model/optimizer/scheduler/history/best state.
- Restarted, and the VM was replaced **again** at epoch 6 (val 0.7762).
  The checkpoint held — 96.4 min of training preserved — and the run
  resumed at epoch 7. Finished cleanly at 21:05.
- **Total wall-clock training time: 192.0 min**, including the resume.

Validation accuracy by epoch: 0.5948 -> 0.6798 -> 0.7260 -> 0.7494 ->
0.7644 -> 0.7762 -> 0.7862 -> 0.7792 -> 0.7910 -> 0.7898 -> 0.7858 ->
**0.7936** (best, epoch 12 — the final epoch). Train accuracy ended at
0.8956 vs val 0.7936: a ~10-point gap, i.e. mild overfitting, and the
saved model is the best-val checkpoint.

## Results (CIFAR-10 test, 10k images)

| Metric | Value |
|---|---|
| Top-1 accuracy | **79.88%** |
| Macro F1 | **0.7982** |
| Runtime | **5.13 ms/image** (CPU, 32x32 input) |
| Params | **4.02M** |
| Training time | **192.0 min** (wall-clock, incl. resume) |

Per-class F1: automobile 0.89, ship 0.88, truck 0.87, frog 0.85,
horse 0.83, airplane 0.82, deer 0.77, bird 0.74, dog 0.70, cat 0.63.

## Honest analysis (for the report's discussion section)

EfficientNet-B0 lands **between the from-scratch CNN (85.22%) and
HOG (52.06%)** — below the CNN, which looks wrong for a pretrained
model until you remember the input size: the CNN ran at 32x32 too, but
it was trained end-to-end for CIFAR, while EfficientNet's pretrained
features were learned at 224px and are being asked to work at 32px.
The efficiency story still holds: 4.02M params, 79.88% — the
accuracy-per-parameter is the best of the three neural methods.

The residual errors are the same hard pairs every method in this
project hits: dog -> cat (18%), cat -> dog (13%); everything else is
7% or less. Cat is the weakest class (0.63) here exactly as it is for
HOG (0.30) and k-NN (0.20) — a dataset-level difficulty, not a method
bug. At 224px on a GPU the number should close much of the gap to
ResNet; that rerun has not been done, so no number is claimed for it.

## Files

- `ron-efficientnet/efficientnet_finetune.py` — the module (model + training loop + metrics, per-epoch checkpoints)
- `ron-efficientnet/train_efficientnet.py` — training script (`PRISM_INPUT_SIZE` env knob, default 224)
- `ron-efficientnet/train_efficientnet.ipynb` — notebook (224px Colab GPU recipe)
- `ron-efficientnet/results/` — metrics.json, confusion_matrix.png/.npy,
  training_curves.png, model.pt, checkpoint.pt, training_log.txt,
  training_stdout.log
