# Method recipes — Duke (custom CNN) & Christine (ResNet-18)

Runnable templates are provided — `methods/custom_cnn.py`,
`methods/resnet18_finetune.py`, plus `scripts/train_custom_cnn.py` and
`scripts/train_resnet18.py`. They are starting points: read every line until
you can explain it (the Q&A requires this), adjust the architecture or
hyperparameters if you can justify it, and disclose AI assistance in the
group's contribution statement.

These notes distill Jethro's October deep-learning lab (ants-vs-bees)
plus standard CIFAR-10 practice.

## What the lab actually measured (245 train / 153 val)

- ResNet-18 **frozen** feature extractor: **95.42%** — the winner on 245 images.
- ResNet-18 **fine-tuned**: 92.81% → **93.46%** improved
  (Adam + dropout + stronger augmentation + gradient clipping + gentler LR 1e-4).
  The only model that improved in round 2.
- Custom 2D CNN: 71.90% → 70.59% (heavier regularization hurt on tiny data).

**Honest caveat:** on CIFAR-10's 45k training images, fine-tuning usually beats
frozen — the opposite of the 245-image lab result. The template fine-tunes
directly; optionally run a frozen-backbone baseline first as a sanity check
and report both.

## Duke — 4-block custom CNN (from scratch)

Implemented in `methods/custom_cnn.py`: 4 × [Conv → BatchNorm → ReLU],
channels 32→64→128→256, MaxPool after blocks 1–3, global average pooling,
one FC layer → 10 (~0.39M params). Clean to defend line by line in Q&A.

- Optimizer: **Adam, LR 1e-3**, batch 128, 30 epochs, weight decay 5e-4,
  cosine schedule. (From-scratch nets tolerate 1e-3 — unlike fine-tuning.)
- Augmentation (this is what moved the needle in the lab): random crop 32×32
  with padding 4 + random horizontal flip, train only. Normalize CIFAR-10
  mean (0.4914, 0.4822, 0.4465) / std (0.2023, 0.1994, 0.2010).
- Overfitting defenses: augmentation + weight decay + keep the
  best-*validation* checkpoint (never pick by test accuracy). Optional:
  dropout 0.3–0.5 before the FC layer, gradient clipping.
- `training_curves.png` is generated automatically — the report asks for it.

## Christine — ResNet-18 fine-tune

Implemented in `methods/resnet18_finetune.py`: ImageNet-pretrained ResNet-18,
final FC replaced with 10 outputs, everything fine-tuned.

- **Adam LR 1e-4** (the lab's gentler LR was the only gainer), batch 64,
  12 epochs, weight decay 1e-4, cosine schedule. Small LR matters: the
  pretrained filters are already good — nudge them, don't relearn them.
- Preprocessing (**"correct preprocessing" rubric point**): resize to 224,
  ImageNet normalization mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225).
  Train augmentation: horizontal flip.
- Expect ~88–93% test accuracy — this should fix HOG's animal problem.
  Check the per-class F1s and say so with numbers in the report.

## Shared evaluation protocol (the brief requires all of these)

Top-1 accuracy (%), macro F1, confusion matrix, runtime ms/image (state CPU
or GPU), parameter count in millions (0 for classical methods like HOG+SVM),
training time in minutes — all on the canonical **45k / 5k / 10k** CIFAR-10
split (`scripts/data_utils.py`, stratified, seed 0).

Reference numbers for context: HOG + SVM slot is done —
**52.06% top-1, macro F1 0.5116, 0.13 ms/image CPU, 0.8 min training**
(`results/hog_svm/`). Strong on rigid classes (truck/auto/ship), weak on
deformable animals — that gap is what your two methods are expected to close.
