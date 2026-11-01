# Slot C — ResNet-18 Fine-Tuning (Christine)

Your template is ready. Two files to read, one command to run.

## Read first (until you can explain every line — the Q&A requires it)

- `methods/resnet18_finetune.py` — loads ImageNet-pretrained ResNet-18,
  swaps the head for 10 CIFAR-10 classes, fine-tunes everything with Adam
  at a small lr (1e-4: the pretrained filters are already good, so we nudge
  them). Includes the residual-connection explanation for your report.
- `scripts/train_resnet18.py` — the pipeline: canonical 45k/5k/10k splits,
  resize to 224 + ImageNet normalization (**this is the "correct
  preprocessing" rubric point**), 12 epochs, then one final test evaluation.

## Run (Colab, GPU runtime)

```bash
python3 scripts/train_resnet18.py
```

Expected: ~20-40 min on a T4 GPU, test accuracy roughly 88-93%.
First run downloads ImageNet weights (~45MB) automatically.
Your outputs land here: `metrics.json`, `confusion_matrix.png`,
`training_curves.png`, `model.pt`, `training_log.txt`.

## Your rubric points

- **Correct preprocessing and output format, same splits**: handled by the
  script (resize224 + ImageNet norm, canonical splits) — understand *why*.
- Write a `TECHNICAL_NOTES.md` like Slot A's: what transfer learning is,
  why the small lr, results, and an honest paragraph on where ResNet-18
  wins/loses vs the other two (expect it to fix HOG's animal problem —
  check the per-class F1s and say so with numbers).
