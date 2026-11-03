# Slot C — ResNet-18 Fine-Tuning (Christine)

Your template is ready. Two files to read, one notebook to run.

## Read first (until you can explain every line — the Q&A requires it)

- `christine-resnet18/resnet18_finetune.py` — loads ImageNet-pretrained ResNet-18,
  swaps the head for 10 CIFAR-10 classes, fine-tunes everything with Adam
  at a small lr (1e-4: the pretrained filters are already good, so we nudge
  them). Includes the residual-connection explanation for your report.
- `shared/cnn_resnet_recipes.md` — lab-distilled tips for your slot.

## Run (Colab, GPU runtime)

Open **`train_resnet18.ipynb`** in Colab (File → Upload notebook),
set Runtime → GPU, and run top to bottom. It clones the repo, downloads
CIFAR-10 automatically, fine-tunes 12 epochs, and saves everything here:
`metrics.json`, `confusion_matrix.png`, `training_curves.png`, `model.pt`,
`training_log.txt`.

(`train_resnet18.py` is the same pipeline as a script, if you prefer it.)

Expected: ~20-40 min on a T4 GPU, test accuracy roughly 88-93%.
First run downloads ImageNet weights (~45MB) automatically.

## Your rubric points

- **Correct preprocessing and output format, same splits**: handled by the
  script (resize224 + ImageNet norm, canonical splits) — understand *why*.
- Write a `TECHNICAL_NOTES.md` like Slot A's: what transfer learning is,
  why the small lr, results, and an honest paragraph on where ResNet-18
  wins/loses vs the other two (expect it to fix HOG's animal problem —
  check the per-class F1s and say so with numbers).
