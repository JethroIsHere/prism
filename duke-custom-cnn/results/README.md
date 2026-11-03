# Slot B — Custom CNN (Duke)

Your template is ready. Two files to read, one notebook to run.

## Read first (until you can explain every line — the Q&A requires it)

- `methods/custom_cnn.py` — the 4-block CNN: Conv→BN→ReLU blocks,
  training loop, best-validation checkpointing, curves + metrics helpers.
- `shared/cnn_resnet_recipes.md` — lab-distilled tips for your slot.

## Run (Colab, GPU runtime)

Open **`train_custom_cnn.ipynb`** in Colab (File → Upload notebook),
set Runtime → GPU, and run top to bottom. It clones the repo, downloads
CIFAR-10 automatically, trains 30 epochs, and saves everything here:
`metrics.json`, `confusion_matrix.png`, `training_curves.png`, `model.pt`,
`training_log.txt`.

(`train_custom_cnn.py` is the same pipeline as a script, if you prefer it.)

Expected: ~30-60 min on a T4 GPU, test accuracy roughly 65-75%.

## Your rubric points

- **Own architecture**: the 4-block design is a starting point. If you change
  it (more blocks, dropout, different widths), note why in your report.
- **Training curves shown**: `training_curves.png` is generated for you.
- **Overfitting addressed**: augmentation + weight decay + best-*validation*
  checkpoint (never pick by test accuracy).
- Write a `TECHNICAL_NOTES.md` like Slot A's: architecture, hyperparams,
  results, and an honest paragraph on where the CNN wins/loses vs HOG.
