# Slot B — Custom CNN (Duke)

Your template is ready. Two files to read, one command to run.

## Read first (until you can explain every line — the Q&A requires it)

- `methods/custom_cnn.py` — the 4-block CNN: Conv→BN→ReLU blocks,
  training loop, best-validation checkpointing, curves + metrics helpers.
- `scripts/train_custom_cnn.py` — the pipeline: canonical 45k/5k/10k splits,
  augmentation (random crop + flip) on train only, Adam + cosine schedule,
  30 epochs, then one final test evaluation.

## Run (Colab, GPU runtime)

```bash
python3 scripts/train_custom_cnn.py
```

Expected: ~30-60 min on a T4 GPU, test accuracy roughly 65-75%.
Your outputs land here: `metrics.json`, `confusion_matrix.png`,
`training_curves.png`, `model.pt`, `training_log.txt`.

## Your rubric points

- **Own architecture**: the 4-block design is a starting point. If you change
  it (more blocks, dropout, different widths), note why in your report.
- **Training curves shown**: `training_curves.png` is generated for you.
- **Overfitting addressed**: augmentation + weight decay + best-*validation*
  checkpoint (never pick by test accuracy).
- Write a `TECHNICAL_NOTES.md` like Slot A's: architecture, hyperparams,
  results, and an honest paragraph on where the CNN wins/loses vs HOG.
