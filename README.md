# Prism — one dataset, three views

Three-way CIFAR-10 image classification comparison: HOG + SVM (classical),
custom CNN (from scratch), fine-tuned ResNet-18 (pretrained).
Computer Vision final project, West Visayas State University.

## Layout (one folder per member)

```
cv-final-project/
├── ron-hog-svm/              # Ron — HOG + SVM Lead (classical) ✓ done
│   ├── hog_svm.py            # from-scratch HOG + linear SVM (the method module)
│   ├── hog_predict.py        # HOGClassifier inference API (for Kurt's system)
│   ├── sanity_hog.py         # 10 synthetic-edge sanity checks
│   ├── train_hog_svm.py      # full training + evaluation
│   └── results/              # metrics.json, confusion_matrix.png, model.npz,
│                             # scaler.npz, TECHNICAL_NOTES.md, logs
├── duke-custom-cnn/          # Duke — Custom CNN Lead (from scratch)
│   ├── custom_cnn.py         # 4-block CNN in PyTorch (the method module)
│   ├── train_custom_cnn.py   # training — Duke runs this on Colab GPU
│   └── results/              # Duke's outputs go here (guide inside)
├── christine-resnet18/       # Christine — ResNet-18 Lead (pretrained)
│   ├── resnet18_finetune.py  # fine-tuning module
│   ├── train_resnet18.py     # training — Christine runs this on Colab GPU
│   └── results/              # Christine's outputs go here (guide inside)
├── kurt-system/              # Kurt — System Development & Evaluation Lead
│   └── README.md             # system requirements, integration APIs, table spec
├── jazylle-docs/             # Jazylle — Documentation & Presentation Lead
│   └── README.md             # report/slides/video checklist + inputs needed
├── shared/
│   ├── data_utils.py         # CIFAR-10 loader + canonical 45k/5k/10k splits
│   └── cnn_resnet_recipes.md # lab-distilled recipes for Duke & Christine
├── data/
│   └── cifar-10-batches-py/  # downloaded once; everyone reuses it.
│                              # Get it with:
│                              # curl -sSL -o - https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
│                              #   | tar -xz --no-same-owner -C data/
└── README.md
```

## Canonical splits (same for all three slots, per the brief)

45,000 train / 5,000 val / 10,000 test — stratified, seed 0, defined once in
`scripts/data_utils.py`. Slot A refits its final model on the full 50k train
after validating hyperparameters on val (standard practice; documented in its
technical notes).

## How to run your slot

**Everyone:** `pip install torch torchvision matplotlib` (Colab already has them).
Use a GPU runtime for slots B and C (Runtime > Change runtime type > T4).

```bash
# Slot A (Ron) -- already done
python3 scripts/sanity_hog.py      # 10 synthetic-edge checks
python3 scripts/train_hog_svm.py   # full training + evaluation

# Slot B (Duke)
python3 scripts/train_custom_cnn.py

# Slot C (Christine)
python3 scripts/train_resnet18.py
```

Each training script prints progress, then writes to its `results/<slot>/`
folder: `metrics.json` (same schema for all three — accuracy, macro F1,
per-class F1, ms/image, params, training time), `confusion_matrix.png`,
`training_curves.png` (slots B/C), the model file, and `training_log.txt`.

## Results so far (Slot A, final)

| Metric | HOG + SVM |
|---|---|
| Top-1 accuracy | 52.06% |
| Macro F1 | 0.5116 |
| Runtime | 0.13 ms/image (CPU) |
| Params | 0M (brief rule; 17,650 weights actually) |
| Training time | 0.8 min |

## For the report / Q&A

- Each method module is heavily commented — read it until you can explain
  every line. The brief requires this in the Q&A.
- AI assistance must be disclosed in the contribution statement.
- Slot A's honest story is in `results/hog_svm/TECHNICAL_NOTES.md`:
  strong on rigid classes (truck/auto/ship), weak on deformable animals
  (cat/bird/dog). Slots B and C should write the same kind of notes.
