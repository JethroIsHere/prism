# Prism — one dataset, five views

Five-way CIFAR-10 image classification comparison: color histogram + k-NN
(classical), HOG + SVM (classical), custom CNN (from scratch), fine-tuned
EfficientNet-B0 (pretrained), fine-tuned ResNet-18 (pretrained).
Computer Vision final project, West Visayas State University.

## Why "Prism"?

A prism takes one beam of light and splits it into separate views of the
same thing. This project does the same to CIFAR-10: one dataset, five
methods looking at it, each answering a different question.

- **Color histogram + k-NN** — what can color alone tell you? (33.40%)
- **HOG + SVM** — what can shape alone tell you? (52.06%)
- **Custom CNN** — what can a network learn from scratch? (85.22%)
- **EfficientNet-B0** — what does transfer learning buy, efficiently?
  (79.88% at CIFAR-native 32×32 input)
- **ResNet-18** — what does transfer learning buy at full resolution?
  (96.30% at 224×224 input)

The arc is the result: hand-crafted features cap out low (color < shape),
learning from scratch jumps the gap, and pretrained models win — with the
honest footnote that the two pretrained numbers were measured at different
input sizes (see the table notes). The best method overall is ResNet-18,
so it is the system default.

## Layout (one folder per member; Ron owns two slots)

```
cv-final-project/
├── ron-hog-svm/              # Ron — HOG + SVM Lead (classical, Slot A) ✓ done
│   ├── hog_svm.py            # from-scratch HOG + linear SVM (the method module)
│   ├── hog_predict.py        # HOGClassifier inference API (for Kurt's system)
│   ├── sanity_hog.py         # 10 synthetic-edge sanity checks
│   ├── train_hog_svm.py      # full training + evaluation
│   └── results/              # metrics.json, confusion_matrix.png, model.npz,
│                             # scaler.npz, TECHNICAL_NOTES.md, logs
├── ron-color-knn/            # Ron — Color histogram + k-NN (classical, Slot D) ✓ done
│   ├── color_knn.py          # 24-dim color histograms + chi-squared k-NN (NumPy)
│   ├── color_knn_predict.py  # ColorKNNClassifier inference API
│   ├── train_color_knn.py    # k sweep on val, test evaluated once
│   └── results/              # metrics.json, confusion_matrix.png, model.npz
├── ron-efficientnet/         # Ron — EfficientNet-B0 (pretrained, Slot E) ✓ done
│   ├── efficientnet_finetune.py  # fine-tuning module (mirrors the ResNet recipe)
│   ├── train_efficientnet.py     # training — Colab GPU at 224, or CPU at 32
│   └── results/              # metrics.json, confusion_matrix.png, model.pt,
│                             # training_curves.png, checkpoint.pt
├── duke-custom-cnn/          # Duke — Custom CNN Lead (from scratch, Slot B) ✓ done
│   ├── custom_cnn.py         # 4-block CNN in PyTorch (the method module)
│   ├── train_custom_cnn.py   # training — Duke runs this on Colab GPU
│   └── results/              # Duke's outputs go here (guide inside)
├── christine-resnet18/       # Christine — ResNet-18 Lead (pretrained, Slot C) ✓ done
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

## Canonical splits (same for all five slots, per the brief)

45,000 train / 5,000 val / 10,000 test — stratified, seed 0, defined once in
`shared/data_utils.py`. Slot A refits its final model on the full 50k train
after validating hyperparameters on val (standard practice; documented in its
technical notes).

## How to run your slot

**Everyone:** `pip install torch torchvision matplotlib` (Colab already has them).
Use a GPU runtime for slots B, C, and E (Runtime > Change runtime type > T4).

```bash
# Slot A (Ron) -- already done
python3 ron-hog-svm/sanity_hog.py      # 10 synthetic-edge checks
python3 ron-hog-svm/train_hog_svm.py   # full training + evaluation

# Slot D (Ron) -- already done, runs in minutes on CPU (NumPy only)
python3 ron-color-knn/train_color_knn.py

# Slot E (Ron) -- already done; on Colab GPU use the default 224px,
# on a CPU-only machine run with PRISM_INPUT_SIZE=32 (~3 h for 12 epochs;
# 224px costs ~134 h on CPU, measured). The script saves a checkpoint
# after every epoch and resumes from it if the run is interrupted.
PRISM_INPUT_SIZE=32 python3 ron-efficientnet/train_efficientnet.py

# Slot B (Duke)
python3 duke-custom-cnn/train_custom_cnn.py

# Slot C (Christine)
python3 christine-resnet18/train_resnet18.py
```

Each training script prints progress, then writes to its `results/<slot>/`
folder: `metrics.json` (same schema for all five — accuracy, macro F1,
per-class F1, ms/image, params, training time), `confusion_matrix.png`,
`training_curves.png` (slots B/C/E), the model file, and `training_log.txt`.

## Results (all five slots, final)

| Metric | A. HOG + SVM | B. Custom CNN | C. ResNet-18 | D. Color + k-NN | E. EfficientNet-B0 |
|---|---|---|---|---|---|
| Top-1 accuracy | 52.06% | 85.22% | **96.30%** | 33.40% | 79.88% |
| Macro F1 | 0.5116 | 0.8519 | **0.963** | 0.3289 | 0.7982 |
| Runtime | 0.13 ms/img (CPU) | 0.29 ms/img (CUDA) | 2.22 ms/img (CUDA) | 14.2 ms/img (CPU) | 5.13 ms/img (CPU) |
| Params | 0M (brief rule; 17,650 weights actually) | 0.39M | 11.18M | 0M | 4.02M |
| Training time | 0.8 min | 9.6 min | 30.4 min | 0.18 min | 192.0 min |

**Read the table honestly — three footnotes:**

1. **Different input sizes.** ResNet-18 was fine-tuned and evaluated at
   224×224; EfficientNet-B0 was fine-tuned and evaluated at CIFAR-native
   32×32 (224px would have taken ~134 hours on the CPU it ran on). So the
   96.30% vs 79.88% gap is not a pure architecture comparison — resolution
   is part of it. The notebook keeps the 224px recipe for a GPU rerun.
2. **Different runtime devices.** CNN and ResNet-18 timings are on CUDA;
   HOG, k-NN, and EfficientNet timings are on CPU. Do not compare the
   ms/image column across devices.
3. **Training time is wall-clock.** EfficientNet's 192 min includes a
   mid-run machine restart it survived via per-epoch checkpoints.

**The story the numbers tell:** color alone is a weak signal (33.40% —
the honest baseline); shape beats color (52.06%); learning features from
scratch beats hand-crafting them (85.22%); pretrained weights win outright
(96.30%). The one class every method struggles with is the cat — lowest
per-class F1 for HOG (0.30), k-NN, and EfficientNet (0.63) alike.

## For the report / Q&A

- Each method module is heavily commented — read it until you can explain
  every line. The brief requires this in the Q&A.
- AI assistance must be disclosed in the contribution statement.
- Slot A's honest story is in `ron-hog-svm/results/TECHNICAL_NOTES.md`:
  strong on rigid classes (truck/auto/ship), weak on deformable animals
  (cat/bird/dog). Slots B–E should write the same kind of notes.
- The system's default method is ResNet-18 (best accuracy), with the
  reason stated in the app: highest top-1 accuracy and macro F1 on the
  shared 10k test set.
