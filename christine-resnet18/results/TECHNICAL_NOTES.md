# ResNet-18 — Technical Notes (Slot C, pretrained)

For Q&A preparation: the model and training pipeline are in `christine-resnet18/resnet18_finetune.py`, with the run recorded in `christine-resnet18/train_resnet18.ipynb`. Be ready to explain each step.

## Model and transfer learning

1. **Start with ImageNet weights:** ResNet-18 has learned general visual features such as edges, textures, and shapes from ImageNet.
2. **Replace the classification head:** the original final layer predicts 1,000 ImageNet classes. It is replaced with a layer that outputs scores for the 10 CIFAR-10 classes.
3. **Fine-tune the network:** all 11,181,642 parameters are trainable. Training updates the pretrained feature layers and the new classifier head for CIFAR-10.
4. **Residual connections:** each residual block learns a change to its input, then adds the original input back: `output = ReLU(F(x) + x)`. When a block changes the feature-map size or channel count, a projection adjusts the shortcut so the two paths can be added. This helps information and gradients pass through the network.
5. **Predict:** the final layer produces 10 scores; the class with the highest score is the prediction.

## Data and preprocessing

- **Dataset split:** canonical, stratified CIFAR-10 split with seed 0: 45,000 training images, 5,000 validation images, and 10,000 test images.
- **Resize:** CIFAR-10 images are 32×32 RGB and are resized to 224×224 for ResNet-18.
- **Normalize:** each RGB channel is standardized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. This matches the scale expected by the pretrained weights.
- **Training augmentation:** random horizontal flips are applied during training only. Validation and test images use resize and normalization without random augmentation.

## Fine-tuning

- Optimizer: Adam
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Scheduler: cosine
- Batch size: 64
- Epochs: 12
- Model selection: the best validation checkpoint is retained; test results are not used to choose settings.
- Hardware: Colab T4 GPU

The learning rate is small because the pretrained layers already contain useful visual features. Small updates adapt those features to CIFAR-10 while reducing the chance of disrupting them too quickly.

## Results (CIFAR-10 test, 10,000 images)

| Metric | Value |
|---|---:|
| Top-1 accuracy | **96.30%** |
| Macro F1 | **0.9630** |
| Runtime | **2.22 ms/image** (CUDA; notebook timing) |
| Trainable parameters | **11.18 M** |
| Training time | **30.4 min** |

The reported runtime uses the notebook's test-loader timing loop. It includes loading and transferring batches as well as model computation, so it is not a pure GPU-only forward-pass measurement.

## Per-class test F1

| CIFAR-10 class | ResNet-18 F1 |
|---|---:|
| Airplane | 0.9725 |
| Automobile | 0.9786 |
| Bird | 0.9565 |
| Cat | 0.9219 |
| Deer | 0.9625 |
| Dog | 0.9333 |
| Frog | 0.9786 |
| Horse | 0.9764 |
| Ship | 0.9765 |
| Truck | 0.9729 |

The mean of these class scores is 0.9630, matching the reported macro F1.

## Honest analysis

ResNet-18 reached 96.30% test accuracy and 0.9630 macro F1, compared with HOG + SVM's 52.06% and 0.5116. That is a difference of **44.24 percentage points** in accuracy and **0.4514** in macro F1.

The largest practical difference is on the animal classes that were difficult for HOG. Using the rounded HOG scores in Slot A's notes:

| Class | HOG + SVM F1 | ResNet-18 F1 | Approx. increase |
|---|---:|---:|---:|
| Bird | 0.38 | 0.9565 | +0.5765 |
| Cat | 0.30 | 0.9219 | +0.6219 |
| Dog | 0.38 | 0.9333 | +0.5533 |

The HOG comparison values are rounded, so the increases are approximate. ResNet-18's learned features handle variation in animal appearance and pose much better than HOG's fixed spatial edge histograms. The tradeoff is compute: this run took 30.4 minutes on a T4 and uses 11.18 million trainable parameters, while HOG + SVM trained in under a minute on CPU.

## Files

- `christine-resnet18/resnet18_finetune.py` — model setup, training, evaluation, and plotting helpers
- `christine-resnet18/train_resnet18.py` — standalone training pipeline
- `christine-resnet18/train_resnet18.ipynb` — Colab training run
- `christine-resnet18/results/` — `metrics.json`, `confusion_matrix.png`, `confusion_matrix.npy`, `training_curves.png`, `model.pt`, and `training_log.txt`

