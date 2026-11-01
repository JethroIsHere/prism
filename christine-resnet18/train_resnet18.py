"""Fine-tune ResNet-18 on CIFAR-10 (Slot C, pretrained, modern).

Steps:
  1. Load the canonical 45k/5k/10k splits (same as every other slot).
  2. Load ImageNet-pretrained ResNet-18, replace the head for 10 classes,
     fine-tune everything with Adam at a small lr; keep the best-val
     checkpoint.
  3. Plot training/validation curves.
  4. Evaluate once on the test set: top-1 accuracy, macro F1, confusion
     matrix, runtime per image, parameter count, training time.
  5. Save metrics.json, confusion_matrix.png, training_curves.png,
     model.pt, training_log.txt.

Run on Colab with a GPU: Runtime > Change runtime type > T4 GPU.
First run downloads the ImageNet weights (~45MB) automatically.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from resnet18_finetune import (build_model, count_parameters, fit, evaluate,
                               plot_curves, confusion_matrix,
                               accuracy_and_macro_f1, plot_confusion_matrix)
from data_utils import load_canonical_splits, CLASS_NAMES, SEED

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# ImageNet statistics -- the pretrained weights expect this normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EPOCHS = 12
BATCH_SIZE = 64
LR = 1e-4


class CifarDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def main():
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"Using device: {device}")

    say("Loading canonical CIFAR-10 splits (45k train / 5k val / 10k test)...")
    Xtr, ytr, Xva, yva, Xte, yte = load_canonical_splits()

    # ResNet-18 expects 224x224 inputs with ImageNet normalization.
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader = DataLoader(CifarDataset(Xtr, ytr, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(device == "cuda"))
    val_loader = DataLoader(CifarDataset(Xva, yva, eval_tf),
                            batch_size=256, num_workers=2)
    test_loader = DataLoader(CifarDataset(Xte, yte, eval_tf),
                             batch_size=256, num_workers=2)

    say("Loading ImageNet-pretrained ResNet-18...")
    model = build_model(num_classes=10, pretrained=True).to(device)
    n_params = count_parameters(model)
    say(f"ResNet-18: {n_params:,} trainable parameters "
        f"({n_params / 1e6:.2f}M)")

    say(f"Fine-tuning ({EPOCHS} epochs, Adam lr={LR})...")
    t0 = time.perf_counter()
    history = fit(model, train_loader, val_loader, device,
                  epochs=EPOCHS, lr=LR, verbose=True)
    train_time_min = (time.perf_counter() - t0) / 60.0
    say(f"Fine-tuning took {train_time_min:.1f} min")
    torch.save(model.state_dict(), os.path.join(RESULTS, "model.pt"))
    plot_curves(history, os.path.join(RESULTS, "training_curves.png"))
    say("Saved training_curves.png")

    say("Evaluating on the test set (once)...")
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, _ in test_loader:
            _ = model(images.to(device))
            if device == "cuda":
                torch.cuda.synchronize()
    ms_per_image = (time.perf_counter() - t0) / len(Xte) * 1000.0
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    cm = confusion_matrix(y_true, y_pred)
    acc, macro_f1, per_class_f1 = accuracy_and_macro_f1(cm)
    say(f"Top-1 accuracy: {acc * 100:.2f}%")
    say(f"Macro F1:       {macro_f1:.4f}")

    metrics = {
        "method": "ResNet-18 (ImageNet pretrained, fine-tuned)",
        "dataset": "CIFAR-10",
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "top1_accuracy_pct": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {n: round(float(v), 4)
                         for n, v in zip(CLASS_NAMES, per_class_f1)},
        "runtime_ms_per_image": round(ms_per_image, 2),
        "runtime_device": device,
        "parameter_count_millions": round(n_params / 1e6, 2),
        "training_time_minutes": round(train_time_min, 1),
        "hyperparams": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                        "lr": LR, "optimizer": "Adam",
                        "weight_decay": 1e-4, "scheduler": "cosine",
                        "preprocessing": "resize224+imagenet_norm",
                        "augmentation": "flip"},
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS, "confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, CLASS_NAMES,
                          f"ResNet-18 confusion matrix (acc {acc * 100:.2f}%)",
                          os.path.join(RESULTS, "confusion_matrix.png"))
    with open(os.path.join(RESULTS, "training_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    say("Done. Results in " + RESULTS)


if __name__ == "__main__":
    main()
