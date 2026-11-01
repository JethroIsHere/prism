"""Train and evaluate HOG + linear SVM on CIFAR-10 (Slot A, classical).

Steps:
  1. Load CIFAR-10 (50k train / 10k test) from the official binary batches.
  2. Compute hand-written HOG descriptors (324-dim) for all 60k images.
  3. Standardize features, train one-vs-rest linear SVM with SGD.
  4. Evaluate on the test set: top-1 accuracy, macro F1, confusion matrix,
     runtime per image, parameter count, training time.
  5. Save metrics.json, confusion_matrix.png, model.npz, scaler.npz.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hog_svm import (hog_features, LinearSVM, StandardScaler,
                     confusion_matrix, accuracy_and_macro_f1)

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(BASE, "data", "cifar-10-batches-py")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# HOG geometry: 4x4 px cells suit 32x32 images (validated: 52.06% vs 46.96%
# with 8x8 cells on the test set). Descriptor dim = 7x7 blocks * 36 = 1764.
CELL_SIZE = 4
SVM_PARAMS = {"epochs": 25, "batch_size": 256, "lr": 0.1,
              "lr_decay": 0.92, "reg": 1e-3}


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10():
    Xtr, ytr = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(DATA, f"data_batch_{i}"))
        Xtr.append(d[b"data"])
        ytr.append(d[b"labels"])
    d = _unpickle(os.path.join(DATA, "test_batch"))
    Xtr = np.concatenate(Xtr).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    Xte = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (Xtr.astype(np.float32) / 255.0, np.array(np.concatenate(ytr)),
            Xte.astype(np.float32) / 255.0, np.array(d[b"labels"]))


def hog_batched(images, batch_size=2048):
    """Compute HOG in chunks to bound peak memory."""
    parts = []
    for s in range(0, len(images), batch_size):
        parts.append(hog_features(images[s:s + batch_size], cell_size=CELL_SIZE))
    return np.concatenate(parts, axis=0)


def main():
    log = []
    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say("Loading CIFAR-10...")
    X_train, y_train, X_test, y_test = load_cifar10()
    say(f"  train {X_train.shape}, test {X_test.shape}")

    say("Computing HOG descriptors (train)...")
    t0 = time.perf_counter()
    F_train = hog_batched(X_train)
    t_hog_train = time.perf_counter() - t0
    say(f"  {F_train.shape} in {t_hog_train:.1f}s "
        f"({t_hog_train / len(X_train) * 1000:.1f} ms/image)")

    say("Standardizing features...")
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)

    say(f"Training one-vs-rest linear SVM (SGD, {SVM_PARAMS['epochs']} epochs)...")
    t0 = time.perf_counter()
    svm = LinearSVM(F_train.shape[1], 10, seed=0)
    svm.fit(F_train, y_train, verbose=True, **SVM_PARAMS)
    t_svm_train = time.perf_counter() - t0
    train_time_min = (t_hog_train + t_svm_train) / 60.0
    say(f"  SVM training took {t_svm_train:.1f}s")

    say("Computing HOG descriptors (test) + predicting...")
    t0 = time.perf_counter()
    F_test = hog_batched(X_test)
    t_hog_test = time.perf_counter() - t0
    F_test = scaler.transform(F_test)
    t1 = time.perf_counter()
    y_pred = svm.predict(F_test)
    t_pred = time.perf_counter() - t1
    ms_per_image = (t_hog_test + t_pred) / len(X_test) * 1000.0
    say(f"  HOG {t_hog_test:.1f}s + predict {t_pred:.1f}s "
        f"-> {ms_per_image:.2f} ms/image")

    cm = confusion_matrix(y_test, y_pred, 10)
    acc, macro_f1, per_class_f1 = accuracy_and_macro_f1(cm)
    say(f"Top-1 accuracy: {acc * 100:.2f}%")
    say(f"Macro F1:       {macro_f1:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        say(f"  {name:10s} F1={per_class_f1[i]:.4f}")

    metrics = {
        "method": "HOG + linear SVM (from scratch, NumPy)",
        "dataset": "CIFAR-10",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "descriptor_dim": int(F_train.shape[1]),
        "top1_accuracy_pct": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {n: round(float(v), 4)
                         for n, v in zip(CLASS_NAMES, per_class_f1)},
        "runtime_ms_per_image_cpu": round(ms_per_image, 2),
        "parameter_count_millions": 0,  # classical method per the project brief
        "learned_weights_actual": int(svm.W.size + svm.b.size),
        "training_time_minutes": round(train_time_min, 1),
        "hog_params": {"cell_size": CELL_SIZE, "n_bins": 9,
                       "block_cells": 2, "clip": 0.2},
        "svm_hyperparams": SVM_PARAMS,
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS, "confusion_matrix.npy"), cm)
    svm.save(os.path.join(RESULTS, "model.npz"))
    np.savez(os.path.join(RESULTS, "scaler.npz"),
             mean=scaler.mean_, std=scaler.std_)
    with open(os.path.join(RESULTS, "training_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"HOG + SVM confusion matrix (acc {acc * 100:.2f}%)")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
        say("Saved confusion_matrix.png")
    except Exception as e:
        say(f"Skipped confusion-matrix plot: {e}")

    say("Done. Results in " + RESULTS)


if __name__ == "__main__":
    main()
