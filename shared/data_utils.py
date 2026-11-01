"""Shared data helpers for the CV final project (all three slots).

Canonical splits (same for every method, per the project brief):
  train: 45,000 images | val: 5,000 images | test: 10,000 images
The train/val split is stratified (4500/500 per class) with seed 0,
so every slot sees identical data.
"""
import os
import pickle

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                        "cifar-10-batches-py")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
SEED = 0


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10_raw(data_dir=DATA_DIR):
    """Raw uint8 images, (N, 32, 32, 3) RGB + label arrays."""
    X_list, y_list = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        X_list.append(d[b"data"])
        y_list.append(d[b"labels"])
    d = _unpickle(os.path.join(data_dir, "test_batch"))
    X_train = np.concatenate(X_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.concatenate(y_list)
    X_test = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(d[b"labels"])
    return X_train, y_train, X_test, y_test


def stratified_train_val_split(X, y, val_per_class=500, seed=SEED):
    """Stratified split -> (Xtr, ytr, Xva, yva). Deterministic."""
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        va_idx.append(idx[:val_per_class])
        tr_idx.append(idx[val_per_class:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def load_canonical_splits(data_dir=DATA_DIR):
    """(Xtr, ytr, Xva, yva, Xte, yte) as uint8 RGB arrays."""
    X_train, y_train, X_test, y_test = load_cifar10_raw(data_dir)
    Xtr, ytr, Xva, yva = stratified_train_val_split(X_train, y_train)
    return Xtr, ytr, Xva, yva, X_test, y_test
