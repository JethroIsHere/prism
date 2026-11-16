"""Train / evaluate color histogram + k-NN on CIFAR-10 (Slot D, classical).

Steps:
  1. Load the canonical 45k/5k/10k splits (same as every other slot).
  2. Compute 24-dim color histograms for train/val/test.
  3. Choose k on the val split (k in {1, 3, 5, 10, 20}) from one shared
     neighbour search -- distances are computed once, not once per k.
  4. Evaluate once on the test set with the chosen k: top-1 accuracy,
     macro F1, confusion matrix, runtime per image, training time.
  5. Save metrics.json, confusion_matrix.png, model.npz (the training
     histograms + labels -- that IS the k-NN model), training_log.txt.

k-NN is lazy: "training" is just storing the training histograms, so the
reported training time is the feature-extraction time. Inference is the
slow part (every test image is compared against all 45k training images),
which is the honest efficiency story for this method.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from color_knn import (color_histograms, knn_search, majority_vote,
                       KNearestNeighbors, confusion_matrix,
                       accuracy_and_macro_f1, plot_confusion_matrix)
from data_utils import load_canonical_splits, CLASS_NAMES, SEED

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

K_CANDIDATES = [1, 3, 5, 10, 20]
K_MAX = max(K_CANDIDATES)


def main():
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    np.random.seed(SEED)

    say("Loading canonical CIFAR-10 splits (45k train / 5k val / 10k test)...")
    Xtr, ytr, Xva, yva, Xte, yte = load_canonical_splits()
    Xtr = Xtr.astype(np.float32) / 255.0
    Xva = Xva.astype(np.float32) / 255.0
    Xte = Xte.astype(np.float32) / 255.0

    say("Computing color histograms (8 bins/channel -> 24-dim)...")
    t0 = time.perf_counter()
    F_train = color_histograms(Xtr)
    t_feat_train = time.perf_counter() - t0
    say(f"  train features {F_train.shape} in {t_feat_train:.1f}s")
    F_val = color_histograms(Xva)
    F_test = color_histograms(Xte)

    # --- choose k on the val split (distances computed once) ---
    say("Searching neighbours for the val split (k up to 20)...")
    t0 = time.perf_counter()
    val_idx, _ = knn_search(F_train, F_val, K_MAX)
    say(f"  val search took {time.perf_counter() - t0:.1f}s")
    best_k, best_acc = None, -1.0
    for k in K_CANDIDATES:
        pred = majority_vote(ytr[val_idx], k)
        acc = float((pred == yva).mean())
        say(f"  k={k:2d}  val accuracy {acc * 100:.2f}%")
        if acc > best_acc:  # ties keep the smaller k (first wins)
            best_k, best_acc = k, acc
    say(f"Chosen k={best_k} (val accuracy {best_acc * 100:.2f}%)")

    clf = KNearestNeighbors(k=best_k)
    clf.train_feats, clf.train_labels = F_train, ytr

    # --- evaluate once on the test set ---
    say("Evaluating on the test set (once)...")
    t0 = time.perf_counter()
    test_idx, _ = knn_search(F_train, F_test, best_k)
    t_search = time.perf_counter() - t0
    y_pred = majority_vote(ytr[test_idx], best_k)
    ms_per_image = t_search / len(Xte) * 1000.0
    say(f"  {ms_per_image:.2f} ms/image (CPU, feature + neighbour search)")

    cm = confusion_matrix(yte, y_pred)
    acc, macro_f1, per_class_f1 = accuracy_and_macro_f1(cm)
    say(f"Top-1 accuracy: {acc * 100:.2f}%")
    say(f"Macro F1:       {macro_f1:.4f}")

    metrics = {
        "method": "Color histogram + k-NN (from scratch, NumPy)",
        "dataset": "CIFAR-10",
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "top1_accuracy_pct": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {n: round(float(v), 4)
                         for n, v in zip(CLASS_NAMES, per_class_f1)},
        "runtime_ms_per_image_cpu": round(ms_per_image, 2),
        "parameter_count_millions": 0,
        "training_time_minutes": round(t_feat_train / 60, 2),
        "histogram_params": {"bins_per_channel": 8, "feature_dim": 24,
                             "distance": "chi-squared"},
        "knn_params": {"k_candidates": K_CANDIDATES, "k_chosen": best_k,
                       "k_val_accuracy_pct": round(best_acc * 100, 2)},
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS, "confusion_matrix.npy"), cm)
    clf.save(os.path.join(RESULTS, "model.npz"))
    plot_confusion_matrix(cm, CLASS_NAMES,
                          f"Color hist + k-NN confusion matrix "
                          f"(acc {acc * 100:.2f}%)",
                          os.path.join(RESULTS, "confusion_matrix.png"))
    with open(os.path.join(RESULTS, "training_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    say("Saved: metrics.json, confusion_matrix.png/.npy, model.npz, "
        "training_log.txt")
    say("Done. Results in " + RESULTS)


if __name__ == "__main__":
    main()
