"""Color histogram + k-NN for CIFAR-10.

Slot D (classical, second) of the Computer Vision final project.
Hand-written in NumPy: per-channel color histograms + chi-squared
k-nearest-neighbours voting. No sklearn, no library call for the core
algorithm -- you must be able to explain every line in the Q&A.

Why this works (the one-paragraph story for the report): a color
histogram throws away all spatial layout and keeps only "how much of each
colour is in the picture". That is enough to tell a blue sky + grey hull
(ship, airplane) from green foliage (frog, deer), but it cannot tell a cat
from a dog -- same colours, different shapes. k-NN then classifies a new
image by the majority label of its k closest training histograms, using
chi-squared distance, the standard way to compare histograms: differences
in rare colours count more than differences in common ones.

Features: 8 bins per channel x 3 channels = 24 numbers per image,
L1-normalised per channel (each channel sums to 1).
"""
import numpy as np

BINS_PER_CHANNEL = 8
FEATURE_DIM = 3 * BINS_PER_CHANNEL


def color_histograms(images, bins_per_channel=BINS_PER_CHANNEL):
    """(N, 32, 32, 3) uint8 or float [0, 1] -> (N, 3*bins) float32.

    Each channel is quantised into `bins_per_channel` equal bins and counted
    with np.bincount (a plain counter, not a learned step). Each channel's
    histogram is normalised to sum to 1, so images are compared by colour
    proportions, not by brightness.
    """
    images = np.asarray(images)
    if images.dtype != np.uint8:
        images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    n = len(images)
    feats = np.zeros((n, 3 * bins_per_channel), dtype=np.float32)
    for c in range(3):
        # Bin index 0..bins-1 for every pixel in this channel.
        q = (images[:, :, :, c].astype(np.int32) * bins_per_channel) // 256
        q = np.clip(q, 0, bins_per_channel - 1)
        # Count bins per image: bincount over (image, bin) pairs.
        flat = (np.arange(n)[:, None] * bins_per_channel
                + q.reshape(n, -1))
        counts = np.bincount(flat.ravel(),
                             minlength=n * bins_per_channel)
        counts = counts.reshape(n, bins_per_channel).astype(np.float32)
        feats[:, c * bins_per_channel:(c + 1) * bins_per_channel] = counts
    # L1-normalise each channel so it sums to 1.
    feats = feats.reshape(n, 3, bins_per_channel)
    feats = feats / np.maximum(feats.sum(axis=2, keepdims=True), 1e-8)
    return feats.reshape(n, -1)


def _chi2_block(a, b):
    """Chi-squared distances between rows of a (m, d) and b (n, d).

    d(a, b) = sum_d (a_d - b_d)^2 / (a_d + b_d + eps). Computed one feature
    dimension at a time so the (m, n) result is the only large array.
    """
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for d in range(a.shape[1]):
        diff = a[:, d:d + 1] - b[None, :, d]
        out += (diff * diff) / (a[:, d:d + 1] + b[None, :, d] + 1e-8)
    return out


def knn_search(train_feats, query_feats, k, chunk=256):
    """Top-k nearest training rows for each query row.

    Returns (indices, distances), each (n_query, k), sorted nearest-first.
    Only the top-k are kept, so memory stays at chunk x n_train.
    """
    n_query = len(query_feats)
    idx_out = np.zeros((n_query, k), dtype=np.int64)
    dist_out = np.zeros((n_query, k), dtype=np.float32)
    for s in range(0, n_query, chunk):
        e = min(s + chunk, n_query)
        d = _chi2_block(query_feats[s:e], train_feats)
        # argpartition is O(n) -- much faster than a full sort.
        part = np.argpartition(d, k, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1)
        idx_out[s:e] = np.take_along_axis(part, order, axis=1)
        dist_out[s:e] = np.take_along_axis(part_d, order, axis=1)
    return idx_out, dist_out


def majority_vote(neighbor_labels, k):
    """Majority label of the first k neighbours per query row.

    Ties are broken by the smallest class id (deterministic).
    """
    n, _ = neighbor_labels.shape
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        counts = np.bincount(neighbor_labels[i, :k], minlength=10)
        preds[i] = int(np.argmax(counts))
    return preds


class KNearestNeighbors:
    """Lazy k-NN classifier: fit() just stores the training histograms."""

    def __init__(self, k=5, bins_per_channel=BINS_PER_CHANNEL):
        self.k = k
        self.bins_per_channel = bins_per_channel
        self.train_feats = None
        self.train_labels = None

    def fit(self, images, labels):
        self.train_feats = color_histograms(
            images, self.bins_per_channel)
        self.train_labels = np.asarray(labels)
        return self

    def _neighbors(self, images, k=None):
        feats = color_histograms(images, self.bins_per_channel)
        idx, _ = knn_search(self.train_feats, feats,
                            k if k is not None else self.k)
        return idx

    def predict(self, images, k=None):
        idx = self._neighbors(images, k)
        return majority_vote(self.train_labels[idx],
                             k if k is not None else self.k)

    def predict_proba(self, images, k=None):
        """Vote share per class -- used as decision scores by the system."""
        k = k if k is not None else self.k
        idx = self._neighbors(images, k)
        labels = self.train_labels[idx]
        proba = np.zeros((len(images), 10), dtype=np.float32)
        for c in range(10):
            proba[:, c] = (labels == c).sum(axis=1) / k
        return proba

    def save(self, path):
        np.savez(path, feats=self.train_feats, labels=self.train_labels,
                 k=np.array(self.k),
                 bins=np.array(self.bins_per_channel))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        clf = cls(k=int(d["k"]), bins_per_channel=int(d["bins"]))
        clf.train_feats = d["feats"]
        clf.train_labels = d["labels"]
        return clf


# ---------------------------------------------------------------------------
# Metrics (self-contained; no sklearn dependency)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy_and_macro_f1(cm):
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp),
                          where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp),
                       where=true_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(tp.sum() / cm.sum()), float(f1.mean()), f1


def plot_confusion_matrix(cm, class_names, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
