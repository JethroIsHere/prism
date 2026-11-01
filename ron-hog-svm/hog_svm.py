"""HOG + linear SVM for CIFAR-10 image classification.

Slot A (classical) of the Computer Vision final project.
Implemented from scratch in NumPy -- no library call for the main steps
(HOG descriptor computation, SVM training). Only NumPy primitives
(arithmetic, hypot, arctan2, bincount, matmul) are used.

Pipeline:  RGB image -> grayscale -> gradients -> cell orientation
histograms (soft voting) -> block L2-Hys normalization -> 324-dim
descriptor -> standardized -> one-vs-rest linear SVM (hinge-loss SGD).
"""

import numpy as np


# ---------------------------------------------------------------------------
# HOG descriptor
# ---------------------------------------------------------------------------

def _rgb_to_grayscale(images):
    """(N, H, W, 3) -> (N, H, W) using the standard luma weights."""
    images = np.asarray(images, dtype=np.float32)
    return (0.299 * images[..., 0]
            + 0.587 * images[..., 1]
            + 0.114 * images[..., 2])


def _gradients(gray):
    """Central differences; borders stay zero. (N, H, W) -> (gx, gy)."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    return gx, gy


def hog_features(images, cell_size=8, n_bins=9, block_cells=2, clip=0.2):
    """Compute HOG descriptors for a batch of RGB images.

    images: (N, H, W, 3), values in [0, 1] (or any consistent scale).
    Returns: (N, D) float32, D = (#blocks) * block_cells^2 * n_bins.
             For 32x32 images with the defaults: 9 blocks * 36 = 324.
    """
    gray = _rgb_to_grayscale(images)
    gx, gy = _gradients(gray)
    magnitude = np.hypot(gx, gy)
    # Unsigned orientation in [0, 180): an edge and its negative are the same.
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0

    n, h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    bin_width = 180.0 / n_bins
    n_cells = cells_y * cells_x

    # --- soft voting: each pixel splits its magnitude between the two
    # --- nearest orientation bins (linear interpolation, Dalal & Triggs).
    b = orientation / bin_width
    b0 = np.floor(b).astype(np.int64) % n_bins
    b1 = (b0 + 1) % n_bins
    f1 = b - np.floor(b)
    f0 = 1.0 - f1

    rows = np.arange(h) // cell_size            # (H,)
    cols = np.arange(w) // cell_size            # (W,)
    cell_id = (rows[:, None] * cells_x + cols[None, :])[None, :, :]  # (1,H,W)

    idx0 = (cell_id * n_bins + b0).reshape(n, -1)
    idx1 = (cell_id * n_bins + b1).reshape(n, -1)
    wgt0 = (magnitude * f0).reshape(n, -1)
    wgt1 = (magnitude * f1).reshape(n, -1)

    # One bincount over the whole batch: offset each image's indices.
    offset = (np.arange(n) * n_cells * n_bins)[:, None]
    hist = np.bincount((idx0 + offset).ravel(), weights=wgt0.ravel(),
                       minlength=n * n_cells * n_bins)
    hist += np.bincount((idx1 + offset).ravel(), weights=wgt1.ravel(),
                        minlength=n * n_cells * n_bins)
    hist = hist.reshape(n, cells_y, cells_x, n_bins)

    # --- block normalization (L2-Hys): normalize, clip, renormalize.
    blocks_y = cells_y - block_cells + 1
    blocks_x = cells_x - block_cells + 1
    blocks = []
    for i in range(blocks_y):
        for j in range(blocks_x):
            blk = hist[:, i:i + block_cells, j:j + block_cells, :].reshape(n, -1)
            blk = blk / np.sqrt((blk ** 2).sum(axis=1, keepdims=True) + 1e-12)
            blk = np.minimum(blk, clip)
            blk = blk / np.sqrt((blk ** 2).sum(axis=1, keepdims=True) + 1e-12)
            blocks.append(blk)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def hog_dimension(image_h=32, image_w=32, cell_size=8, n_bins=9, block_cells=2):
    """Descriptor length for a given image size (no data needed)."""
    cy, cx = image_h // cell_size, image_w // cell_size
    return (cy - block_cells + 1) * (cx - block_cells + 1) * block_cells ** 2 * n_bins


# ---------------------------------------------------------------------------
# One-vs-rest linear SVM trained with SGD on the hinge loss
# ---------------------------------------------------------------------------

class StandardScaler:
    """Mean/std normalization fitted on the training features."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearSVM:
    """Multiclass linear SVM, one-vs-rest, minibatch SGD.

    Minimizes  sum_c max(0, 1 - y_c * (x . w_c + b_c)) + (reg/2) * ||W||^2
    with y_c in {+1, -1}. Uses the subgradient of the hinge loss.
    """

    def __init__(self, n_features, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.n_classes = n_classes

    def fit(self, X, y, epochs=15, batch_size=256, lr=0.1,
            lr_decay=0.9, reg=1e-4, verbose=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n = X.shape[0]
        # One-hot labels in {+1, -1} for the one-vs-rest problems.
        Y = np.where(np.arange(self.n_classes)[None, :] == y[:, None], 1.0, -1.0)
        rng = np.random.default_rng(1)
        n_batches = int(np.ceil(n / batch_size))

        for epoch in range(epochs):
            eta = lr * (lr_decay ** epoch)
            perm = rng.permutation(n)
            total_loss = 0.0
            for bi in range(n_batches):
                idx = perm[bi * batch_size:(bi + 1) * batch_size]
                Xb, Yb = X[idx], Y[idx]
                m = Xb.shape[0]
                scores = Xb @ self.W + self.b          # (m, C)
                margins = Yb * scores
                violated = margins < 1.0              # only these pay hinge loss
                total_loss += np.maximum(0.0, 1.0 - margins).sum()
                # Subgradients of the regularized hinge objective.
                dW = reg * self.W - (Xb.T @ (Yb * violated)) / m
                db = -(Yb * violated).sum(axis=0) / m
                self.W -= eta * dW
                self.b -= eta * db
            total_loss = total_loss / n + 0.5 * reg * (self.W ** 2).sum()
            if verbose:
                print(f"  epoch {epoch + 1:2d}/{epochs}  lr={eta:.4f}  "
                      f"loss={total_loss:.4f}", flush=True)
        return self

    def decision(self, X):
        return np.asarray(X, dtype=np.float32) @ self.W + self.b

    def predict(self, X):
        return np.argmax(self.decision(X), axis=1)

    def save(self, path):
        np.savez(path, W=self.W, b=self.b)

    @classmethod
    def load(cls, path, n_classes):
        d = np.load(path)
        svm = cls(d["W"].shape[0], n_classes)
        svm.W, svm.b = d["W"], d["b"]
        return svm


# ---------------------------------------------------------------------------
# Metrics (hand-rolled so the module has no sklearn dependency)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy_and_macro_f1(cm):
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(tp.sum() / cm.sum()), float(f1.mean()), f1
