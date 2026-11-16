"""Color histogram + k-NN -- inference interface for the group system.

Owned by: Ron (Slot D, classical method, second).

Loads the trained artifacts from ``ron-color-knn/results/`` and exposes a
small, stable API that the application/system (Kurt's slot) can build
against::

    import sys

    sys.path.insert(0, "ron-color-knn")

    from color_knn_predict import ColorKNNClassifier

    clf = ColorKNNClassifier.from_results("ron-color-knn/results")

    labels = clf.predict(images)          # (N,) int class ids 0-9

    scores = clf.decision_scores(images)  # (N, 10) neighbour vote shares

``images`` may be one image ``(H, W, 3)`` or a batch ``(N, H, W, 3)`` --
a NumPy array (uint8 or float32), a PIL Image, or a file path
(auto-resized to 32x32 with bilinear interpolation).

Class ids follow ``CLASSES`` below (CIFAR-10 order).

The "model" (``model.npz``) is the 45k training histograms + labels with
the chosen k -- k-NN is lazy, it stores the training set and votes at
predict time. See ``ron-color-knn/results/metrics.json`` for the measured
accuracy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_knn import KNearestNeighbors, color_histograms, knn_search  # noqa: E402

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

_IMAGE_SIZE = 32


def _bilinear_resize(img, size):
    """Resize (H, W, 3) float image to (size, size, 3). NumPy only."""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    ys = (np.arange(size) + 0.5) * h / size - 0.5
    xs = (np.arange(size) + 0.5) * w / size - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None, None]   # (size, 1, 1)
    wx = (xs - x0)[None, :, None]   # (1, size, 1)
    top = img[np.ix_(y0, x0)] * (1 - wx) + img[np.ix_(y0, x1)] * wx
    bot = img[np.ix_(y1, x0)] * (1 - wx) + img[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def _to_batch(images):
    """Normalize input to (N, 32, 32, 3) float32 in [0, 255].

    Accepts a file path (str / PathLike, loaded via PIL), a PIL Image,
    or a NumPy array.
    """
    if isinstance(images, (str, os.PathLike)):
        try:
            from PIL import Image
        except ImportError:
            raise ValueError(
                "Pillow is required to load image file paths: pip install pillow")
        images = Image.open(images)
    if hasattr(images, "convert"):  # PIL Image
        images = np.asarray(images.convert("RGB"))
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"expected (H, W, 3) or (N, H, W, 3), got shape {arr.shape}")
    if arr.max() <= 1.0:
        arr = arr * 255.0  # also accept [0, 1] floats
    return np.stack([_bilinear_resize(im, _IMAGE_SIZE) for im in arr])


class ColorKNNClassifier:
    """Trained color-histogram + k-NN, ready for the application/system."""

    def __init__(self, knn):
        self.knn = knn

    @classmethod
    def from_results(cls, results_dir):
        """Load ``model.npz`` saved by train_color_knn.py."""
        knn = KNearestNeighbors.load(os.path.join(results_dir, "model.npz"))
        return cls(knn)

    def decision_scores(self, images):
        batch = _to_batch(images)
        feats = color_histograms(batch / 255.0,
                                 self.knn.bins_per_channel)
        idx, _ = knn_search(self.knn.train_feats, feats, self.knn.k)
        labels = self.knn.train_labels[idx]
        scores = np.zeros((len(batch), 10), dtype=np.float32)
        for c in range(10):
            scores[:, c] = (labels == c).sum(axis=1) / self.knn.k
        return scores

    def predict(self, images):
        return np.argmax(self.decision_scores(images), axis=1)

    def predict_labels(self, images):
        return [CLASSES[i] for i in self.predict(images)]


def main():
    results_dir, image_path = sys.argv[1], sys.argv[2]
    try:
        from PIL import Image
        img = Image.open(image_path)
    except ImportError:
        raise SystemExit("Pillow required for the CLI demo: pip install pillow")
    clf = ColorKNNClassifier.from_results(results_dir)
    scores = clf.decision_scores(img)[0]
    order = np.argsort(scores)[::-1]
    print(f"image: {image_path}")
    for i in order[:3]:
        print(f"  {CLASSES[i]:>10s}  vote_share={scores[i]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python ron-color-knn/color_knn_predict.py <results_dir> <image>")
    main()
