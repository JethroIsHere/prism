# Ron — Color histogram + k-NN (Slot D, classical, second)

Hand-written NumPy: 8-bins-per-channel color histograms (24-dim) + chi-squared
k-NN voting. The brief's classical list names this method directly
("color histogram + k-NN").

## Files

- `color_knn.py` — features, chi-squared search, KNearestNeighbors, metrics
- `train_color_knn.py` — run this (or the notebook) to train + evaluate
- `train_color_knn.ipynb` — same pipeline as a notebook, with outputs
- `color_knn_predict.py` — `ColorKNNClassifier` inference API for Kurt's system
- `results/` — metrics.json, confusion matrix, model.npz (the 45k training
  histograms + labels — that IS the k-NN model), training_log.txt

## How to run

```bash
python train_color_knn.py
```

NumPy only, CPU, a few minutes. k is chosen on the 5k val split from
{1, 3, 5, 10, 20}; the test set is evaluated once with the chosen k.

## For the system (Kurt)

```python
import sys
sys.path.insert(0, "ron-color-knn")
from color_knn_predict import ColorKNNClassifier

clf = ColorKNNClassifier.from_results("ron-color-knn/results")
print(clf.predict_labels("photo.jpg"))  # e.g. ['ship']
```

`decision_scores()` returns neighbour vote shares per class.
