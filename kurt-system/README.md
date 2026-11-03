# Kurt — System Development & Evaluation Lead

## The system (required by the brief)

Build **one working application** that turns the comparison into something a
user can try. Requirements:

1. **Input**: the user uploads an image or video.
2. **Output**: results drawn on the input (for classification: the predicted
   label, ideally from all three methods side by side).
3. **Recommendation**: the system names your best method by default and
   explains why (in our case: expected winner is ResNet-18 — confirm with
   the real numbers once all three slots are done).
4. **Deployment**: any application — Gradio, Streamlit, Flask, or even a
   simple desktop GUI. Pick what you can finish fastest.

It must solve a **real local problem** — frame it, e.g. "classify products /
plants / campus objects from a phone photo". (CIFAR-10 classes are the demo
stand-in; the framing is what makes it a system, not a script.)

## Integration APIs

**Ron's method (ready now):**
```python
import sys
sys.path.insert(0, "../ron-hog-svm")
from hog_predict import HOGClassifier, CLASSES

clf = HOGClassifier.from_results("../ron-hog-svm/results")
print(clf.predict_labels("photo.jpg"))  # e.g. ['truck']
```
`predict(images)` takes one image or a batch — a NumPy array, a PIL Image,
or a file path (any size, auto-resized to 32x32). `decision_scores(images)`
gives raw scores.

**Duke's & Christine's methods (when their training finishes):** ask them to
add a small wrapper mirroring the same interface — `predict(images)` returning
class ids and a `CLASSES` tuple in CIFAR-10 order — or write thin wrappers
yourself around their `results/*/model.pt` files.

## Build order (avoid the bottleneck)

1. Build the UI skeleton **now** with dummy/mock predictions.
2. Plug in Ron's HOG classifier (available today).
3. Plug in Duke's and Christine's models as each finishes — don't wait for
   both.
4. Test end-to-end and record the demo.

## Evaluation consolidation (your other half)

Each slot's `results/*/metrics.json` uses the **same schema** (accuracy,
macro F1, per-class F1, ms/image, params, training time). Build the
comparison table from those three files — no reformatting needed:

| | Top-1 acc | Macro F1 | ms/image | Params (M) | Train time |
|---|---|---|---|---|---|
| HOG + SVM | 52.06% | 0.5116 | 0.13 (CPU) | 0 | 0.8 min |
| Custom CNN | … | … | … | … | … |
| ResNet-18 | … | … | … | … | … |

Plus the grouped bar chart (accuracy per method) and a short written
summary of strengths/limitations per method — Jazylle needs these for
the report and slides.
