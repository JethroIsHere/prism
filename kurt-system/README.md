# Kurt — System Development & Evaluation Lead

## The system (required by the brief)

Build **one working application** that turns the comparison into something a
user can try. Requirements:

1. **Input**: the user uploads an image or video.
2. **Output**: results drawn on the input (for classification: the predicted
   label, ideally from all five methods side by side).
3. **Recommendation**: the system names your best method by default and
   explains why (confirmed: ResNet-18 is crowned the winner based on verified
   evaluation metrics — 96.30% top-1 accuracy and 0.9630 macro F1).
4. **Deployment**: Flask web application with a dark-mode brutalist crimson bento
   UI (HTML5, CSS3, Vanilla JavaScript, Chart.js).

It must solve a **real local problem** — frame it, e.g. "classify products /
plants / campus objects from a phone photo". (CIFAR-10 classes are the demo
stand-in; the framing is what makes it a system, not a script.)

## Installation & Setup

All application dependencies are defined in `requirements.txt`:
- `Flask==3.0.0` & `Werkzeug==3.0.1` (Web server and API request routing)
- `Pillow>=10.2.0` (Image upload handling and preprocessing)
- `numpy>=1.26.0` (Array operations, HOG feature extraction, and metric arrays)
- `torch>=2.0.0` & `torchvision>=0.15.0` (PyTorch inference for Custom CNN & ResNet-18)
- `matplotlib>=3.7.0` (Plotting and visualization utilities)

### Quick Start
```bash
# Navigate to the system directory
cd kurt-system

# Install dependencies
pip install -r requirements.txt

# Start the Flask development server (hosted at http://127.0.0.1:5000)
python app.py
```
> **Runtime Note:** Model checkpoints default to CPU execution (`map_location="cpu"`), ensuring seamless local demonstration without dedicated GPU hardware.

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

**Duke's & Christine's methods (fully integrated):**
Integrated directly via PyTorch state-dict loaders into `backend/prediction_service.py`:
- **Duke's Custom 4-Block CNN**: Loaded from `duke-custom-cnn/results/model.pt` into `CustomCNN(num_classes=10)` with CIFAR-10 normalization (`mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2023, 0.1994, 0.2010)`).
- **Christine's ResNet-18**: Loaded from `christine-resnet18/results/model.pt` into `build_model(num_classes=10, pretrained=False)` resized to 224×224 with ImageNet normalization (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).

All five models run live inference on image upload via `POST /api/predict`.

## Build order (status)

1. [x] Build the UI skeleton with dark-mode brutalist crimson bento styling.
2. [x] Plug in Ron's HOG classifier (`ron-hog-svm/results/`).
3. [x] Plug in Duke's Custom CNN and Christine's ResNet-18 (`results/*/model.pt`).
4. [x] Plug in Ron's Color Histogram + k-NN and EfficientNet-B0 (`results/*`).
5. [x] Wire dynamic `/api/metrics` consolidation, live multi-model prediction, and automated recommendation.
6. [x] Test end-to-end and record the demo.

## Evaluation consolidation (your other half)

Each slot's `results/*/metrics.json` uses the **same schema** (accuracy,
macro F1, per-class F1, ms/image, params, training time). Build the
comparison table from those five files — no reformatting needed:

| Method | Top-1 acc | Macro F1 | ms/image | Params (M) | Train time |
|---|---|---|---|---|---|
| Color + k-NN | 33.40% | 0.3289 | 14.2 (CPU) | 0 | 0.18 min |
| HOG + SVM | 52.06% | 0.5116 | 0.13 (CPU) | 0 | 0.8 min |
| Custom CNN | 85.22% | 0.8519 | 0.29 (CUDA) | 0.39 | 9.6 min |
| EfficientNet-B0 | 79.88% | 0.7982 | 5.13 (CPU) | 4.02 | 192.0 min |
| ResNet-18 | 96.30% | 0.9630 | 2.22 (CUDA) | 11.18 | 30.4 min |

The frontend dynamically fetches these metrics via `GET /api/metrics` to populate the comparison table and render the interactive Chart.js bar chart. An automated recommendation engine compares verified metrics and crowns ResNet-18 as the top performer, while the written strengths/limitations summary for Jazylle's report and slides is documented in the UI's Architectural Overview (`#about`) section.
