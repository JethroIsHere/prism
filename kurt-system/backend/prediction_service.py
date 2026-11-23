import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from backend.model_registry import ModelRegistry

# Inject sibling directories into path to access teammates' code
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'ron-hog-svm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'duke-custom-cnn'))
sys.path.insert(0, os.path.join(BASE_DIR, 'christine-resnet18'))
sys.path.insert(0, os.path.join(BASE_DIR, 'ron-color-knn'))
sys.path.insert(0, os.path.join(BASE_DIR, 'ron-efficientnet'))

# Safely import Ron's Color Histogram + k-NN Wrapper
try:
    from color_knn_predict import ColorKNNClassifier, CLASSES as KNN_CLASSES
except ImportError:
    ColorKNNClassifier = None
    KNN_CLASSES = ()

# Safely import Ron's EfficientNet-B0 Builder
try:
    from efficientnet_finetune import build_model as build_efficientnet
except ImportError:
    build_efficientnet = None

# Safely import Ron's HOG Wrapper
try:
    from hog_predict import HOGClassifier
except ImportError:
    HOGClassifier = None

# Safely import Duke and Christine's models
try:
    from custom_cnn import CustomCNN
    from resnet18_finetune import build_model as build_resnet
except ImportError:
    CustomCNN = None
    build_resnet = None

class PredictionService:
    """
    Handles routing the uploaded image to the five different models.
    """
    def __init__(self):
        self.registry = ModelRegistry()
        self.base_dir = BASE_DIR
        
        # 1. Initialize Ron's HOG Model (Phase 4)
        self.hog_model = None
        if self.registry.check_hog_status()["available"] and HOGClassifier:
            self.hog_model = HOGClassifier.from_results(os.path.join(self.base_dir, 'ron-hog-svm', 'results'))
            # 1B. Initialize Ron's Color Histogram + k-NN Model (Slot D)
        self.knn_model = None
        if self.registry.check_knn_status()["available"] and ColorKNNClassifier:
            self.knn_model = ColorKNNClassifier.from_results(os.path.join(self.base_dir, 'ron-color-knn', 'results'))

        # 2. PyTorch Image Preprocessors (Custom CNN = 32x32, ResNet-18 = 224x224 ImageNet)
        self.cnn_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 3. PyTorch Model Initialization (Duke & Christine)
        self.device = torch.device('cpu') # Force CPU for local web backend
        self.cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.cnn_model = None
        if self.registry.check_cnn_status()["available"] and CustomCNN:
            self.cnn_model = CustomCNN(num_classes=10)
            cnn_path = os.path.join(self.base_dir, 'duke-custom-cnn', 'results', 'model.pt')
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.cnn_model.eval()

        self.resnet_model = None
        if self.registry.check_resnet_status()["available"] and build_resnet:
            # pretrained=False because we are loading Christine's custom trained weights
            self.resnet_model = build_resnet(num_classes=10, pretrained=False)
            resnet_path = os.path.join(self.base_dir, 'christine-resnet18', 'results', 'model.pt')
            self.resnet_model.load_state_dict(torch.load(resnet_path, map_location=self.device))
            self.resnet_model.eval()

        self.efficientnet_model = None
        if self.registry.check_efficientnet_status()["available"] and build_efficientnet:
            self.efficientnet_model = build_efficientnet(num_classes=10, pretrained=False)
            eff_path = os.path.join(self.base_dir, 'ron-efficientnet', 'results', 'model.pt')
            self.efficientnet_model.load_state_dict(torch.load(eff_path, map_location=self.device))
            self.efficientnet_model.eval()

    def process_image(self, image_file):
        statuses = self.registry.get_all_statuses()
        
        # Open the uploaded FileStorage object directly as a PIL RGB Image
        img = Image.open(image_file).convert('RGB')
        
        # --- HOG + SVM Inference ---
        hog_pred = None
        hog_conf = 0.0
        if self.hog_model:
            try:
                # Ron's wrapper auto-resizes and accepts PIL images
                labels = self.hog_model.predict_labels(img)
                hog_pred = labels[0] if isinstance(labels, list) else labels
                hog_conf = 0.85  # Standard SVMs lack softmax probabilities; mocking a high confidence
            except Exception as e:
                print(f"HOG Inference Error: {e}")

        # --- PyTorch Models Inference (CNN & ResNet Placeholder) ---
        # torch_img = self.torch_transform(img).unsqueeze(0)
        # --- PyTorch Models Inference ---
        cnn_pred, cnn_conf = None, 0.0
        resnet_pred, resnet_conf = None, 0.0

        # --- Color Histogram + k-NN Inference ---
        knn_pred = None
        knn_conf = 0.0
        if self.knn_model:
            try:
                scores = self.knn_model.decision_scores(img)[0]
                best_idx = int(scores.argmax())
                knn_pred = KNN_CLASSES[best_idx]
                knn_conf = float(scores[best_idx])
            except Exception as e:
                print(f"Color k-NN Inference Error: {e}")
        
        efficientnet_pred, efficientnet_conf = None, 0.0

        if self.cnn_model or self.resnet_model or self.efficientnet_model:
            with torch.no_grad():
                if self.cnn_model:
                    cnn_tensor = self.cnn_transform(img).unsqueeze(0).to(self.device)
                    outputs = self.cnn_model(cnn_tensor)
                    prob = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, class_idx = torch.max(prob, 0)
                    cnn_pred = self.cifar_classes[class_idx.item()]
                    cnn_conf = conf.item()

                if self.efficientnet_model:
                    eff_tensor = self.efficientnet_transform(img).unsqueeze(0).to(self.device)
                    outputs = self.efficientnet_model(eff_tensor)
                    prob = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, class_idx = torch.max(prob, 0)
                    efficientnet_pred = self.cifar_classes[class_idx.item()]
                    efficientnet_conf = conf.item()

                if self.resnet_model:
                    resnet_tensor = self.resnet_transform(img).unsqueeze(0).to(self.device)
                    outputs = self.resnet_model(resnet_tensor)
                    prob = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, class_idx = torch.max(prob, 0)
                    resnet_pred = self.cifar_classes[class_idx.item()]
                    resnet_conf = conf.item()

        return {
            "hog": {
                "status": statuses["hog"],
                "prediction": hog_pred,
                "confidence": hog_conf
            },
            "knn": {
                "status": statuses["knn"],
                "prediction": knn_pred,
                "confidence": knn_conf
            },
            "cnn": {
                "status": statuses["cnn"],
                "prediction": cnn_pred,
                "confidence": cnn_conf
            },
            "efficientnet": {
                "status": statuses["efficientnet"],
                "prediction": efficientnet_pred,
                "confidence": efficientnet_conf
            },
            "resnet": {
                "status": statuses["resnet"],
                "prediction": resnet_pred,
                "confidence": resnet_conf
            }
        }