import os

class ModelRegistry:
    """
    Manages the availability of the three Computer Vision models.
    It checks the repository directory structure for required artifacts.
    """
    def __init__(self):
        # Navigate up from kurt-system/backend -> kurt-system -> root prism folder
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        self.hog_dir = os.path.join(self.base_dir, 'ron-hog-svm', 'results')
        self.cnn_dir = os.path.join(self.base_dir, 'duke-custom-cnn', 'results')
        self.resnet_dir = os.path.join(self.base_dir, 'christine-resnet18', 'results')

    def check_hog_status(self):
        """Checks if Ron's HOG + SVM artifacts are present."""
        model_path = os.path.join(self.hog_dir, 'model.npz')
        scaler_path = os.path.join(self.hog_dir, 'scaler.npz')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            return {"available": True, "message": "Ready"}
        return {"available": False, "message": "Artifacts Missing"}

    def check_cnn_status(self):
        """Checks if Duke's Custom CNN artifacts are present."""
        model_path = os.path.join(self.cnn_dir, 'model.pt')
        if os.path.exists(model_path):
            return {"available": True, "message": "Ready"}
        return {"available": False, "message": "Pending Training"}

    def check_resnet_status(self):
        """Checks if Christine's ResNet-18 artifacts are present."""
        model_path = os.path.join(self.resnet_dir, 'model.pt')
        if os.path.exists(model_path):
            return {"available": True, "message": "Ready"}
        return {"available": False, "message": "Pending Training"}

    def get_all_statuses(self):
        """Returns a consolidated dictionary of all model statuses."""
        return {
            "hog": self.check_hog_status(),
            "cnn": self.check_cnn_status(),
            "resnet": self.check_resnet_status()
        }