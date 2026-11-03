from backend.model_registry import ModelRegistry
import time

class PredictionService:
    """
    Handles routing the uploaded image to the three different models.
    """
    def __init__(self):
        self.registry = ModelRegistry()

    def process_image(self, image_file):
        """
        Simulates processing the uploaded image.
        In Phase 4, this will pass the image to the actual ML models.
        """
        statuses = self.registry.get_all_statuses()
        
        # Simulate inference computation time (1 second)
        time.sleep(1)

        # Build the response JSON
        results = {
            "hog": {
                "status": statuses["hog"],
                "prediction": None,
                "confidence": 0
            },
            "cnn": {
                "status": statuses["cnn"],
                "prediction": None,
                "confidence": 0
            },
            "resnet": {
                "status": statuses["resnet"],
                "prediction": None,
                "confidence": 0
            }
        }
        
        return results