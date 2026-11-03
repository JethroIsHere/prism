import os
import json

class MetricsService:
    """
    Reads and consolidates the evaluation results (metrics.json) from all three models.
    """
    def __init__(self):
        # Navigate up to the main prism repository folder
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Define the exact paths to the expected metrics files
        self.metrics_paths = {
            "hog": os.path.join(self.base_dir, 'ron-hog-svm', 'results', 'metrics.json'),
            "cnn": os.path.join(self.base_dir, 'duke-custom-cnn', 'results', 'metrics.json'),
            "resnet": os.path.join(self.base_dir, 'christine-resnet18', 'results', 'metrics.json')
        }

    def _load_json(self, filepath):
        """
        Safely attempts to open and read a JSON file.
        Returns None if the file does not exist or contains invalid JSON.
        """
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError:
            # Handle the case where the file exists but is corrupted or empty
            return None

    def get_consolidated_metrics(self):
        """
        Gathers the metrics from all three models into a single dictionary.
        """
        return {
            "hog": self._load_json(self.metrics_paths["hog"]),
            "cnn": self._load_json(self.metrics_paths["cnn"]),
            "resnet": self._load_json(self.metrics_paths["resnet"])
        }