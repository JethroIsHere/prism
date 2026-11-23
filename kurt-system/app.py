from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.model_registry import ModelRegistry
from backend.prediction_service import PredictionService
from backend.metrics_service import MetricsService
import os

# Initialize Flask app. By default, it looks for HTML in /templates and CSS/JS in /static.
app = Flask(__name__, static_folder='static', template_folder='templates')

# Security: Reject images larger than 5MB to prevent server overload
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# Initialize our custom backend logic
registry = ModelRegistry()
predictor = PredictionService()
metrics_svc = MetricsService()

# --- HTML ROUTES ---

@app.route('/')
def index():
    """Serves the main frontend dashboard."""
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return send_from_directory(assets_dir, filename)

# --- API ROUTES ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Checks if the Flask server is running."""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/models', methods=['GET'])
def get_model_status():
    """Returns the availability status of all five models: HOG, k-NN, CNN, EfficientNet, and ResNet."""
    statuses = registry.get_all_statuses()
    return jsonify(statuses), 200

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns the consolidated evaluation results from metrics.json files."""
    metrics_data = metrics_svc.get_consolidated_metrics()
    return jsonify(metrics_data), 200

@app.route('/api/confusion-matrix/<model_key>', methods=['GET'])
def get_confusion_matrix(model_key):
    folder_map = {
        'hog': 'ron-hog-svm',
        'knn': 'ron-color-knn',
        'cnn': 'duke-custom-cnn',
        'efficientnet': 'ron-efficientnet',
        'resnet': 'christine-resnet18'
    }
    if model_key not in folder_map:
        return jsonify({'error': 'Invalid model key'}), 404
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', folder_map[model_key], 'results'))
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    if not os.path.exists(cm_path):
        return jsonify({'error': 'Confusion matrix image not generated for this model'}), 404
    return send_from_directory(results_dir, 'confusion_matrix.png')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Accepts an image upload and returns predictions from all models."""
    
    # 1. Validate the incoming HTTP request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    
    # 2. Validate the user actually selected a file
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    # 3. Validate file format
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format. Please use JPG or PNG."}), 400

    # 4. Pass the image to our prediction service
    try:
        results = predictor.process_image(file)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server dynamically on localhost port 5000
    print("Starting PRISM Flask Server...")
    app.run(host='127.0.0.1', port=5000, debug=True)