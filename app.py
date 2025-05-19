import os
import json
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model class from deploy.py
from deployment.deploy import XRayDiagnosisModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'deployment/best_model.pth'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model at startup
try:
    model = XRayDiagnosisModel(model_path=app.config['MODEL_PATH'])
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/')
def index():
    """Render the main page with upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for X-ray prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser submits an empty part
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run prediction
            result = model.predict(filepath)
            
            # Generate visualization image
            plt.figure(figsize=(12, 6))
            
            # Load and plot the original image
            image = Image.open(filepath).convert('RGB')
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('X-Ray Image')
            plt.axis('off')
            
            # Get sorted probabilities (highest first)
            items = list(result['probabilities'].items())
            items.sort(key=lambda x: x[1], reverse=True)
            classes = [item[0] for item in items]
            probs = [item[1] for item in items]
            
            # Plot predictions
            plt.subplot(1, 2, 2)
            colors = ['green' if p > model.model.threshold else 'gray' for p in probs]
            
            bars = plt.barh(classes, probs, color=colors)
            plt.axvline(x=model.model.threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Threshold ({model.model.threshold})')
            plt.xlim(0, 1)
            plt.title('Disease Probabilities')
            plt.xlabel('Probability')
            plt.legend()
            
            # Add probability values
            for i, v in enumerate(probs):
                if v > 0.1:
                    plt.text(v + 0.02, i, f'{v:.2f}', va='center')
            
            plt.tight_layout()
            
            # Save the figure to a BytesIO object
            img_data = io.BytesIO()
            plt.savefig(img_data, format='png')
            img_data.seek(0)
            plt.close()
            
            # Encode the image as base64
            encoded_img = base64.b64encode(img_data.getvalue()).decode('utf-8')
            
            # Get positive findings
            positives = [k for k, v in result['predictions'].items() if v]
            
            # Generate heatmaps for the top 3 most probable findings
            heatmaps = []
            for i, (class_name, prob) in enumerate(items[:3]):
                if prob > 0.1:  # Only generate for significant probabilities
                    try:
                        # Find the class index for this class name
                        class_idx = model.class_names.index(class_name)
                        
                        # Generate the heatmap
                        _, _, superimposed_img, _ = model.generate_heatmap(filepath, class_idx)
                        
                        # Convert to base64
                        _, heatmap_buffer = cv2.imencode('.png', superimposed_img)
                        heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
                        
                        heatmaps.append({
                            'class_name': class_name,
                            'probability': prob,
                            'heatmap': heatmap_base64
                        })
                    except Exception as e:
                        logger.error(f"Error generating heatmap for {class_name}: {str(e)}")
            
            # Add visualization to result
            api_result = {
                'probabilities': result['probabilities'],
                'predictions': result['predictions'],
                'positive_findings': positives,
                'visualization': encoded_img,
                'heatmaps': heatmaps
            }
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(api_result)
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the API."""
    health_status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    }
    return jsonify(health_status)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 