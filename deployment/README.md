# X-ray Diagnosis AI - Deployment

This directory contains the deployment version of the X-ray diagnosis AI model for detecting 15 different conditions in chest X-rays.

## Model Download

The model file (`best_model.pth`) is too large for GitHub hosting. You can download it from:
- Google Drive: [best_model.pth](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)
- Or contact the repository owner for access

**Important:** After downloading, place the model file in the `deployment` directory.

## Model Performance

The deployed model achieves a mean Average Precision (mAP) of 0.34 across 15 different conditions. Performance varies by condition:

| Condition         | Average Precision | Sensitivity | Specificity |
|-------------------|-------------------|-------------|-------------|
| Effusion          | 0.55              | 0.44        | 0.89        |
| Emphysema         | 0.53              | 0.39        | 0.98        |
| Cardiomegaly      | 0.44              | 0.36        | 0.98        |
| Edema             | 0.42              | 0.38        | 0.96        |
| Infiltration      | 0.42              | 0.33        | 0.85        |
| Other conditions  | <0.40             | varies      | varies      |

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The `deploy.py` script provides a command-line interface for making predictions:

```bash
# Basic usage
python deploy.py --model best_model.pth --image path/to/image.jpg

# Save results to JSON
python deploy.py --model best_model.pth --image path/to/image.jpg --output results.json

# Generate visualization
python deploy.py --model best_model.pth --image path/to/image.jpg --visualize

# Batch processing
python deploy.py --model best_model.pth --image path/to/image_directory --output batch_results.json

# Change classification threshold
python deploy.py --model best_model.pth --image path/to/image.jpg --threshold 0.4
```

### Python API

You can also use the model programmatically:

```python
from deploy import XRayDiagnosisModel

# Initialize model
model = XRayDiagnosisModel(model_path='best_model.pth')

# Single image prediction
result = model.predict('path/to/image.jpg')

# Batch prediction
results = model.batch_predict(['path/to/image1.jpg', 'path/to/image2.jpg'])

# Generate visualization
model.visualize_prediction('path/to/image.jpg', save_path='prediction.png')
```

## Integration with Web Services

The model can be easily integrated with web frameworks like Flask or FastAPI:

### Flask Example

```python
from flask import Flask, request, jsonify
from deploy import XRayDiagnosisModel
import os

app = Flask(__name__)
model = XRayDiagnosisModel(model_path='best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file temporarily
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)
    
    # Make prediction
    try:
        result = model.predict(temp_path)
        os.remove(temp_path)  # Clean up
        return jsonify(result)
    except Exception as e:
        os.remove(temp_path)  # Clean up
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Note on Data Requirements

This model was trained on a limited dataset of ~7,000 images. Commercial chest X-ray AI systems typically use 50,000-200,000+ images for training. The performance could be significantly improved with more training data. 