# X-Ray Diagnosis AI

A deep learning system for multi-label classification of 15 different conditions in chest X-rays using the BioViL vision transformer architecture.

## Project Overview

This project implements a chest X-ray diagnosis system capable of detecting multiple pathologies from a single X-ray image. The model is based on a Vision Transformer (ViT) architecture and was trained on the NIH Chest X-ray dataset.

### Features

- Multi-label classification of 15 different chest X-ray findings
- Pre-trained Vision Transformer (ViT) backbone
- TensorBoard integration for experiment tracking
- Training, validation, and inference pipelines
- Performance evaluation metrics
- Visualization of model predictions

## Model Performance

The current best model achieves a mean Average Precision (mAP) of 0.34 across 15 different conditions. Performance varies by condition:

| Condition         | Average Precision | Sensitivity | Specificity |
|-------------------|-------------------|-------------|-------------|
| Effusion          | 0.55              | 0.44        | 0.89        |
| Emphysema         | 0.53              | 0.39        | 0.98        |
| Cardiomegaly      | 0.44              | 0.36        | 0.98        |
| Edema             | 0.42              | 0.38        | 0.96        |
| Infiltration      | 0.42              | 0.33        | 0.85        |
| Other conditions  | <0.40             | varies      | varies      |

*Note: The model was trained on a limited dataset of ~7,000 images. Commercial chest X-ray AI systems typically use 50,000-200,000+ images for training. Performance could be significantly improved with more training data.*

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/athan37/xray-diagnosis-poc.git
   cd xray-diagnosis-poc
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the NIH dataset (optional for training):
   ```bash
   python download_nih.py
   ```

## Usage

### Training

To train a new model from scratch:

```bash
python train.py --batch_size 8 --num_epochs 10 --lr 1e-5
```

### Continue Training From Checkpoint

To continue training from a checkpoint:

```bash
python continue_training.py --checkpoint checkpoints/your_checkpoint.pth --num_epochs 5
```

### Evaluation

To evaluate a trained model:

```bash
python test_model.py --checkpoint checkpoints/your_checkpoint.pth
```

### Prediction

For making predictions on new images:

```bash
python predict.py --checkpoint checkpoints/your_checkpoint.pth --image path/to/image.jpg
```

## Deployment

The project includes a deployment directory with everything needed to deploy the model:

```bash
cd deployment
python deploy.py --model best_model.pth --image path/to/image.jpg --visualize
```

See the [deployment README](deployment/README.md) for more details on deployment options.

## Project Structure

- `model.py` - Model architecture definition
- `dataset.py` - Dataset loading and preprocessing
- `train.py` - Training script
- `continue_training.py` - Script for resuming training
- `test_model.py` - Evaluation script
- `predict.py` - Inference script
- `download_nih.py` - Script to download NIH dataset
- `find_best_examples.py` - Utility to find best prediction examples
- `deployment/` - Model deployment package

## Future Work

- [ ] Expand training dataset to improve performance
- [ ] Implement attention visualization
- [ ] Add model explainability methods
- [ ] Create web demo interface
- [ ] Support for DICOM format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH for providing the Chest X-ray dataset
- BioViL and Vision Transformer architecture developers
