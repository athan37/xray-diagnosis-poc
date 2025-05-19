import os
import torch
from PIL import Image
import argparse
from model import XRayClassifier
import matplotlib.pyplot as plt
import numpy as np

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Add safe globals for numpy types
    import torch.serialization
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
    
    # Load checkpoint with weights_only=False for backward compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = XRayClassifier(
        num_classes=len(checkpoint['class_names']),
        model_type='densenet121',
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

def predict_image(model, image_path, device, class_names, threshold=0.5):
    """Make prediction on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = model.get_transforms()['val']
    if transform:
        image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probs = outputs.cpu().numpy()[0]
        predictions = (probs > threshold).astype(int)
    
    # Create results dictionary
    results = {
        'image_path': image_path,
        'predictions': {},
        'probabilities': {}
    }
    
    for i, class_name in enumerate(class_names):
        results['predictions'][class_name] = bool(predictions[i])
        results['probabilities'][class_name] = float(probs[i])
    
    return results

def visualize_prediction(image_path, results):
    """Visualize the image and its predictions"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('X-Ray Image')
    plt.axis('off')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    classes = list(results['probabilities'].keys())
    probs = list(results['probabilities'].values())
    colors = ['green' if p > 0.5 else 'red' for p in probs]
    
    plt.barh(classes, probs, color=colors)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.title('Disease Probabilities')
    plt.xlabel('Probability')
    
    # Add probability values
    for i, v in enumerate(probs):
        plt.text(v + 0.02, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Make predictions on X-ray images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/densenet121/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Classification threshold')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save visualization (if None, will display)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model, class_names = load_model(args.checkpoint, device)
    print(f'\nModel loaded with classes: {class_names}')
    
    # Make prediction
    results = predict_image(model, args.image, device, class_names, args.threshold)
    
    # Print results
    print('\nPrediction Results:')
    print(f"Image: {os.path.basename(results['image_path'])}")
    print("\nProbabilities:")
    for class_name, prob in results['probabilities'].items():
        print(f"{class_name}: {prob:.2%}")
    print("\nPredictions (threshold = {args.threshold}):")
    for class_name, pred in results['predictions'].items():
        print(f"{class_name}: {'Positive' if pred else 'Negative'}")
    
    # Visualize
    fig = visualize_prediction(args.image, results)
    if args.output:
        fig.savefig(args.output)
        print(f'\nVisualization saved to: {args.output}')
    else:
        plt.show()

if __name__ == '__main__':
    main() 