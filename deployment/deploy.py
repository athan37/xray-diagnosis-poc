import os
import torch
import numpy as np
from PIL import Image
import json
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BioViLClassifier

class XRayDiagnosisModel:
    """Wrapper class for X-ray diagnosis model for easy deployment"""
    
    def __init__(self, model_path, device=None, threshold=0.5):
        """
        Initialize the X-ray diagnosis model
        
        Args:
            model_path (str): Path to model checkpoint
            device (str): Device to use ('cuda', 'mps', or 'cpu')
            threshold (float): Classification threshold
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f'Using device: {self.device}')
        
        # Load checkpoint
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise
        
        # Get class names from checkpoint
        self.class_names = self.checkpoint.get('class_names', None)
        if self.class_names is None:
            raise ValueError("No class names found in checkpoint")
        
        # Initialize model
        self.model = BioViLClassifier(
            num_classes=len(self.class_names),
            pretrained=False,
            threshold=threshold
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully with {len(self.class_names)} classes")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model input
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        try:
            # Load and convert to RGB if needed
            image = Image.open(image_path).convert('RGB')
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def predict(self, image_path, return_probabilities=True):
        """
        Make prediction on a single image
        
        Args:
            image_path (str): Path to image file
            return_probabilities (bool): Whether to return probabilities or binary predictions
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            predictions = (probabilities > self.model.threshold).astype(int)
        
        # Build results dictionary
        results = {
            'image_path': image_path,
            'predictions': {},
            'probabilities': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            results['predictions'][class_name] = bool(predictions[i])
            results['probabilities'][class_name] = float(probabilities[i])
        
        if return_probabilities:
            return results
        else:
            # Return only binary predictions
            return {
                'image_path': image_path,
                'predictions': results['predictions']
            }
    
    def batch_predict(self, image_paths, return_probabilities=True):
        """
        Make predictions on multiple images
        
        Args:
            image_paths (list): List of image paths
            return_probabilities (bool): Whether to return probabilities or binary predictions
            
        Returns:
            list: List of dictionaries with prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize the prediction for an image
        
        Args:
            image_path (str): Path to image file
            save_path (str): Path to save the visualization (None to display)
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with visualization
        """
        # Get prediction
        results = self.predict(image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('X-Ray Image')
        plt.axis('off')
        
        # Get sorted probabilities (highest first)
        items = list(results['probabilities'].items())
        items.sort(key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in items]
        probs = [item[1] for item in items]
        
        # Plot predictions
        plt.subplot(1, 2, 2)
        colors = ['green' if p > self.model.threshold else 'gray' for p in probs]
        
        bars = plt.barh(classes, probs, color=colors)
        plt.axvline(x=self.model.threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({self.model.threshold})')
        plt.xlim(0, 1)
        plt.title('Disease Probabilities')
        plt.xlabel('Probability')
        plt.legend()
        
        # Add probability values
        for i, v in enumerate(probs):
            if v > 0.1:  # Only show text for significant probabilities
                plt.text(v + 0.02, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='X-ray Diagnosis AI Deployment Script')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (or directory of images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, or cpu)')
    args = parser.parse_args()
    
    # Initialize model
    model = XRayDiagnosisModel(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold
    )
    
    # Check if input is a directory or a single image
    if os.path.isdir(args.image):
        # Get all image files in directory
        image_files = [
            os.path.join(args.image, f) for f in os.listdir(args.image)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        
        # Make batch predictions
        results = model.batch_predict(image_files)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    else:
        # Single image prediction
        result = model.predict(args.image)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Image: {os.path.basename(result['image_path'])}")
        
        # Print probabilities sorted by value (highest first)
        print("\nProbabilities:")
        items = list(result['probabilities'].items())
        items.sort(key=lambda x: x[1], reverse=True)
        for class_name, prob in items:
            print(f"{class_name}: {prob:.4f}")
        
        # Print positive predictions
        print("\nPositive Findings:")
        positives = [k for k, v in result['predictions'].items() if v]
        if positives:
            for finding in positives:
                print(f"- {finding}")
        else:
            print("- No positive findings")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        # Visualize
        if args.visualize:
            viz_path = args.output.replace('.json', '.png') if args.output else 'prediction.png'
            model.visualize_prediction(args.image, save_path=viz_path)
            print(f"Visualization saved to {viz_path}")

if __name__ == '__main__':
    main() 