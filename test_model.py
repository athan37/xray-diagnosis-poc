import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import pandas as pd
from tqdm import tqdm

from model import BioViLClassifier
from dataset import NIHXRayDataset

def plot_sample_predictions(model, test_loader, class_names, num_samples=5, threshold=0.5, save_dir='results'):
    """Plot some sample predictions from the test set"""
    model.eval()
    device = next(model.parameters()).device
    
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random samples
    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Store batch
            if len(all_images) < num_samples:
                # Add images and labels to our collection
                if isinstance(inputs, dict):
                    all_images.append(inputs)
                else:
                    all_images.append(inputs.cpu())
                all_labels.append(labels.cpu())
                
                # Get predictions
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
            else:
                break
    
    # Plot samples
    for i in range(min(num_samples, len(all_images))):
        image = all_images[i][0]  # First image in batch
        label = all_labels[i][0]  # First label in batch
        pred = all_preds[i][0]    # First prediction in batch
        
        # Convert tensor to PIL image for display
        if image.shape[0] == 3:  # If image is already in [C, H, W] format
            # Convert normalized tensor back to PIL image
            img_for_plot = transforms.ToPILImage()(image)
        else:
            # If we have a different format, just use the first image
            img_for_plot = Image.fromarray(image.numpy().astype(np.uint8))
        
        # Create figure with image and predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display image
        ax1.imshow(img_for_plot, cmap='gray')
        ax1.set_title('X-ray Image')
        ax1.axis('off')
        
        # Display predictions vs ground truth
        bars = ax2.barh(range(len(class_names)), pred.numpy(), alpha=0.6, label='Predicted')
        ax2.barh(range(len(class_names)), label.numpy(), alpha=0.3, color='r', label='Ground Truth')
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)
        ax2.set_xlim(0, 1)
        ax2.set_title('Predictions vs Ground Truth')
        ax2.axvline(x=threshold, color='k', linestyle='--', alpha=0.3, label=f'Threshold ({threshold})')
        ax2.legend()
        
        # Add value annotations to bars
        for j, (p, l) in enumerate(zip(pred.numpy(), label.numpy())):
            if p >= threshold:
                ax2.text(p + 0.01, j, f'{p:.2f}', va='center', fontweight='bold')
            if l == 1:
                ax2.text(0.01, j, 'GT', va='center', color='darkred', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()
        
    print(f"Sample predictions saved to {save_dir}")

def evaluate_model(model, test_loader, class_names, threshold=0.5):
    """Evaluate model performance on test set"""
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall metrics
    mAP = average_precision_score(all_labels, all_preds, average='macro')
    binary_preds = (all_preds > threshold).astype(int)
    
    # Calculate per-class metrics
    per_class_metrics = []
    for i, class_name in enumerate(class_names):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        
        # Count positives and true positives
        positives = np.sum(all_labels[:, i])
        true_positives = np.sum(binary_preds[:, i] * all_labels[:, i])
        
        # Calculate sensitivity (true positive rate)
        if positives > 0:
            sensitivity = true_positives / positives
        else:
            sensitivity = float('nan')
        
        # Calculate specificity (true negative rate)
        negatives = len(all_labels[:, i]) - positives
        true_negatives = np.sum((1 - binary_preds[:, i]) * (1 - all_labels[:, i]))
        if negatives > 0:
            specificity = true_negatives / negatives
        else:
            specificity = float('nan')
        
        per_class_metrics.append({
            'Class': class_name,
            'AP': ap,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Positives': positives
        })
    
    # Create a DataFrame for better display
    df_metrics = pd.DataFrame(per_class_metrics)
    
    # Sort by AP score (descending)
    df_metrics = df_metrics.sort_values('AP', ascending=False)
    
    return mAP, df_metrics

def main():
    parser = argparse.ArgumentParser(description='Test trained model on NIH X-ray dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample predictions to visualize')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint with default settings, trying with weights_only=False: {str(e)}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    class_names = checkpoint.get('class_names', None)
    if class_names is None:
        print("Warning: No class names found in checkpoint. Using default NIH dataset.")
    
    # Create dataset
    test_dataset = NIHXRayDataset(
        label_file='data/nih/reduced_data.csv',
        image_dir='data/nih/images/images',
        model_type='vit'
    )
    
    # If class names not in checkpoint, get from dataset
    if class_names is None:
        class_names = test_dataset.get_class_names()
    
    # Create data loader for the full dataset
    # We'll split it after loading to ensure consistent splits with training
    full_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model and load weights
    num_classes = len(class_names)
    model = BioViLClassifier(
        num_classes=num_classes,
        pretrained=False,
        threshold=args.threshold
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Split into train and test same as during training
    # We'll use only the test set for evaluation
    train_size = int(0.8 * len(test_dataset))
    val_size = len(test_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        test_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Same seed for consistent split
    )
    
    # Create test data loader
    test_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Model loaded from checkpoint: {args.checkpoint}")
    print(f"Evaluation on {len(val_dataset)} test samples")
    
    # Evaluate model
    mAP, df_metrics = evaluate_model(model, test_loader, class_names, args.threshold)
    
    # Print results
    print("\n===== Model Evaluation =====")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print("\nPer-class metrics:")
    print(df_metrics.to_string(index=False))
    
    # Save metrics to CSV
    os.makedirs(args.results_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(args.results_dir, 'metrics.csv'), index=False)
    
    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(
        model, 
        test_loader, 
        class_names, 
        num_samples=args.num_samples,
        threshold=args.threshold,
        save_dir=args.results_dir
    )
    
    print(f"\nResults saved to {args.results_dir}")

if __name__ == '__main__':
    main() 