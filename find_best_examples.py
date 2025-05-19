import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from model import BioViLClassifier

# Constants
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

def load_model(checkpoint_path, threshold=0.4):
    """Load model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Trying to load with weights_only=False due to: {str(e)}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get class names or use default NIH classes
    class_names = checkpoint.get('class_names', CLASS_NAMES)
    
    # Initialize model
    model = BioViLClassifier(
        num_classes=len(class_names),
        pretrained=False,
        threshold=threshold
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names, device

def evaluate_image_quality(image_path):
    """Evaluate the quality of an X-ray image based on multiple metrics."""
    try:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0  # Image couldn't be loaded
            
        # Convert to float for calculations
        img_float = img.astype(float) / 255.0
        
        # 1. Contrast: standard deviation of pixel values
        contrast = np.std(img_float) * 100
        
        # 2. Sharpness using Laplacian filter
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = np.mean(np.abs(laplacian)) * 0.1
        
        # 3. Entropy: measure of information content
        histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
        histogram = histogram / np.sum(histogram)
        non_zero_hist = histogram[histogram > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist)) * 10
        
        # Combine metrics (weights can be adjusted as needed)
        quality_score = (0.4 * contrast + 0.4 * sharpness + 0.2 * entropy)
        return min(quality_score, 100.0)  # Cap at 100
        
    except Exception as e:
        print(f"Error evaluating image quality: {str(e)}")
        return 0  # Return 0 for failed quality assessment

def predict_image(model, image_path, device):
    """Make prediction on a single image"""
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.sigmoid(outputs)
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        
        return probs, True
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, False

def find_best_examples(model, class_names, device, csv_path, images_dir, output_dir, 
                     conditions, num_per_condition=3, max_images=500):
    """Find the best examples for each condition based on prediction confidence and image quality."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file containing labels
    df = pd.read_csv(csv_path)
    
    # Get list of all image files
    all_files = df['Image Index'].tolist()
    if max_images and max_images < len(all_files):
        all_files = np.random.choice(all_files, max_images, replace=False).tolist()
    
    # Track best examples for each condition
    best_examples = {condition: [] for condition in conditions}
    
    # Debug counts
    condition_counts = {condition: 0 for condition in conditions}
    prediction_above_threshold = {condition: 0 for condition in conditions}
    
    print(f"Searching for best examples among {len(all_files)} images...")
    for i, filename in enumerate(all_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(all_files)} images")
            
        # Get image path
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            continue
            
        # Get predictions
        predictions, success = predict_image(model, image_path, device)
        if not success:
            continue
            
        # Get true labels
        row = df[df['Image Index'] == filename]
        if len(row) == 0:
            continue
            
        findings = row['Finding Labels'].iloc[0].split('|')
            
        # Evaluate image quality
        quality_score = evaluate_image_quality(image_path)
        
        # Check each condition of interest
        for condition in conditions:
            # Check if this image has this condition
            has_condition = condition in findings
            
            # Skip if not a positive example for this condition
            if not has_condition:
                continue
                
            # Count condition occurrence
            condition_counts[condition] += 1
                
            # Get the index for this condition in our model predictions
            if condition in class_names:
                condition_idx = class_names.index(condition)
                pred_score = predictions[condition_idx]
                
                # For debugging
                if pred_score > 0.5:
                    prediction_above_threshold[condition] += 1
                
                # Combined score favors both high prediction confidence and good image quality
                combined_score = 0.7 * pred_score + 0.3 * (quality_score / 100.0)
                
                # Only consider correctly predicted cases (prediction > 0.5)
                if pred_score > 0.5:
                    example = {
                        'filename': filename,
                        'prediction': pred_score,
                        'quality_score': quality_score,
                        'combined_score': combined_score
                    }
                    
                    best_examples[condition].append(example)
                    # Sort and keep only the top examples
                    best_examples[condition] = sorted(
                        best_examples[condition], 
                        key=lambda x: x['combined_score'], 
                        reverse=True
                    )[:num_per_condition]

    # Print debug counts
    print("\nCondition occurrences in the dataset:")
    for condition, count in condition_counts.items():
        print(f"{condition}: {count} occurrences, {prediction_above_threshold[condition]} predictions above threshold")
    
    # Save the best examples
    summary_rows = []
    for condition, examples in best_examples.items():
        if not examples:
            print(f"No good examples found for {condition}")
            continue
            
        # Create directory for this condition
        condition_dir = os.path.join(output_dir, condition.replace(' ', '_'))
        os.makedirs(condition_dir, exist_ok=True)
        
        for example in examples:
            # Copy the image to the output directory
            src_path = os.path.join(images_dir, example['filename'])
            dst_path = os.path.join(condition_dir, example['filename'])
            
            # Read and write the image (effectively copying it)
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)
            
            print(f"Found good example for {condition}: {example['filename']} "
                  f"(prediction={example['prediction']:.3f}, quality={example['quality_score']:.1f})")
            
            # Add to summary
            summary_rows.append({
                'condition': condition,
                'filename': example['filename'],
                'prediction': example['prediction'],
                'quality_score': example['quality_score'],
                'combined_score': example['combined_score']
            })
    
    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, 'best_examples_summary.csv'), index=False)
        print(f"Summary saved to {os.path.join(output_dir, 'best_examples_summary.csv')}")
    else:
        print("No good examples found for any condition")

def main():
    parser = argparse.ArgumentParser(description='Find best X-ray examples for demonstration')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/biovil_nih_continued_20250518_131059/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str, default='data/nih/reduced_data.csv',
                      help='Path to dataset CSV file')
    parser.add_argument('--images_dir', type=str, default='data/nih/images/images',
                      help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='results/best_examples',
                      help='Directory to save best examples')
    parser.add_argument('--conditions', type=str, nargs='+', 
                      default=['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax'],
                      help='Conditions to find examples for')
    parser.add_argument('--num_per_condition', type=int, default=2,
                      help='Number of examples per condition')
    parser.add_argument('--max_images', type=int, default=500,
                      help='Maximum number of images to process (for faster execution)')
    parser.add_argument('--threshold', type=float, default=0.4,
                      help='Classification threshold')
    
    args = parser.parse_args()
    
    # Load model
    model, class_names, device = load_model(args.checkpoint, args.threshold)
    
    # Find best examples
    find_best_examples(
        model=model,
        class_names=class_names,
        device=device,
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        conditions=args.conditions,
        num_per_condition=args.num_per_condition,
        max_images=args.max_images
    )

if __name__ == '__main__':
    main() 