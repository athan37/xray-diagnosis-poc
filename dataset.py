import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np

class NIHXRayDataset(Dataset):
    def __init__(self, label_file, image_dir='data/nih/images/images', model_type='vit', transform=None):
        """
        Args:
            label_file (str): Path to the CSV file with labels
            image_dir (str): Directory with all the images
            model_type (str): Type of model to use ('vit' or 'resnet')
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = pd.read_csv(label_file)
        self.image_dir = image_dir
        self.model_type = model_type
        
        # Get all unique labels from the dataset
        all_labels = set()
        for labels in self.df['Finding Labels'].str.strip('[]').str.split(','):
            all_labels.update([label.strip().strip("'") for label in labels])
        self.classes = sorted(list(all_labels))
        
        # Create binary columns for each label
        for label in self.classes:
            self.df[label] = self.df['Finding Labels'].str.contains(label).astype(int)
        
        # Standard transforms for ViT
        if transform is not None:
            self.transform = transform
        else:
            if model_type == 'vit':
                # ViT expects 224x224 images, normalized with ImageNet stats
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Default transforms for other models
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['Image Index'])
        try:
            # Load image and convert to RGB if grayscale
            image = Image.open(img_name)
            if image.mode != 'RGB':  # Ensure image is RGB
                image = image.convert('RGB')
            
            # Get labels for this image
            labels = self.df.iloc[idx][self.classes].values.astype(float)
            
            # Apply transforms to the image
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(labels, dtype=torch.float)
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a placeholder image and zeros for labels
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, torch.zeros(len(self.classes))
    
    def get_class_names(self):
        return self.classes 