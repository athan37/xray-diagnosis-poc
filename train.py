import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model import BioViLClassifier
from dataset import NIHXRayDataset

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, save_dir):
    best_map = 0.0
    # Get class names from dataset
    class_names = train_loader.dataset.dataset.get_class_names()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            if isinstance(inputs, dict):  # BioViL inputs
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.sigmoid().detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        
        # Calculate training metrics
        train_map = average_precision_score(train_labels, train_preds, average='macro')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                if isinstance(inputs, dict):  # BioViL inputs
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.sigmoid().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Calculate validation metrics
        val_map = average_precision_score(val_labels, val_preds, average='macro')
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mAP/train', train_map, epoch)
        writer.add_scalar('mAP/val', val_map, epoch)
        
        # Log per-class metrics
        for i, class_name in enumerate(class_names):
            train_ap = average_precision_score(train_labels[:, i], train_preds[:, i])
            val_ap = average_precision_score(val_labels[:, i], val_preds[:, i])
            writer.add_scalar(f'AP/train/{class_name}', train_ap, epoch)
            writer.add_scalar(f'AP/val/{class_name}', val_ap, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train mAP: {train_map:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}')
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': val_map,
                'class_names': class_names
            }, os.path.join(save_dir, 'best_model.pth'))
            
            # Save predictions for best model
            np.save(os.path.join(save_dir, 'best_val_preds.npy'), val_preds)
            np.save(os.path.join(save_dir, 'best_val_labels.npy'), val_labels)

def main():
    parser = argparse.ArgumentParser(description='Train BioViL on NIH Chest X-ray dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = NIHXRayDataset(
        label_file='data/nih/reduced_data.csv',
        image_dir='data/nih/images/images',
        model_type='vit'
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Use consistent seed for reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    num_classes = len(train_dataset.dataset.get_class_names())
    model = BioViLClassifier(
        num_classes=num_classes,
        pretrained=True,
        threshold=args.threshold
    ).to(device)
    
    # Store class names for later use
    class_names = train_dataset.dataset.get_class_names()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel Summary:')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {class_names}\n')
    
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('checkpoints', f'biovil_nih_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', f'biovil_nih_{timestamp}'))
    
    # Train model
    print('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        writer=writer,
        save_dir=save_dir
    )
    
    writer.close()
    print('Training complete!')

if __name__ == '__main__':
    main() 