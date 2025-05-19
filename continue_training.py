import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import argparse
import os
from datetime import datetime
from dataset import NIHXRayDataset
from model import BioViLClassifier

def train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch, num_epochs, device, writer, save_dir):
    best_map = 0.0
    # Get class names from dataset
    class_names = train_loader.dataset.dataset.get_class_names()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch + num_epochs} [Train]')
        for inputs, labels in train_pbar:
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
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{start_epoch + num_epochs} [Val]')
            for inputs, labels in val_pbar:
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
        
        print(f'Epoch {epoch+1}/{start_epoch + num_epochs}:')
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
    parser = argparse.ArgumentParser(description='Continue training BioViL on NIH Chest X-ray dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to continue from')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of additional epochs to train')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--threshold', type=float, default=0.4, help='Classification threshold')
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
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint with default settings, trying with weights_only=False: {str(e)}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get class names
    class_names = checkpoint.get('class_names', train_dataset.dataset.get_class_names())
    
    # Initialize model
    num_classes = len(class_names)
    model = BioViLClassifier(
        num_classes=num_classes,
        pretrained=False,  # We're loading from checkpoint
        threshold=args.threshold
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get the epoch from checkpoint
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel Summary:')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {class_names}')
    print(f'Continuing training from epoch {start_epoch}\n')
    
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Update learning rate to the new value
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Could not load optimizer state: {str(e)}")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('checkpoints', f'biovil_nih_continued_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', f'biovil_nih_continued_{timestamp}'))
    
    # Train model
    print('Starting continued training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        start_epoch=start_epoch,
        num_epochs=args.num_epochs,
        device=device,
        writer=writer,
        save_dir=save_dir
    )
    
    writer.close()
    print('Training complete!')

if __name__ == '__main__':
    main() 