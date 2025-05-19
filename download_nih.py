import os
import pandas as pd
import numpy as np
import requests
import gzip
import shutil
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def download_nih_dataset():
    """Download NIH ChestX-ray8 dataset (smaller version)"""
    # Create data directory
    os.makedirs('data/nih', exist_ok=True)
    
    # Download data from NIH (smaller version - first 4 parts)
    base_url = "https://nihcc.box.com/shared/static"
    files = {
        'images': [
            "https://nihcc.box.com/shared/static/v8aj2y8erm09oenziiqgq0m6pfp3dlu9.zip",  # Images_001.zip
            "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.zip",  # Images_002.zip
            "https://nihcc.box.com/shared/static/f1t00wwtdk3sat64bajdvt9id0w19zkk.zip",  # Images_003.zip
            "https://nihcc.box.com/shared/static/48m1nfx6emnmwmz2ly2bzl91v6m5p97q.zip",  # Images_004.zip
        ],
        'labels': "https://nihcc.box.com/shared/static/bqlyf7jop9tfdt3r0jikf1u3npv6ys1j.zip"  # Data_Entry_2017_v2020.csv
    }
    
    # Download and extract images
    print("Downloading images...")
    for i, url in enumerate(files['images'], 1):
        zip_file = f'data/nih/Images_{i:03d}.zip'
        if not os.path.exists(zip_file):
            print(f"\nDownloading Images_{i:03d}.zip...")
            try:
                download_file(url, zip_file)
                print(f"Extracting Images_{i:03d}.zip...")
                shutil.unpack_archive(zip_file, 'data/nih/images')
            except Exception as e:
                print(f"Error downloading/extracting Images_{i:03d}.zip: {str(e)}")
                print("Please download manually from: https://nihcc.app.box.com/v/ChestXray-NIHCC")
                return
    
    # Download and extract labels
    print("\nDownloading labels...")
    labels_file = 'data/nih/Data_Entry_2017_v2020.zip'
    if not os.path.exists(labels_file):
        try:
            download_file(files['labels'], labels_file)
            print("Extracting labels...")
            shutil.unpack_archive(labels_file, 'data/nih')
        except Exception as e:
            print(f"Error downloading/extracting labels: {str(e)}")
            print("Please download manually from: https://nihcc.app.box.com/v/ChestXray-NIHCC")
            return
    
    # Read and process labels
    print("\nProcessing labels...")
    df = pd.read_csv('data/nih/Data_Entry_2017_v2020.csv')
    
    # Filter to only include ChestX-ray8 diseases
    chestxray8_diseases = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Pneumonia',
        'Pneumothorax'
    ]
    
    # Convert Finding Labels to binary columns
    all_labels = set()
    for labels in df['Finding Labels'].str.split('|'):
        all_labels.update(labels)
    
    # Keep only ChestX-ray8 diseases
    all_labels = [label for label in all_labels if label in chestxray8_diseases]
    
    # Create binary columns for each label
    for label in all_labels:
        df[label] = df['Finding Labels'].str.contains(label).astype(int)
    
    # Keep only columns we need
    columns_to_keep = ['Image Index', 'Finding Labels'] + all_labels
    df = df[columns_to_keep]
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save processed data
    train_df.to_csv('data/nih/train_labels.csv', index=False)
    val_df.to_csv('data/nih/val_labels.csv', index=False)
    
    print("\nDataset preparation complete!")
    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print("\nAvailable labels (ChestX-ray8 diseases):")
    for label in sorted(all_labels):
        print(f"- {label}")

if __name__ == "__main__":
    download_nih_dataset() 