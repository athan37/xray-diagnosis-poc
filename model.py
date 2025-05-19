import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoProcessor, ViTForImageClassification, ViTModel

class XRayClassifier(nn.Module):
    def __init__(self, num_classes=2, model_type='densenet121', pretrained=True, threshold=0.5):
        """
        Args:
            num_classes (int): Number of output classes
            model_type (str): Type of model to use ('densenet121', 'medclip', 'biovil')
            pretrained (bool): Whether to use pretrained weights
            threshold (float): Classification threshold for multi-label predictions
        """
        super(XRayClassifier, self).__init__()
        self.model_type = model_type
        self.threshold = threshold
        
        if model_type == 'densenet121':
            # Load pretrained DenseNet121
            self.backbone = timm.create_model('densenet121', 
                                            pretrained=pretrained,
                                            in_chans=1,  # For grayscale images
                                            num_classes=0)  # Remove classification head
            
            # Get the feature dimension
            self.feature_dim = self.backbone.num_features
            
            # Add custom classification head with sigmoid activation for multi-label
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label classification
            )
            
        elif model_type in ['medclip', 'biovil']:
            # Initialize vision encoder from MedCLIP/BioViL
            model_name = 'microsoft/BiomedVLP-CXR-BERT-general' if model_type == 'biovil' else 'StanfordAIMI/MedCLIP'
            self.backbone = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Get the feature dimension (768 for BERT-based models)
            self.feature_dim = 768
            
            # Add custom classification head with sigmoid activation for multi-label
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label classification
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def forward(self, x):
        if self.model_type == 'densenet121':
            # Get features from DenseNet
            features = self.backbone(x)
            # Apply classification head
            return self.classifier(features)
            
        elif self.model_type in ['medclip', 'biovil']:
            # Process input for MedCLIP/BioViL
            inputs = self.processor(images=x, return_tensors="pt", padding=True)
            inputs = {k: v.to(x.device) for k, v in inputs.items()}
            
            # Get image embeddings
            outputs = self.backbone(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            
            # Apply classification head
            return self.classifier(features)
    
    def predict(self, x):
        """Get binary predictions using threshold"""
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > self.threshold).float()
            
    def get_transforms(self):
        """Get the appropriate transforms for the model"""
        if self.model_type == 'densenet121':
            return {
                'train': timm.data.create_transform(
                    input_size=(224, 224),
                    is_training=True,
                    mean=[0.485],
                    std=[0.229],
                    auto_augment='rand-m9-mstd0.5-inc1',
                ),
                'val': timm.data.create_transform(
                    input_size=(224, 224),
                    is_training=False,
                    mean=[0.485],
                    std=[0.229],
                )
            }
        elif self.model_type in ['medclip', 'biovil']:
            # MedCLIP/BioViL have their own processors
            return None

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, threshold=0.5):
        super(ViTClassifier, self).__init__()
        self.threshold = threshold
        
        # Load ViT model
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Replace the classification head for multi-label classification
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass with standard torch tensors (B, C, H, W)
        """
        outputs = self.vit(pixel_values=x)
        return outputs.logits
    
    def predict(self, x):
        """Make binary predictions based on threshold"""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > self.threshold).float()
    
    def get_transforms(self):
        """ViT has its own processor, so we return None"""
        return None 

class BioViLClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, threshold=0.5):
        super(BioViLClassifier, self).__init__()
        self.threshold = threshold
        
        # Load ViT model but only use the backbone (vision encoder)
        if pretrained:
            self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Add hook to get last hidden state
        self.last_hidden_state = None
        self.backbone.encoder.layer[-1].register_forward_hook(self._get_last_hidden_state)
        
        # Classification head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def _get_last_hidden_state(self, module, input, output):
        """Hook to get the last hidden state for Grad-CAM visualization"""
        self.last_hidden_state = output
    
    def get_last_hidden_state(self):
        """Return the last hidden state for Grad-CAM"""
        return self.last_hidden_state
        
    def forward(self, x):
        """Forward pass with standard torch tensor images [B, C, H, W]"""
        # Process through ViT backbone
        outputs = self.backbone(pixel_values=x)
        # Use the pooled output (CLS token) for classification
        pooled_output = outputs.pooler_output
        # Apply classification head
        logits = self.classifier(pooled_output)
        return logits
    
    def predict(self, x):
        """Make binary predictions based on threshold"""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > self.threshold).float() 