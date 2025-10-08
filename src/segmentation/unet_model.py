"""
U-Net Deep Learning Model for Candida Segmentation
Enhanced segmentation approach for conference paper comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import time
from sklearn.model_selection import train_test_split

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation of Candida organisms
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (Contracting Path)
        self.enc1 = self.double_conv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = self.double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = self.double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = self.double_conv(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, n_classes, 1)
        
    def double_conv(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)
        
        return torch.sigmoid(self.final(dec4))

class CandidaDataset(Dataset):
    """
    Dataset class for loading Candida images and masks
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, input_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.input_size = input_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.input_size)
        mask = cv2.resize(mask, self.input_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC to CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask

class CandidaSegmentationTrainer:
    """
    Training pipeline for U-Net segmentation model with CUDA optimization
    """
    
    def __init__(self, model, device=None):
        # Auto-detect best device with CUDA 13.0 support
        if device is None:
            if torch.cuda.is_available():
                device = f'cuda:{torch.cuda.current_device()}'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = 'cpu'
                print("CUDA not available, using CPU")
        
        self.device = device
        self.model = model.to(device)
        
        # Optimize for CUDA 13.0
        if 'cuda' in str(device):
            # Enable mixed precision training for better performance
            self.scaler = torch.cuda.amp.GradScaler()
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Optimize memory usage
            torch.cuda.empty_cache()
        else:
            self.scaler = None
            
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for images, masks in train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Use mixed precision if CUDA is available
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Clear cache periodically to prevent memory issues
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
                
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f'Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping with model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_unet_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Early stopping
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        return train_losses, val_losses

def create_training_data(image_dir, mask_dir):
    """
    Prepare training data from existing images and color-based masks
    """
    image_paths = []
    mask_paths = []
    
    # Get all processed images that have corresponding masks
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Find corresponding image
            image_name = mask_file.replace('.png', '.jp2')
            
            # Search in all data subfolders
            for root, dirs, files in os.walk(image_dir):
                if image_name in files:
                    image_path = os.path.join(root, image_name)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
                        break
    
    return image_paths, mask_paths

def main():
    """
    Main training function
    """
    # Paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    image_dir = os.path.join(project_root, 'data')
    mask_dir = os.path.join(project_root, 'results', 'segmentation_masks')
    
    # Create training data
    image_paths, mask_paths = create_training_data(image_dir, mask_dir)
    print(f"Found {len(image_paths)} training samples")
    
    if len(image_paths) == 0:
        print("No training data found. Run color-based segmentation first.")
        return
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = CandidaDataset(train_images, train_masks)
    val_dataset = CandidaDataset(val_images, val_masks)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model and trainer
    model = UNet(n_channels=3, n_classes=1)
    trainer = CandidaSegmentationTrainer(model)
    
    # Train model
    print("Starting U-Net training...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=50)
    
    # Save model
    model_path = os.path.join(project_root, 'results', 'unet_candida_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()