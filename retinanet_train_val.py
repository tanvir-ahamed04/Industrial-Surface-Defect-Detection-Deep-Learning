# retinanet_train_val.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import ANNOTATION_DIR, DEVICE, IMAGE_DIR, NUM_EPOCHS
from dataset import NEUDETDataset
from metrics import calculate_map
from retinanet_train import get_retinanet, train_one_epoch
from utils import collate_fn

def compute_loss(model, images, targets):
    """
    Compute loss without updating gradients
    """
    # Temporarily set to training mode to get loss dict
    was_training = model.training
    model.train()
    
    with torch.no_grad():
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
    
    # Restore mode
    if not was_training:
        model.eval()
    
    return total_loss.item()

def validate(model, loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            
            # Convert targets to device
            device_targets = []
            for target in targets:
                device_target = {}
                for key, value in target.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype == np.float64 or value.dtype == np.float32:
                            value = torch.from_numpy(value).float()
                        elif value.dtype == np.int64 or value.dtype == np.int32:
                            value = torch.from_numpy(value).long()
                        else:
                            value = torch.from_numpy(value)
                    device_target[key] = value.to(DEVICE)
                device_targets.append(device_target)
            
            total_loss += compute_loss(model, images, device_targets)
    
    return total_loss / len(loader)

def train_with_validation():
    # Split dataset
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    model = get_retinanet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_map = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer)
        
        # Validation
        val_loss = validate(model, val_loader)
        
        # Calculate mAP
        map_result = calculate_map(model, val_loader, DEVICE)
        current_map = map_result['map'].item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"mAP: {current_map:.4f}")
        
        # Save best model
        if current_map > best_map:
            best_map = current_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': best_map,
            }, 'best_retinanet.pth')
        
        # Save checkpoint
        torch.save(model.state_dict(), f'retinanet_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    model = train_with_validation()