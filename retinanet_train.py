# Focal Loss
import torch
import torch.nn as nn
import numpy as np  # ADD THIS IMPORT
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from tqdm import tqdm

from config import *
from dataset import NEUDETDataset
from utils import collate_fn

def get_retinanet():
    """
    Create RetinaNet model with pre-trained backbone and custom head for 7 classes.
    """
    # Load model with pre-trained COCO weights
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    
    # Get the current head to understand its structure
    classification_head = model.head.classification_head
    
    # Get the number of anchors per location
    num_anchors = classification_head.num_anchors
    
    # Get the input channels for cls_logits
    conv_block = classification_head.conv
    in_channels = conv_block[-1].out_channels
    
    # Replace the classification layer (cls_logits)
    classification_head.cls_logits = nn.Conv2d(
        in_channels, 
        num_anchors * NUM_CLASSES,  # 7 classes
        kernel_size=3, 
        stride=1, 
        padding=1
    )
    
    # Proper initialization for the new layer
    nn.init.normal_(classification_head.cls_logits.weight, std=0.01)
    nn.init.constant_(classification_head.cls_logits.bias, -4.6)
    
    # Update num_classes in the head
    classification_head.num_classes = NUM_CLASSES
    
    return model

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in tqdm(loader):
        # Move images to device
        images = [img.to(DEVICE) for img in images]
        
        # FIX: Convert numpy arrays to tensors and move to device
        device_targets = []
        for target in targets:
            device_target = {}
            for key, value in target.items():
                # Convert numpy arrays to tensors
                if isinstance(value, np.ndarray):
                    if value.dtype == np.float64 or value.dtype == np.float32:
                        value = torch.from_numpy(value).float()
                    elif value.dtype == np.int64 or value.dtype == np.int32:
                        value = torch.from_numpy(value).long()
                    else:
                        value = torch.from_numpy(value)
                # Move to device
                device_target[key] = value.to(DEVICE)
            device_targets.append(device_target)
        
        # Forward pass
        loss_dict = model(images, device_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)

def main():
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_retinanet().to(DEVICE)
    
    # Set different learning rates for backbone and head
    params = [
        {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower LR for backbone
        {'params': model.head.parameters(), 'lr': 1e-4},      # Higher LR for head
    ]
    
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, loader, optimizer)
        print(f"[RetinaNet] Epoch {epoch+1}: Loss {loss:.4f}")
        torch.save(model.state_dict(),
                   f"retinanet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()