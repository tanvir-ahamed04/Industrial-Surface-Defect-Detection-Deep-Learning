# utils.py
import torch

def collate_fn(batch):
    """
    Custom collate function for object detection
    """
    images, targets = zip(*batch)
    
    # Images are already tensors
    images = list(images)
    
    # Targets are dictionaries with tensors
    targets = list(targets)
    
    return images, targets