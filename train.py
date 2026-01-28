import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import NEUDETDataset
from model import get_model
from utils import collate_fn

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in tqdm(loader):
        images = [img.to(DEVICE) for img in images]
        
        # Fix: Convert all values in targets to tensors on device
        device_targets = []
        for target in targets:
            device_target = {}
            for key, value in target.items():
                # If it's a numpy array, convert to tensor first
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                # Move to device
                device_target[key] = value.to(DEVICE)
            device_targets.append(device_target)
        
        loss_dict = model(images, device_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)
def main():
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_model().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, loader, optimizer)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {loss:.4f}")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
if __name__ == "__main__":
    main()
