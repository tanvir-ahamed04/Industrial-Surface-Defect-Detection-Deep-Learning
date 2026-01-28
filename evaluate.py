import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from config import *
from dataset import NEUDETDataset
from model import get_model
from utils import collate_fn
def evaluate_model(model, loader, iou_thresh=0.5, score_thresh=0.5):
    model.eval()

    true_positives = 0
    false_negatives = 0
    total_gt = 0

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(DEVICE)
                total_gt += len(gt_boxes)

                if len(output["boxes"]) == 0:
                    false_negatives += len(gt_boxes)
                    continue

                pred_boxes = output["boxes"]
                scores = output["scores"]

                keep = scores >= score_thresh
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = ious.max(dim=0)

                true_positives += (max_ious >= iou_thresh).sum().item()
                false_negatives += (max_ious < iou_thresh).sum().item()

    recall = true_positives / (true_positives + false_negatives + 1e-6)
    return recall
def main():
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("model_epoch_30.pth"))
    model.eval()

    recall = evaluate_model(model, loader)
    print(f"\nDefect Recall @ IoU=0.5 : {recall:.4f}")
if __name__ == "__main__":
    main()
