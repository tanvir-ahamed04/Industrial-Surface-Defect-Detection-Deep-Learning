import torch
import cv2
import os
from torch.utils.data import DataLoader

from config import *
from dataset import NEUDETDataset
from model import get_model
from utils import collate_fn
os.makedirs("results", exist_ok=True)
def draw_boxes(image, boxes, scores, thresh=0.6):
    image = image.copy()
    for box, score in zip(boxes, scores):
        if score < thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return image
def main():
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, collate_fn=collate_fn)

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("model_epoch_30.pth"))
    model.eval()

    with torch.no_grad():
        for idx, (images, _) in enumerate(loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            img = images[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")

            boxes = outputs[0]["boxes"].cpu().numpy()
            scores = outputs[0]["scores"].cpu().numpy()

            vis = draw_boxes(img, boxes, scores)
            cv2.imwrite(f"results/result_{idx}.png",
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if idx == 10:
                break
if __name__ == "__main__":
    main()
