import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import *
from dataset import NEUDETDataset
from utils import collate_fn
from model import get_model
from torchvision.models.detection import retinanet_resnet50_fpn
from evaluate import evaluate_model
def load_retinanet():
    model = retinanet_resnet50_fpn(
        weights="DEFAULT",
        num_classes=NUM_CLASSES
    )
    return model
def main():
    dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, collate_fn=collate_fn)

    results = []

    # Faster R-CNN
    frcnn = get_model().to(DEVICE)
    frcnn.load_state_dict(torch.load("model_epoch_30.pth"))
    recall_frcnn = evaluate_model(frcnn, loader)

    results.append({
        "Model": "Faster R-CNN",
        "Recall@0.5": recall_frcnn
    })

    # RetinaNet
    retinanet = load_retinanet().to(DEVICE)
    retinanet.load_state_dict(torch.load("retinanet_epoch_30.pth"))
    recall_retina = evaluate_model(retinanet, loader)

    results.append({
        "Model": "RetinaNet (Focal Loss)",
        "Recall@0.5": recall_retina
    })

    df = pd.DataFrame(results)
    df.to_csv("model_comparison.csv", index=False)
    print(df)
if __name__ == "__main__":
    main()


# Loads Faster R-CNN and RetinaNet

# Evaluates Recall @ IoU=0.5

# Saves results to a CSV (for thesis tables)