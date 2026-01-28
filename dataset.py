import os
import random
import cv2
import torch
import xml.etree.ElementTree as ET
import numpy as np  # Add this import
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augmentations import copy_paste_augmentation
from config import CLASS_TO_IDX, IMAGE_SIZE

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[label])

    return boxes, labels

class NEUDETDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted([f for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc",
                                   label_fields=["labels"]))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(
            self.annotation_dir,
            img_name.replace(".jpg", ".xml").replace(".png", ".xml")
        )

        # Check if XML exists
        if not os.path.exists(xml_path):
            print(f"Warning: XML not found for {img_name}")
            # Return empty targets
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=image, bboxes=[], labels=[])
            image = transformed["image"]
            
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return image, target

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = parse_voc_xml(xml_path)

        # Apply transformations
        transformed = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels
        )

        image = transformed["image"]
        
        # Convert boxes and labels to tensors
        if transformed["bboxes"]:
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 
                    if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # Apply copy-paste augmentation (with proper tensor conversion)
        if random.random() < 0.5 and len(self.images) > 1:
            idx2 = random.randint(0, len(self.images) - 1)
            while idx2 == idx:  # Ensure different image
                idx2 = random.randint(0, len(self.images) - 1)
            
            # Get second image
            img2_name = self.images[idx2]
            img2_path = os.path.join(self.image_dir, img2_name)
            xml2_path = os.path.join(
                self.annotation_dir,
                img2_name.replace(".jpg", ".xml").replace(".png", ".xml")
            )
            
            if os.path.exists(xml2_path):
                image2 = cv2.imread(img2_path)
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                boxes2, labels2 = parse_voc_xml(xml2_path)
                
                # Transform second image
                transformed2 = self.transform(
                    image=image2,
                    bboxes=boxes2,
                    labels=labels2
                )
                
                image2_tensor = transformed2["image"]
                
                # Convert to numpy for augmentation
                # Note: Albumentations returns normalized tensors, need to denormalize
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image2_np = image2_tensor.permute(1, 2, 0).cpu().numpy()
                
                # Denormalize for augmentation
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image2_np = image2_np * std + mean
                
                # Ensure values are in valid range
                image_np = np.clip(image_np, 0, 1)
                image2_np = np.clip(image2_np, 0, 1)
                
                # Prepare target2
                if transformed2["bboxes"]:
                    boxes2_tensor = torch.tensor(transformed2["bboxes"], dtype=torch.float32)
                    labels2_tensor = torch.tensor(transformed2["labels"], dtype=torch.int64)
                else:
                    boxes2_tensor = torch.zeros((0, 4), dtype=torch.float32)
                    labels2_tensor = torch.zeros((0,), dtype=torch.int64)
                
                target2 = {
                    "boxes": boxes2_tensor,
                    "labels": labels2_tensor,
                    "image_id": torch.tensor([idx2]),
                    "area": (boxes2_tensor[:, 3] - boxes2_tensor[:, 1]) * 
                            (boxes2_tensor[:, 2] - boxes2_tensor[:, 0]) 
                            if len(boxes2_tensor) > 0 else torch.zeros((0,), dtype=torch.float32),
                    "iscrowd": torch.zeros((len(boxes2_tensor),), dtype=torch.int64)
                }
                
                # Apply copy-paste augmentation
                image_np, target = copy_paste_augmentation(
                    image_np, target, image2_np, target2
                )
                
                # Convert back to tensor and normalize
                image_np = np.clip(image_np, 0, 1)
                image_np = (image_np - mean) / std
                image = torch.tensor(image_np).permute(2, 0, 1).float()

        return image, target