# run_pipeline.py
import argparse

import cv2
import torch
from torch.utils.data import DataLoader

from config import DEVICE, IMAGE_DIR, ANNOTATION_DIR
from dataset import NEUDETDataset
from metrics import calculate_map
from retinanet_train import get_retinanet
from retinanet_train_val import train_with_validation
from test_retinanet import predict, visualize_results
from utils import collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'test', 'inference'])
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--model', type=str, default='best_retinanet.pth')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_validation()
    
    elif args.mode == 'test':
        model = get_retinanet().to(DEVICE)
        model.load_state_dict(torch.load(args.model))
        
        # Load test dataset
        test_dataset = NEUDETDataset(IMAGE_DIR, ANNOTATION_DIR)
        test_loader = DataLoader(test_dataset, batch_size=2, 
                                shuffle=False, collate_fn=collate_fn)
        
        # Calculate metrics
        metrics = calculate_map(model, test_loader, DEVICE)
        print("Test Results:")
        print(f"mAP: {metrics['map']:.4f}")
        print(f"mAP50: {metrics['map_50']:.4f}")
        print(f"mAP75: {metrics['map_75']:.4f}")
    
    elif args.mode == 'inference':
        if not args.image:
            print("Please provide --image argument")
            return
            
        model = get_retinanet().to(DEVICE)
        model.load_state_dict(torch.load(args.model))
        
        # Run inference
        boxes, scores, labels = predict(model, args.image)
        
        # Visualize
        class_names = ['background', 'slag', 'broken', 'crazing', 'inclusion', 'patches', 'pitted']
        result_image = visualize_results(args.image, boxes, scores, labels, class_names)
        
        # Save and show
        cv2.imwrite('result.jpg', result_image)
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()