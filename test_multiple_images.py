# test_multiple_images.py
import torch
import cv2
import numpy as np
import os
from torchvision.ops import nms
from retinanet_train import get_retinanet

def test_all_images(folder_path, confidence=0.05):
    """Test all images in a folder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # Get all images
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    
    class_names = ['background', 'crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    
    print(f"Testing {len(images)} images with confidence threshold: {confidence}")
    print("="*80)
    
    results = []
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        
        # Get expected class from filename
        expected_class = None
        for cls in class_names[1:]:  # Skip background
            if cls in img_name.lower():
                expected_class = cls
                break
        
        # Process image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        image_resized = cv2.resize(image_rgb, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Apply threshold
        mask = scores >= confidence
        if np.sum(mask) > 0:
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            
            # Apply NMS
            if len(filtered_boxes) > 0:
                keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), 0.3)
                filtered_boxes = filtered_boxes[keep]
                filtered_scores = filtered_scores[keep]
                filtered_labels = filtered_labels[keep]
            
            # Get detected classes
            detected_classes = []
            for label in filtered_labels:
                if 1 <= label <= 6:
                    detected_classes.append(class_names[label])
            
            # Check if correct class detected
            correct = expected_class in detected_classes if expected_class else False
            
            # Store result
            results.append({
                'image': img_name,
                'expected': expected_class,
                'detected': list(set(detected_classes)),  # Unique classes
                'count': len(filtered_boxes),
                'correct': correct,
                'scores': filtered_scores.tolist()
            })
            
            # Print
            print(f"{img_name:30} | Expected: {expected_class or 'Unknown':15} | "
                  f"Detected: {', '.join(set(detected_classes)) or 'None':20} | "
                  f"Count: {len(filtered_boxes):2d} | "
                  f"{'✓' if correct else '✗'}")
        else:
            results.append({
                'image': img_name,
                'expected': expected_class,
                'detected': [],
                'count': 0,
                'correct': False,
                'scores': []
            })
            print(f"{img_name:30} | Expected: {expected_class or 'Unknown':15} | "
                  f"Detected: {'None':20} | Count: 0  | ✗")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_detected = sum(1 for r in results if r['count'] > 0)
    total_correct = sum(1 for r in results if r['correct'])
    
    print(f"Images tested: {len(images)}")
    print(f"Images with detections: {total_detected} ({total_detected/len(images)*100:.1f}%)")
    print(f"Correct classifications: {total_correct} ({total_correct/len(images)*100:.1f}%)")
    
    # Show confusion
    print("\nMost common misclassifications:")
    misclassifications = {}
    for r in results:
        if not r['correct'] and r['expected'] and r['detected']:
            key = f"{r['expected']} -> {r['detected'][0] if r['detected'] else 'None'}"
            misclassifications[key] = misclassifications.get(key, 0) + 1
    
    for mis, count in sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {mis}: {count} times")

# Run
folder_path = r'D:\graduetion thesis\SDDP\test\img'
test_all_images(folder_path, confidence=0.05)