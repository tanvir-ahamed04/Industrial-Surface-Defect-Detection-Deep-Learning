# test_with_lower_threshold.py
import torch
import cv2
import numpy as np
from retinanet_train import get_retinanet

def test_with_lower_threshold():
    """Test with lower confidence threshold (0.05)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load current model (epoch 30)
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # Test all images with 0.05 threshold
    test_images = [
        ('crazing_5.jpg', 'crazing'),
        ('patches_224.jpg', 'patches'),
        ('pitted_surface_184.jpg', 'pitted_surface'),
        ('rolled-in_scale_13.jpg', 'rolled-in_scale'),
        ('rolled-in_scale_226.jpg', 'rolled-in_scale'),
        ('scratches_300.jpg', 'scratches')
    ]
    
    print("="*80)
    print("TESTING WITH LOWER CONFIDENCE THRESHOLD (0.05)")
    print("="*80)
    
    for img_name, expected_class in test_images:
        img_path = f"D:\\graduetion thesis\\SDDP\\test\\img\\{img_name}"
        
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        print(f"\n{img_name:30} | Expected: {expected_class}")
        print(f"  Raw predictions: {len(scores)}")
        
        if len(scores) == 0:
            print("  No predictions")
            continue
        
        # Use 0.05 threshold (lower)
        valid_detections = []
        for i in range(len(scores)):
            if scores[i] >= 0.05:  # Lower threshold
                # Rescale
                box = boxes[i] * w / 512
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                
                # Size filter
                if 20 <= width <= 80 and 20 <= height <= 80:
                    label = int(labels[i])
                    if 1 <= label <= 6:
                        class_names = ['crazing', 'inclusion', 'patches', 
                                      'pitted_surface', 'rolled-in_scale', 'scratches']
                        class_name = class_names[label-1]
                        
                        valid_detections.append({
                            'class': class_name,
                            'score': scores[i],
                            'size': f"{width}x{height}"
                        })
        
        if valid_detections:
            # Sort by score
            valid_detections.sort(key=lambda x: x['score'], reverse=True)
            print(f"  Valid detections: {len(valid_detections)}")
            
            for det in valid_detections[:3]:  # Show top 3
                correct = "✓" if det['class'] == expected_class else "✗"
                print(f"    {correct} {det['class']}: {det['score']:.3f} ({det['size']})")
        else:
            print("  No valid detections")

if __name__ == "__main__":
    test_with_lower_threshold()