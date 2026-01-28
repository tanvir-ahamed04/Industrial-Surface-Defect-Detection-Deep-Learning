# thesis_results_current.py
import torch
import cv2
import numpy as np
import os
from retinanet_train import get_retinanet

def create_thesis_results_current():
    """Create thesis results with current model (epoch 30)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # Test images
    test_cases = [
        ('crazing_5.jpg', 'crazing', 'Partial Success'),
        ('patches_224.jpg', 'patches', 'Misclassified'),
        ('pitted_surface_184.jpg', 'pitted_surface', 'Failed'),
        ('rolled-in_scale_13.jpg', 'rolled-in_scale', 'Success'),
        ('rolled-in_scale_226.jpg', 'rolled-in_scale', 'Failed'),
        ('scratches_300.jpg', 'scratches', 'Failed')
    ]
    
    print("="*80)
    print("THESIS RESULTS - CURRENT MODEL (30 Epochs)")
    print("="*80)
    print(f"{'Image':25} {'Expected':12} {'Result':15} {'Detection':12} {'Confidence':10}")
    print("-"*80)
    
    results = []
    
    for img_name, expected_class, expected_result in test_cases:
        img_path = f"D:\\graduetion thesis\\SDDP\\test\\img\\{img_name}"
        
        # Load image
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        # Process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Find best detection (score > 0.05, size 20-80px)
        best_detection = None
        best_score = 0
        
        for i in range(len(scores)):
            if scores[i] >= 0.05:
                box = predictions['boxes'].cpu().numpy()[i] * w / 512
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                
                if 20 <= width <= 80 and 20 <= height <= 80:
                    label = int(labels[i])
                    if 1 <= label <= 6:
                        class_names = ['crazing', 'inclusion', 'patches', 
                                      'pitted_surface', 'rolled-in_scale', 'scratches']
                        class_name = class_names[label-1]
                        
                        if scores[i] > best_score:
                            best_score = scores[i]
                            best_detection = {
                                'class': class_name,
                                'score': scores[i],
                                'size': f"{width}x{height}"
                            }
        
        # Determine result
        if best_detection:
            if best_detection['class'] == expected_class:
                result = "✓ Correct"
                status = "Success"
            else:
                result = f"✗ Wrong ({best_detection['class']})"
                status = "Misclassified"
            
            detection_info = f"{best_detection['class']}"
            confidence = f"{best_detection['score']:.3f}"
        else:
            result = "✗ No detection"
            status = "Failed"
            detection_info = "None"
            confidence = "N/A"
        
        # Store
        results.append({
            'image': img_name,
            'expected': expected_class,
            'result': status,
            'detection': detection_info,
            'confidence': confidence
        })
        
        # Print
        print(f"{img_name:25} {expected_class:12} {status:15} {detection_info:12} {confidence:10}")
        
        # Draw on image
        result_image = image.copy()
        
        if best_detection:
            # Find and draw the best detection
            for i in range(len(scores)):
                if scores[i] == best_score:
                    box = predictions['boxes'].cpu().numpy()[i] * w / 512
                    x1, y1, x2, y2 = map(int, box)
                    
                    color = (0, 255, 0) if best_detection['class'] == expected_class else (0, 0, 255)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{best_detection['class']}: {best_detection['score']:.2f}"
                    cv2.putText(result_image, label_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break
        
        # Add status
        status_color = (0, 255, 0) if status == "Success" else (0, 0, 255)
        status_text = f"{expected_class}: {status}"
        cv2.putText(result_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Save
        cv2.imwrite(f"thesis_current_{img_name}", result_image)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successes = sum(1 for r in results if r['result'] == "Success")
    misclassified = sum(1 for r in results if r['result'] == "Misclassified")
    failed = sum(1 for r in results if r['result'] == "Failed")
    
    print(f"Total images: {len(results)}")
    print(f"Success: {successes} ({successes/len(results)*100:.1f}%)")
    print(f"Misclassified: {misclassified} ({misclassified/len(results)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(results)*100:.1f}%)")
    
    if successes > 0:
        avg_confidence = np.mean([float(r['confidence']) for r in results if r['confidence'] != 'N/A'])
        print(f"Average confidence: {avg_confidence:.3f}")
    
    print(f"\nImages saved with 'thesis_current_' prefix")

if __name__ == "__main__":
    create_thesis_results_current()