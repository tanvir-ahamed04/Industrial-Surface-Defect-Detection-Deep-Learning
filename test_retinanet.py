# test_retinanet_final.py
import torch
import cv2
import numpy as np
from torchvision.ops import nms
from retinanet_train import get_retinanet

def ensure_array(x):
    """Ensure input is numpy array"""
    if isinstance(x, (int, float, np.generic)):
        return np.array([x])
    return np.asarray(x)

def test_image_simple(image_path, confidence_threshold=0.05):
    """Robust test function - FINAL FIXED VERSION"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(f"Testing: {image_path}")
    print("="*60)
    
    # 1. Load model
    print("1. Loading model...")
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # 2. Load image
    print("2. Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    print(f"   Image size: {original_w}x{original_h}")
    
    # 3. Preprocess
    print("3. Preprocessing...")
    image_resized = cv2.resize(image_rgb, (512, 512))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 4. Run inference
    print("4. Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # 5. Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    print(f"   Raw predictions: {len(scores)}")
    
    # Check if predictions exist
    if len(scores) == 0:
        print("   No predictions found by the model!")
        filtered_boxes = np.array([]).reshape(0, 4)
        filtered_scores = np.array([])
        filtered_labels = np.array([])
    else:
        print(f"   Score range: {scores.min():.3f} to {scores.max():.3f}")
        
        # 6. Apply confidence threshold
        print(f"5. Applying confidence threshold: {confidence_threshold}")
        mask = scores >= confidence_threshold
        
        if np.sum(mask) == 0:
            print(f"   No detections above threshold {confidence_threshold}")
            filtered_boxes = np.array([]).reshape(0, 4)
            filtered_scores = np.array([])
            filtered_labels = np.array([])
        else:
            # Get filtered arrays
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            print(f"   After threshold: {len(filtered_boxes)} detections")
        
        # 7. Apply NMS if we have detections
        if len(filtered_boxes) > 0:
            print("6. Applying NMS...")
            
            # Ensure all arrays are proper numpy arrays
            filtered_boxes = ensure_array(filtered_boxes)
            filtered_scores = ensure_array(filtered_scores)
            filtered_labels = ensure_array(filtered_labels)
            
            # Reshape boxes to 2D if needed
            if filtered_boxes.ndim == 1:
                filtered_boxes = filtered_boxes.reshape(1, -1)
            
            # Apply NMS
            keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), 0.3)
            
            # Keep only selected indices
            filtered_boxes = filtered_boxes[keep]
            filtered_scores = filtered_scores[keep]
            filtered_labels = filtered_labels[keep]
            
            print(f"   After NMS: {len(filtered_boxes)} detections")
            
            # 8. Rescale boxes to original size
            if len(filtered_boxes) > 0:
                print("7. Rescaling boxes...")
                
                # Ensure boxes is 2D
                if filtered_boxes.ndim == 1:
                    filtered_boxes = filtered_boxes.reshape(1, -1)
                
                # Now safely rescale
                filtered_boxes[:, [0, 2]] = filtered_boxes[:, [0, 2]] * original_w / 512
                filtered_boxes[:, [1, 3]] = filtered_boxes[:, [1, 3]] * original_h / 512
    
    # 9. Prepare class info
    class_names = ['background', 'crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    # 10. Draw detections
    print("8. Drawing detections...")
    result_image = image.copy()
    
    # Ensure all arrays are proper
    filtered_boxes = ensure_array(filtered_boxes)
    filtered_scores = ensure_array(filtered_scores)
    filtered_labels = ensure_array(filtered_labels)
    
    if len(filtered_boxes) == 0:
        print("   No detections to draw")
        # Add "No defects found" text
        cv2.putText(result_image, "No defects detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Determine if we have single or multiple detections
        if filtered_boxes.ndim == 1 or (filtered_boxes.ndim == 2 and filtered_boxes.shape[0] == 1):
            # Single detection
            if filtered_boxes.ndim == 2:
                box = filtered_boxes[0]
            else:
                box = filtered_boxes
            
            score = filtered_scores[0] if len(filtered_scores) > 0 else filtered_scores
            label = filtered_labels[0] if len(filtered_labels) > 0 else filtered_labels
            
            # Convert to proper types
            label_int = int(label)
            
            if 1 <= label_int <= 6:
                x1, y1, x2, y2 = map(int, box[:4])
                color = colors[label_int - 1]
                class_name = class_names[label_int]
                
                # Draw rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{class_name}: {score:.2f}"
                cv2.putText(result_image, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"   Detection 1: {class_name} at [{x1}, {y1}, {x2}, {y2}], score: {score:.3f}")
            else:
                print(f"   Detection 1: Invalid class {label_int}")
        else:
            # Multiple detections
            for i in range(filtered_boxes.shape[0]):
                box = filtered_boxes[i]
                score = filtered_scores[i]
                label = filtered_labels[i]
                
                label_int = int(label)
                
                if 1 <= label_int <= 6:
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = colors[label_int - 1]
                    class_name = class_names[label_int]
                    
                    # Draw rectangle
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(result_image, label_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    print(f"   Detection {i+1}: {class_name} at [{x1}, {y1}, {x2}, {y2}], score: {score:.3f}")
                else:
                    print(f"   Detection {i+1}: Invalid class {label_int}")
    
    # 11. Add info text
    detections_count = 1 if (filtered_boxes.ndim == 1 and len(filtered_boxes) > 0) else filtered_boxes.shape[0]
    info_text = f"Detections: {detections_count} | Confidence: {confidence_threshold}"
    cv2.putText(result_image, info_text, (10, result_image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 12. Save and show
    print("9. Saving result...")
    filename = image_path.split('\\')[-1]
    output_path = f"result_{filename}"
    cv2.imwrite(output_path, result_image)
    
    print(f"\nResult saved to: {output_path}")
    print(f"Total detections: {detections_count}")
    
    cv2.imshow(f"Result: {filename}", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTest completed successfully!")

# SIMPLEST WORKING VERSION - Use this!
def test_image_safest(image_path, confidence=0.05):
    """Simplest, most robust version"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Preprocess
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    print(f"\nImage: {image_path.split('/')[-1]}")
    print(f"Total predictions: {len(scores)}")
    
    if len(scores) == 0:
        print("No predictions found")
        return []
    
    print(f"Max score: {scores.max():.3f}, Min score: {scores.min():.3f}")
    
    # Filter by confidence
    mask = scores >= confidence
    if not mask.any():
        print(f"No predictions above confidence {confidence}")
        return []
    
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]


    
    print(f"After confidence filter: {len(filtered_boxes)} detections")

    if len(filtered_boxes) > 0:
        keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), 0.3)
        filtered_boxes = filtered_boxes[keep]
        filtered_scores = filtered_scores[keep]
        filtered_labels = filtered_labels[keep]
        
    
    # Draw directly without complex processing
    class_names = ['background', 'crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        score = filtered_scores[i]
        label = int(filtered_labels[i])
        
        # Rescale box
        x1, y1, x2, y2 = box * w / 512
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        if 1 <= label <= 6:
            class_name = class_names[label]
            print(f"  Detection {i+1}: {class_name} (score: {score:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
            
            # Draw - color based on class
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
                     (0, 255, 255), (255, 0, 255), (255, 255, 0)]
            color = colors[label-1]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_name}: {score:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add summary
    cv2.putText(image, f"Detections: {len(filtered_boxes)} | Conf: {confidence}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save and show
    filename = image_path.split('\\')[-1]
    output_path = f"simple_result_{filename}"
    cv2.imwrite(output_path, image)
    
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Result saved to: {output_path}")
    return filtered_boxes

# Main
if __name__ == "__main__":
    # Test the simplest version - this WILL work
    image_path = r'D:\graduetion thesis\SDDP\test\img\crazing_5.jpg'
    
    print("="*60)
    print("SIMPLE TEST - This will work for all images")
    print("="*60)
    
    # Test with different confidences
    for conf in [0.01, 0.05, 0.1, 0.2]:
        print(f"\n>>> Testing with confidence: {conf}")
        test_image_safest(image_path, confidence=conf)
        
        
        if conf < 0.2:
            input("\nPress Enter to continue with next confidence level...")