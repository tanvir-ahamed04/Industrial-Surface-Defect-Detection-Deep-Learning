# test_retinanet_final_fixed.py
import torch
import cv2
import numpy as np
from torchvision.ops import nms
from retinanet_train import get_retinanet

def ensure_2d_array(arr):
    """Ensure array is at least 2D"""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 1:
            return arr.reshape(-1, 1) if arr.shape[0] > 4 else arr.reshape(1, -1)
        return arr
    else:
        # Scalar or list
        return np.array([arr]) if np.isscalar(arr) else np.array(arr)

def ensure_1d_array(arr):
    """Ensure array is 1D"""
    if isinstance(arr, np.ndarray):
        if arr.ndim > 1:
            return arr.flatten()
        return arr
    else:
        return np.array([arr]) if np.isscalar(arr) else np.array(arr)

def filter_by_size(boxes, scores, labels, min_size=20, max_size=100):
    """Filter out boxes that are too small or too large"""
    # Ensure arrays are proper
    boxes = ensure_2d_array(boxes)
    scores = ensure_1d_array(scores)
    labels = ensure_1d_array(labels)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    for i in range(len(boxes)):
        box = boxes[i] if boxes.ndim == 2 else boxes
        score = scores[i] if len(scores) > 1 else scores[0] if len(scores) == 1 else scores
        label = labels[i] if len(labels) > 1 else labels[0] if len(labels) == 1 else labels
        
        # Ensure box has 4 elements
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1
            
            # Keep only reasonably sized boxes
            if min_size <= width <= max_size and min_size <= height <= max_size:
                filtered_boxes.append(box[:4])
                filtered_scores.append(score)
                filtered_labels.append(label)
    
    if len(filtered_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])
    
    return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_labels)

def test_image_improved(image_path, confidence_threshold=0.1, 
                       iou_threshold=0.5, min_box_size=20, max_box_size=80):
    """Improved test function - FINAL FIXED VERSION"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(f"Testing: {image_path}")
    print(f"Parameters: Confidence={confidence_threshold}, "
          f"IoU={iou_threshold}, Box Size={min_box_size}-{max_box_size}px")
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
        return image, [], [], []
    
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
    
    if len(scores) == 0:
        print("   No predictions found!")
        return image, [], [], []
    
    print(f"   Score range: {scores.min():.3f} to {scores.max():.3f}")
    
    # 6. Apply confidence threshold
    print(f"5. Applying confidence threshold: {confidence_threshold}")
    mask = scores >= confidence_threshold
    
    if np.sum(mask) == 0:
        print(f"   No detections above threshold {confidence_threshold}")
        return image, [], [], []
    
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    
    print(f"   After confidence filter: {len(filtered_boxes)} detections")
    
    # 7. Rescale boxes to original size
    print("6. Rescaling boxes to original size...")
    filtered_boxes = filtered_boxes * original_w / 512
    
    # 8. Apply size filtering
    print(f"7. Applying size filtering ({min_box_size}-{max_box_size}px)...")
    filtered_boxes, filtered_scores, filtered_labels = filter_by_size(
        filtered_boxes, filtered_scores, filtered_labels,
        min_size=min_box_size, max_size=max_box_size
    )
    
    print(f"   After size filtering: {len(filtered_boxes)} detections")
    
    if len(filtered_boxes) == 0:
        print("   No detections after size filtering")
        return image, [], [], []
    
    # 9. Apply NMS - Handle single vs multiple boxes
    print(f"8. Applying NMS (IoU threshold: {iou_threshold})...")
    
    # Ensure boxes are 2D for NMS
    if filtered_boxes.ndim == 1:
        filtered_boxes = filtered_boxes.reshape(1, -1)
    
    if len(filtered_boxes) > 0:
        keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), iou_threshold)
        
        # Apply NMS - handle scalar indices
        if keep.numel() == 1:  # Single element
            filtered_boxes = filtered_boxes[keep.item():keep.item()+1]
            filtered_scores = filtered_scores[keep.item():keep.item()+1]
            filtered_labels = filtered_labels[keep.item():keep.item()+1]
        else:
            filtered_boxes = filtered_boxes[keep]
            filtered_scores = filtered_scores[keep]
            filtered_labels = filtered_labels[keep]
    
    print(f"   After NMS: {len(filtered_boxes)} final detections")
    
    # 10. Prepare class info
    class_names = ['background', 'crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), 
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    # 11. Draw detections
    print("9. Drawing detections...")
    result_image = image.copy()
    
    # Ensure arrays are proper for iteration
    filtered_boxes = ensure_2d_array(filtered_boxes)
    filtered_scores = ensure_1d_array(filtered_scores)
    filtered_labels = ensure_1d_array(filtered_labels)
    
    if len(filtered_boxes) == 0:
        print("   No detections to draw")
        cv2.putText(result_image, "No defects detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Handle single vs multiple detections
        num_detections = filtered_boxes.shape[0] if filtered_boxes.ndim == 2 else 1
        
        for i in range(num_detections):
            # Get detection info
            if num_detections == 1:
                box = filtered_boxes[0] if filtered_boxes.ndim == 2 else filtered_boxes
                score = filtered_scores[0] if len(filtered_scores) > 0 else filtered_scores
                label = filtered_labels[0] if len(filtered_labels) > 0 else filtered_labels
            else:
                box = filtered_boxes[i]
                score = filtered_scores[i]
                label = filtered_labels[i]
            
            label_int = int(label) if hasattr(label, '__iter__') else int(label)
            
            if 1 <= label_int <= 6:
                # Ensure box has 4 elements
                if len(box) >= 4:
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = colors[label_int - 1]
                    class_name = class_names[label_int]
                    
                    # Calculate box size
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Draw rectangle
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with size info
                    label_text = f"{class_name}: {score:.2f} ({width}x{height})"
                    cv2.putText(result_image, label_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    print(f"   Detection {i+1}: {class_name} (score: {score:.3f}) "
                          f"at [{x1}, {y1}, {x2}, {y2}], size: {width}x{height}px")
    
    # 12. Add info text
    num_final_detections = filtered_boxes.shape[0] if filtered_boxes.ndim == 2 else (1 if len(filtered_boxes) > 0 else 0)
    info_text = (f"Detections: {num_final_detections} | "
                 f"Conf: {confidence_threshold} | "
                 f"IoU: {iou_threshold} | "
                 f"Size: {min_box_size}-{max_box_size}px")
    cv2.putText(result_image, info_text, (10, result_image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 13. Save and show
    print("10. Saving result...")
    filename = image_path.split('\\')[-1]
    output_path = f"final_result_{filename}"
    cv2.imwrite(output_path, result_image)
    
    print(f"\nResult saved to: {output_path}")
    print(f"Total final detections: {num_final_detections}")
    
    cv2.imshow(f"Final Result: {filename}", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTest completed successfully!")
    
    return result_image, filtered_boxes, filtered_scores, filtered_labels

# ULTRA SIMPLE VERSION THAT ALWAYS WORKS
def test_ultra_simple():
    """Ultra simple version that always works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_retinanet().to(device)
    model.load_state_dict(torch.load('retinanet_epoch_30.pth', map_location=device, weights_only=False))
    model.eval()
    
    # Test image
    image_path = r'D:\graduetion thesis\SDDP\test\img\scratches_300.jpg'
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Preprocess
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Get ALL predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    print(f"Image: {image_path}")
    print(f"Total raw predictions: {len(scores)}")
    
    if len(scores) == 0:
        print("No predictions found")
        return
    
    # Just process manually without complex array operations
    class_names = ['crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # Simple processing - no NMS, just show top predictions
    print("\nTop predictions (score > 0.1, size 20-80px):")
    
    count = 0
    for i in range(len(scores)):
        if scores[i] >= 0.1:
            # Rescale box
            box = boxes[i] * w / 512
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            
            # Check size
            if 20 <= width <= 80 and 20 <= height <= 80:
                label = int(labels[i])
                if 1 <= label <= 6:
                    class_name = class_names[label-1]
                    print(f"  {count+1}. {class_name}: {scores[i]:.3f} "
                          f"at [{x1}, {y1}, {x2}, {y2}], size: {width}x{height}px")
                    
                    # Draw with simple color
                    color = (0, 255, 0)  # Green for all
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{class_name}: {scores[i]:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    count += 1
    
    print(f"\nTotal valid detections: {count}")
    
    # Save
    cv2.imwrite("ultra_simple_final.jpg", image)
    cv2.imshow("Ultra Simple Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Result saved to: ultra_simple_final.jpg")

# Main execution - USE THIS!
if __name__ == "__main__":
    print("="*60)
    print("ULTRA SIMPLE TEST - This will always work")
    print("="*60)
    
    # Use the ultra simple version
    test_ultra_simple()
    
    # Or use the improved version
    # image_path = r'D:\graduetion thesis\SDDP\test\img\crazing_5.jpg'
    # test_image_improved(
    #     image_path=image_path,
    #     confidence_threshold=0.1,
    #     iou_threshold=0.5,
    #     min_box_size=20,
    #     max_box_size=80
    # )