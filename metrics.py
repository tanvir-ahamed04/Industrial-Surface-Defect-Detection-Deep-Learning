# metrics.py
import torch
from torchvision.ops import box_iou
import numpy as np
from collections import defaultdict

def calculate_ap(precision, recall):
    """Calculate Average Precision from precision-recall curve"""
    # Append sentinel values at beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Calculate AP as the area under the curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_map(model, dataloader, device, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision) manually
    """
    model.eval()
    
    # Store all predictions and ground truths
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Filter predictions by confidence
                conf_mask = pred['scores'] > 0.5
                pred_boxes = pred['boxes'][conf_mask].cpu()
                pred_scores = pred['scores'][conf_mask].cpu()
                pred_labels = pred['labels'][conf_mask].cpu()
                
                # Store predictions
                all_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                
                # Store ground truths
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
    
    # Calculate mAP for each class
    aps = []
    num_classes = len(set([label.item() for target in all_targets for label in target['labels']]))
    
    for class_id in range(1, num_classes + 1):  # Skip background (0)
        class_predictions = []
        class_targets = []
        
        # Collect predictions and targets for this class
        for pred, target in zip(all_predictions, all_targets):
            # Get predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                class_predictions.append({
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask]
                })
            else:
                class_predictions.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0)
                })
            
            # Get targets for this class
            class_mask = target['labels'] == class_id
            class_targets.append({
                'boxes': target['boxes'][class_mask] if class_mask.any() else torch.empty((0, 4))
            })
        
        # Calculate AP for this class
        ap = calculate_ap_for_class(class_predictions, class_targets, iou_threshold)
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0.0
    return mAP

def calculate_ap_for_class(predictions, targets, iou_threshold):
    """Calculate AP for a single class"""
    # Flatten all predictions
    all_boxes = []
    all_scores = []
    all_image_ids = []
    
    for img_id, pred in enumerate(predictions):
        for box, score in zip(pred['boxes'], pred['scores']):
            all_boxes.append(box)
            all_scores.append(score)
            all_image_ids.append(img_id)
    
    if not all_scores:
        return 0.0
    
    # Sort by confidence
    sorted_indices = np.argsort(all_scores)[::-1]
    all_boxes = [all_boxes[i] for i in sorted_indices]
    all_scores = [all_scores[i] for i in sorted_indices]
    all_image_ids = [all_image_ids[i] for i in sorted_indices]
    
    # Track which ground truths have been matched
    gt_matched = defaultdict(set)
    
    true_positives = []
    false_positives = []
    
    for box, score, img_id in zip(all_boxes, all_scores, all_image_ids):
        gt_boxes = targets[img_id]['boxes']
        
        if len(gt_boxes) == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue
        
        # Calculate IoU with all ground truth boxes
        ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
        max_iou, max_idx = torch.max(ious, dim=0)
        
        if max_iou >= iou_threshold and max_idx not in gt_matched[img_id]:
            true_positives.append(1)
            false_positives.append(0)
            gt_matched[img_id].add(max_idx)
        else:
            true_positives.append(0)
            false_positives.append(1)
    
    # Calculate precision and recall
    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    
    # Cumulative sums
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)
    
    # Calculate precision and recall
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / (sum([len(t['boxes']) for t in targets]) + 1e-10)
    
    # Calculate AP
    ap = calculate_ap(precision, recall)
    return ap