# real_world_test.py
import torch
import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.ops import nms
from retinanet_train import get_retinanet

class RealWorldTester:
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.class_names = ['background', 'crazing', 'inclusion', 
                           'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        
    def load_model(self, model_path):
        """Load trained model"""
        model = get_retinanet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        original_h, original_w = image.shape[:2]
        image_resized = cv2.resize(image, (512, 512))
        
        # Normalize
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_w, original_h
    
    def predict(self, image):
        """Run inference on single image"""
        with torch.no_grad():
            image_tensor, orig_w, orig_h = self.preprocess_image(image)
            predictions = self.model(image_tensor)[0]
        
        return self.postprocess(predictions, orig_w, orig_h)
    
    def postprocess(self, predictions, orig_w, orig_h):
        """Process model output"""
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Apply confidence threshold
        mask = scores >= self.conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Apply NMS
        if len(boxes) > 0:
            keep = nms(
                torch.tensor(boxes),
                torch.tensor(scores),
                self.iou_thresh
            )
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        
        # Rescale boxes to original size
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * orig_w / 512
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * orig_h / 512
        
        return boxes, scores, labels
    
    def visualize(self, image, boxes, scores, labels, save_path=None):
        """Visualize predictions on image"""
        img_display = image.copy()
        
        # Colors for different classes
        colors = {
            1: (255, 0, 0),    # Red - crazing
            2: (0, 255, 0),    # Green - inclusion
            3: (0, 0, 255),    # Blue - patches
            4: (255, 255, 0),  # Cyan - pitted_surface
            5: (255, 0, 255),  # Magenta - rolled-in_scale
            6: (0, 255, 255),  # Yellow - scratches
        }
        
        for box, score, label in zip(boxes, scores, labels):
            if score < self.conf_thresh:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(label, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_name = self.class_names[label] if label < len(self.class_names) else f'Class {label}'
            label_text = f"{label_name}: {score:.2f}"
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(img_display, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Text
            cv2.putText(img_display, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add detection summary
        summary = f"Detections: {len(boxes)} | Confidence: {self.conf_thresh}"
        cv2.putText(img_display, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_display)
        
        return img_display
    
    def test_single_image(self, image_path):
        """Test on single image"""
        print(f"\nTesting: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Run inference
        start_time = time.time()
        boxes, scores, labels = self.predict(image)
        inference_time = time.time() - start_time
        
        # Display results
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Detections: {len(boxes)}")
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            print(f"  {i+1}: {self.class_names[label]} - Confidence: {score:.3f} - Box: {box.astype(int)}")
        
        # Visualize
        result = self.visualize(image, boxes, scores, labels)
        
        # Display
        cv2.imshow('Detection Result', result)
        cv2.waitKey(0)
        
        return boxes, scores, labels
    
    def test_folder(self, folder_path, save_results=True):
        """Test all images in a folder"""
        folder = Path(folder_path)
        image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + \
                     list(folder.glob('*.jpeg')) + list(folder.glob('*.bmp'))
        
        print(f"\nTesting {len(image_files)} images in {folder_path}")
        
        results = []
        total_time = 0
        
        for img_path in image_files:
            start_time = time.time()
            image = cv2.imread(str(img_path))
            
            if image is None:
                continue
            
            boxes, scores, labels = self.predict(image)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Save visualization
            if save_results:
                save_path = f"results/{img_path.stem}_result.jpg"
                self.visualize(image, boxes, scores, labels, save_path)
            
            results.append({
                'image': img_path.name,
                'detections': len(boxes),
                'time': inference_time,
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
            print(f"  {img_path.name}: {len(boxes)} detections in {inference_time:.3f}s")
        
        # Summary
        avg_time = total_time / len(image_files) if image_files else 0
        total_detections = sum(r['detections'] for r in results)
        
        print(f"\nSummary:")
        print(f"Total images: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"FPS: {1/avg_time:.1f}" if avg_time > 0 else "FPS: N/A")
        
        return results
    
    def test_video(self, video_path, output_path='output_video.mp4'):
        """Test on video file"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            start_time = time.time()
            boxes, scores, labels = self.predict(frame)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Visualize
            result_frame = self.visualize(frame, boxes, scores, labels)
            
            # Write frame
            out.write(result_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        avg_time = total_time / frame_count if frame_count > 0 else 0
        
        print(f"\nVideo processing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Average inference time per frame: {avg_time:.3f}s")
        print(f"Output saved to: {output_path}")
        
        return output_path
    
    def live_camera_test(self, camera_id=0):
        """Real-time testing with webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        print("Live camera testing - Press 'q' to quit")
        
        fps_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Run inference
            boxes, scores, labels = self.predict(frame)
            
            inference_time = time.time() - start_time
            fps = 1 / inference_time if inference_time > 0 else 0
            fps_history.append(fps)
            
            # Visualize
            result_frame = self.visualize(frame, boxes, scores, labels)
            
            # Display FPS
            avg_fps = np.mean(fps_history[-30:]) if len(fps_history) > 0 else 0
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Live Defect Detection', result_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-world defect detection testing')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['image', 'folder', 'video', 'live'],
                       help='Test mode')
    parser.add_argument('--input', type=str, help='Input path (image/folder/video)')
    parser.add_argument('--model', type=str, default='best_retinanet.pth',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Initialize tester
    tester = RealWorldTester(args.model, confidence_threshold=args.conf)
    
    if args.mode == 'image':
        if not args.input:
            print("Please provide --input for image mode")
            return
        tester.test_single_image(args.input)
    
    elif args.mode == 'folder':
        if not args.input:
            print("Please provide --input for folder mode")
            return
        tester.test_folder(args.input)
    
    elif args.mode == 'video':
        if not args.input:
            print("Please provide --input for video mode")
            return
        tester.test_video(args.input)
    
    elif args.mode == 'live':
        tester.live_camera_test()

if __name__ == "__main__":
    main()