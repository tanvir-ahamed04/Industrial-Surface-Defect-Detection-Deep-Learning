# generate_test_scenarios.py
import cv2
import numpy as np
import random
from pathlib import Path

def create_test_scenarios(input_folder, output_folder):
    """Generate various test scenarios"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*.jpg')) + \
                  list(Path(input_folder).glob('*.png'))
    
    for img_path in image_files[:10]:  # Process first 10 images
        image = cv2.imread(str(img_path))
        
        # 1. Original
        cv2.imwrite(f"{output_folder}/{img_path.stem}_original.jpg", image)
        
        # 2. Low Light
        low_light = (image * 0.3).astype(np.uint8)
        cv2.imwrite(f"{output_folder}/{img_path.stem}_low_light.jpg", low_light)
        
        # 3. High Contrast
        high_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=50)
        cv2.imwrite(f"{output_folder}/{img_path.stem}_high_contrast.jpg", high_contrast)
        
        # 4. Blurred (simulating motion blur)
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        cv2.imwrite(f"{output_folder}/{img_path.stem}_blurred.jpg", blurred)
        
        # 5. Noisy
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        cv2.imwrite(f"{output_folder}/{img_path.stem}_noisy.jpg", noisy)
        
        # 6. Rotated
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(f"{output_folder}/{img_path.stem}_rotated.jpg", rotated)
        
        print(f"Generated test scenarios for {img_path.name}")

if __name__ == "__main__":
    # Update these paths
    create_test_scenarios("IMAGES/", "real_world_test/scenarios/")