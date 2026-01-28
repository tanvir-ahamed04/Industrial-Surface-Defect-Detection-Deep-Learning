# qualitative_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_qualitative_analysis():
    """Create qualitative analysis with visual examples"""
    
    # Case studies
    case_studies = [
        {
            'image': 'crazing_5.jpg',
            'title': 'Case 1: Successful Detection - Crazing',
            'description': 'Model correctly identified crazing defect with moderate confidence (0.172). The bounding box appropriately covers the defect region.',
            'status': 'Success',
            'color': 'green'
        },
        {
            'image': 'rolled-in_scale_13.jpg',
            'title': 'Case 2: High Confidence Detection - Rolled-in Scale',
            'description': 'Model achieved highest confidence (0.823) for rolled-in-scale defect. The bounding box shows accurate localization.',
            'status': 'Success',
            'color': 'green'
        },
        {
            'image': 'patches_224.jpg',
            'title': 'Case 3: Misclassification - Patches as Pitted Surface',
            'description': 'Model detected defect but misclassified patches as pitted_surface (0.106 confidence). Shows class confusion issue.',
            'status': 'Misclassified',
            'color': 'orange'
        },
        {
            'image': 'pitted_surface_184.jpg',
            'title': 'Case 4: Failure Case - No Detection',
            'description': 'Model failed to detect any defect in pitted_surface image. Likely due to insufficient training on this class.',
            'status': 'Failure',
            'color': 'red'
        }
    ]
    
    print("="*80)
    print("QUALITATIVE ANALYSIS - Case Studies")
    print("="*80)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, case in enumerate(case_studies):
        img_path = os.path.join(r'D:\graduetion thesis\SDDP\test\img', case['image'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load {case['image']}")
            continue
        
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add status overlay
        h, w = image_rgb.shape[:2]
        overlay = image_rgb.copy()
        
        # Add colored border based on status
        border_color = {
            'Success': (0, 255, 0),
            'Misclassified': (255, 165, 0),
            'Failure': (255, 0, 0)
        }[case['status']]
        
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), border_color, 10)
        
        # Blend overlay
        alpha = 0.3
        image_with_border = cv2.addWeighted(overlay, alpha, image_rgb, 1-alpha, 0)
        
        # Add text label
        status_text = f"{case['status']}"
        cv2.putText(image_with_border, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
        
        # Display
        axes[idx].imshow(image_with_border)
        axes[idx].set_title(case['title'], fontsize=11, fontweight='bold', color=case['color'])
        axes[idx].axis('off')
        
        # Add description below
        axes[idx].text(0.5, -0.15, case['description'], 
                      transform=axes[idx].transAxes,
                      ha='center', va='top', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        print(f"\n{case['title']}")
        print(f"Status: {case['status']}")
        print(f"Image: {case['image']}")
        print(f"Description: {case['description']}")
    
    plt.suptitle('Qualitative Analysis: Defect Detection Case Studies\n'
                'RetinaNet Performance on NEU-DET Dataset', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('qualitative_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisual analysis saved as 'qualitative_analysis.png'")

def create_error_analysis():
    """Analyze error patterns"""
    
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    error_types = [
        {
            'type': 'False Negative',
            'description': 'Model fails to detect existing defects',
            'examples': ['pitted_surface_184.jpg', 'rolled-in_scale_226.jpg', 'scratches_300.jpg'],
            'percentage': '50% (3/6 images)',
            'possible_causes': ['Insufficient training', 'Low contrast defects', 'Small defect size']
        },
        {
            'type': 'Misclassification',
            'description': 'Model detects defect but wrong class',
            'examples': ['patches_224.jpg (detected as pitted_surface)'],
            'percentage': '17% (1/6 images)',
            'possible_causes': ['Similar visual features between classes', 'Class imbalance', 'Insufficient class-specific features']
        },
        {
            'type': 'Low Confidence',
            'description': 'Detections with confidence < 0.5',
            'examples': ['crazing_5.jpg (0.172)', 'patches_224.jpg (0.106)'],
            'percentage': '33% (2/3 detected images)',
            'possible_causes': ['Under-trained model', 'Complex defect patterns', 'Ambiguous features']
        }
    ]
    
    print("\nTable: Error Type Analysis")
    print("-"*100)
    print(f"{'Error Type':20} {'Description':30} {'Examples':30} {'Frequency':15}")
    print("-"*100)
    
    for error in error_types:
        print(f"{error['type']:20} {error['description']:30} "
              f"{', '.join(error['examples']):30} {error['percentage']:15}")
    
    print("\nPossible Improvements:")
    for error in error_types:
        print(f"\n{error['type']}:")
        for cause in error['possible_causes']:
            print(f"  • {cause}")

if __name__ == "__main__":
    create_qualitative_analysis()
    create_error_analysis()