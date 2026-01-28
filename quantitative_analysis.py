# quantitative_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_quantitative_analysis():
    """Generate quantitative analysis for thesis"""
    
    # Your actual results (update with your real data)
    results = {
        'Defect Type': ['Crazing', 'Inclusion', 'Patches', 
                       'Pitted Surface', 'Rolled-in Scale', 'Scratches'],
        'Test Images': [1, 1, 1, 1, 2, 1],  # Number of test images per class
        'Detected': [1, 0, 0, 0, 1, 0],      # Images where defect was detected
        'Correctly Classified': [1, 0, 0, 0, 1, 0],  # Correct classification when detected
        'Average Confidence': [0.172, 0.000, 0.106, 0.000, 0.479, 0.000],  # Your actual scores
        'Detection Rate (%)': [100, 0, 0, 0, 50, 0],  # Detection rate per class
    }
    
    df = pd.DataFrame(results)
    
    # Calculate overall metrics
    total_images = df['Test Images'].sum()
    total_detected = df['Detected'].sum()
    total_correct = df['Correctly Classified'].sum()
    
    overall_detection_rate = (total_detected / total_images) * 100
    classification_accuracy = (total_correct / max(total_detected, 1)) * 100
    avg_confidence = df['Average Confidence'][df['Average Confidence'] > 0].mean()
    
    print("="*80)
    print("QUANTITATIVE ANALYSIS - RetinaNet for Steel Defect Detection")
    print("="*80)
    
    print("\nTable 1: Performance by Defect Type")
    print(df.to_string(index=False))
    
    print(f"\nTable 2: Overall Performance Metrics")
    print("-"*60)
    print(f"{'Metric':30} {'Value':20}")
    print("-"*60)
    print(f"{'Total Test Images':30} {total_images:20}")
    print(f"{'Images with Detections':30} {total_detected:20}")
    print(f"{'Overall Detection Rate':30} {overall_detection_rate:19.1f}%")
    print(f"{'Classification Accuracy':30} {classification_accuracy:19.1f}%")
    print(f"{'Average Confidence':30} {avg_confidence:20.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Detection Rate by Class
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Defect Type'], df['Detection Rate (%)'], 
                   color=['red', 'blue', 'green', 'orange', 'purple', 'cyan'])
    ax1.set_title('Detection Rate by Defect Type', fontweight='bold')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}%', ha='center', va='bottom')
    
    # 2. Confidence Scores
    ax2 = axes[0, 1]
    confidence_data = df[df['Average Confidence'] > 0]
    bars2 = ax2.bar(confidence_data['Defect Type'], confidence_data['Average Confidence'],
                   color=['red', 'green', 'purple'])
    ax2.set_title('Average Confidence Scores', fontweight='bold')
    ax2.set_ylabel('Confidence Score')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Confusion Matrix (Simplified)
    ax3 = axes[1, 0]
    confusion_data = np.array([
        [1, 0, 0, 0, 0, 0],  # Crazing
        [0, 0, 0, 0, 0, 0],  # Inclusion
        [0, 0, 0, 1, 0, 0],  # Patches -> Pitted
        [0, 0, 0, 0, 0, 0],  # Pitted
        [1, 0, 0, 0, 1, 0],  # Rolled (2 images: 1 correct, 1 missed)
        [0, 0, 0, 0, 0, 0],  # Scratches
    ])
    
    im = ax3.imshow(confusion_data, cmap='Blues', interpolation='nearest')
    ax3.set_title('Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted Class')
    ax3.set_ylabel('True Class')
    ax3.set_xticks(range(6))
    ax3.set_yticks(range(6))
    ax3.set_xticklabels(['Cr', 'In', 'Pa', 'Pi', 'Ro', 'Sc'], fontsize=9)
    ax3.set_yticklabels(['Cr', 'In', 'Pa', 'Pi', 'Ro', 'Sc'], fontsize=9)
    
    # Add text in cells
    for i in range(6):
        for j in range(6):
            if confusion_data[i, j] > 0:
                ax3.text(j, i, str(int(confusion_data[i, j])),
                        ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax3)
    
    # 4. Performance Summary
    ax4 = axes[1, 1]
    metrics = ['Detection\nRate', 'Classification\nAccuracy', 'Avg\nConfidence']
    values = [overall_detection_rate, classification_accuracy, avg_confidence * 100]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars4 = ax4.bar(metrics, values, color=colors)
    ax4.set_title('Overall Performance Metrics', fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_ylim(0, 100)
    
    for bar, value in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.suptitle('Quantitative Analysis: RetinaNet on NEU-DET Dataset\n'
                '(30 Epochs Training)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('quantitative_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save data to CSV
    df.to_csv('quantitative_results.csv', index=False)
    
    print(f"\nVisualizations saved as 'quantitative_analysis.png'")
    print(f"Data saved as 'quantitative_results.csv'")
    
    return df

if __name__ == "__main__":
    generate_quantitative_analysis()