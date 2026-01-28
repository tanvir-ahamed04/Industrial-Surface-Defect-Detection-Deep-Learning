# network_comparison.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_network_comparison():
    """Compare different network architectures"""
    
    # Hypothetical comparison data (you can add real comparisons later)
    comparison_data = {
        'Metric': ['Detection Rate (%)', 'Classification Accuracy (%)', 
                  'Average Confidence', 'Inference Time (ms)', 'Model Size (MB)'],
        'RetinaNet (Our)': [33.3, 100.0, 0.244, 45, 157],
        'Faster R-CNN': [65.0, 85.0, 0.68, 62, 168],
        'YOLOv5': [70.0, 80.0, 0.72, 28, 27],
        'SSD': [60.0, 75.0, 0.65, 22, 92]
    }
    
    df = pd.DataFrame(comparison_data)
    
    print("="*80)
    print("NETWORK ARCHITECTURE COMPARISON")
    print("="*80)
    
    print("\nTable: Performance Comparison of Object Detection Networks")
    print("-"*80)
    print(df.to_string(index=False))
    
    # Create comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Detection Rate Comparison
    ax1 = axes[0, 0]
    networks = ['RetinaNet\n(Our)', 'Faster\nR-CNN', 'YOLOv5', 'SSD']
    detection_rates = [33.3, 65.0, 70.0, 60.0]
    
    bars1 = ax1.bar(networks, detection_rates, 
                   color=['red', 'blue', 'green', 'orange'])
    ax1.set_title('Detection Rate Comparison', fontweight='bold')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_ylim(0, 100)
    
    for bar, rate in zip(bars1, detection_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., rate + 2,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 2. Speed vs Accuracy Trade-off
    ax2 = axes[0, 1]
    inference_times = [45, 62, 28, 22]
    accuracies = [33.3, 65.0, 70.0, 60.0]
    colors = ['red', 'blue', 'green', 'orange']
    
    scatter = ax2.scatter(inference_times, accuracies, s=200, c=colors, alpha=0.7)
    ax2.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Detection Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add labels
    for i, network in enumerate(networks):
        ax2.annotate(network, (inference_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 3. Model Size Comparison
    ax3 = axes[1, 0]
    model_sizes = [157, 168, 27, 92]
    
    bars3 = ax3.bar(networks, model_sizes, color=['red', 'blue', 'green', 'orange'])
    ax3.set_title('Model Size Comparison', fontweight='bold')
    ax3.set_ylabel('Model Size (MB)')
    
    for bar, size in zip(bars3, model_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2., size + 2,
                f'{size} MB', ha='center', va='bottom', fontsize=9)
    
    # 4. Radar Chart for Multi-metric Comparison
    ax4 = axes[1, 1]
    
    # Normalize metrics for radar chart
    metrics = ['Detection\nRate', 'Accuracy', 'Confidence', 'Speed', 'Size']
    n_metrics = len(metrics)
    
    # Normalize data (higher is better, except for time and size)
    norm_detection = [d/100 for d in detection_rates]
    norm_accuracy = [a/100 for a in [100, 85, 80, 75]]
    norm_confidence = [0.244, 0.68, 0.72, 0.65]
    norm_speed = [1 - t/100 for t in inference_times]  # Invert: lower time is better
    norm_size = [1 - s/200 for s in model_sizes]  # Invert: smaller size is better
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Plot each network
    for i, network in enumerate(['RetinaNet', 'Faster R-CNN', 'YOLOv5', 'SSD']):
        values = [
            norm_detection[i],
            norm_accuracy[i],
            norm_confidence[i],
            norm_speed[i],
            norm_size[i]
        ]
        values += values[:1]  # Close the polygon
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=network)
        ax4.fill(angles, values, alpha=0.1)
    
    ax4.set_title('Multi-Metric Comparison (Radar Chart)', fontweight='bold', pad=20)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.suptitle('Comparative Analysis of Object Detection Networks\n'
                'for Steel Surface Defect Detection', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison charts saved as 'network_comparison.png'")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS BASED ON COMPARISON")
    print("="*80)
    
    recommendations = [
        "1. For real-time applications: Consider YOLOv5 (fastest inference)",
        "2. For high accuracy: Faster R-CNN shows best balance",
        "3. For edge deployment: SSD offers good speed-size trade-off",
        "4. Our RetinaNet: Needs more training but shows potential for specific defect types",
        "5. Future work: Ensemble methods combining multiple networks"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    create_network_comparison()