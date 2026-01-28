# Industrial Surface Defect Detection (ISDD)

## Project Overview
The **Industrial Surface Defect Detection (ISDD)** project is an AI-powered solution designed to identify and classify surface defects in industrial materials. This project leverages deep learning techniques to achieve high accuracy and efficiency in defect detection, making it suitable for real-world industrial applications.

## Features
- **Deep Learning Models**: Utilizes state-of-the-art deep learning architectures such as RetinaNet for defect detection.
- **Customizable Thresholds**: Allows testing with upper and lower thresholds for fine-tuning detection sensitivity.
- **Data Augmentation**: Includes preprocessing and augmentation techniques to enhance model robustness.
- **Model Comparison**: Tools to compare the performance of different models and configurations.
- **Visualization**: Generates visualizations of predictions and results for qualitative analysis.
- **Performance Benchmarking**: Evaluates model efficiency and accuracy on test datasets.
- **Pipeline Automation**: End-to-end pipeline for training, testing, and evaluation.

## Dataset
The project uses the **NEU-DET** dataset, which contains:
- **Annotations**: XML files describing defect locations and types.
- **Images**: High-resolution images of industrial surfaces with various defect types, such as:
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches

## Efficiency
- **Accuracy**: Achieves high accuracy in detecting and classifying defects.
- **Scalability**: Can handle large datasets and adapt to new defect types with retraining.
- **Real-World Testing**: Validated on real-world industrial scenarios to ensure practical applicability.

## Project Structure
- **augmentations.py**: Data augmentation techniques.
- **compare_models.py**: Compare the performance of different models.
- **config.py**: Configuration settings for the project.
- **dataset.py**: Dataset loading and preprocessing.
- **evaluate.py**: Evaluation metrics and methods.
- **generate_test_scenarios.py**: Generate test scenarios for evaluation.
- **metrics.py**: Custom metrics for model evaluation.
- **model.py**: Model architecture definitions.
- **retinanet_train.py**: Training script for RetinaNet.
- **test_retinanet.py**: Testing script for RetinaNet.
- **utils.py**: Utility functions.
- **visualize.py**: Visualization tools.

## How to Use
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Dataset**:
   - Place the NEU-DET dataset in the `data/NEU-DET/` directory.
3. **Train the Model**:
   ```bash
   python retinanet_train.py
   ```
4. **Evaluate the Model**:
   ```bash
   python evaluate.py
   ```
5. **Test on New Images**:
   Place images in the `test/img/` directory and run:
   ```bash
   python test_retinanet.py
   ```

## Results
- **Quantitative Analysis**: Results are stored in `quantitative_results.csv`.
- **Qualitative Analysis**: Visual results are generated for manual inspection.

## Future Work
- Extend the dataset to include more defect types.
- Optimize model architectures for faster inference.
- Integrate the solution into industrial production lines.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- The NEU-DET dataset creators for providing the dataset.
- Open-source contributors for libraries and tools used in this project.