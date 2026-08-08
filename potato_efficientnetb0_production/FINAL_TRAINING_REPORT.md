# Potato Disease Classification
# Final Training & Evaluation Report

## 1. Project Overview

This experiment develops an image classification model for three potato disease classes using EfficientNetB0 transfer learning followed by fine-tuning.

The final model was selected using validation performance and evaluated separately on the held-out test dataset.

## 2. Dataset

Number of classes: 3

Classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

Input image size: 224 x 224 x 3

Total test samples: 216

## 3. Model Architecture

Architecture: EfficientNetB0

Training strategy: Transfer Learning + Fine-Tuning

Input shape: (None, 224, 224, 3)

Output shape: (None, 3)

Total parameters: 4,378,278

Trainable parameters: 1,812,067

Non-trainable parameters: 2,566,211

## 4. Stage-1 Training

Stage-1 epochs: 11

Best Stage-1 validation accuracy: 98.60%

## 5. Fine-Tuning

Fine-tuning epochs: 8

Best fine-tuning epoch: 1

Best fine-tuning validation accuracy: 98.60%

## 6. Final Test Evaluation

Total test samples: 216

Correct predictions: 212

Incorrect predictions: 4

Final test accuracy: 98.15%

Final test error rate: 1.85%

Average prediction confidence: 97.98%

Average confidence for correct predictions: 98.70%

Average confidence for incorrect predictions: 59.95%

## 7. Error Analysis

Highest-error class: Potato___Late_blight

Highest class error rate: 4.00%

Number of misclassified samples: 4

Detailed error-analysis files should be stored alongside the final project artifacts.

## 8. Class Mapping

- 0 -> Potato___Early_blight
- 1 -> Potato___Late_blight
- 2 -> Potato___healthy

## 9. Production Artifacts

- potato_efficientnetb0_final.keras
- class_mapping.json
- preprocessing_config.json
- model_metadata.json
- experiment_metadata.json
- production_manifest.json

## 10. Reproducibility Information

TensorFlow version: 2.21.0

Python version: 3.10.20

Operating system: macOS-26.5-arm64-arm-64bit

Experiment timestamp: 2026-08-08T12:49:21.861000

## 11. Final Conclusion

The experiment completed the image-classification workflow using EfficientNetB0 transfer learning and fine-tuning.

The final test accuracy was 98.15%.

The final test error rate was 1.85%.

The selected model and supporting metadata have been saved as production artifacts.

## 12. Important Limitation

The reported performance represents the observed performance on the available held-out test dataset.

Real-world performance should additionally be validated using an independent external dataset representing the intended deployment environment.