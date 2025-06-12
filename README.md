# **Visual Defect Detection in Manufacturing Using Transfer Learning**


[![Visual Defect Detection Demo](https://img.shields.io/badge/View-Demo-blue)](https://drive.google.com/file/d/1R75UEFXbb-Cu14rA76tv5t3pCYByBxPp/view)

## Project Overview 
This project aims to build an automated visual inspection system that classifies manufactured products as defective or non-defective using deep learning and transfer learning techniques. In industrial settings, even minor visual defects can impact product quality, and human inspection is often slow, inconsistent, or costly. Transfer learning enables efficient training on limited defect datasets by leveraging pre-trained CNNs, making it ideal for industrial use cases.

## Implementation

Steps:
- Select a Pre-trained Model:
We use a robust CNN model ResNet-50 pre-trained on ImageNet, which has generalized visual feature extraction capabilities.

- Prepare the Dataset:
Use the MVTec Anomaly Detection (MVTec AD) dataset, which includes high-resolution images of various industrial products like metal parts, circuit boards, and textiles, annotated for normal and defective cases.

- Fine-tune the Model:
Freeze the initial layers of the model and retrain the final layers on the defect classification task. Experiment with selectively unfreezing deeper layers for improved performance. Implement data augmentation techniques.

- Evaluation:
Evaluate performance using accuracy, precision, recall and F1 score. Compare against a baseline model trained from scratch or a simpler machine learning method. Compare against baseline models trained from scratch. Visualize performance with confusion matrices and sample predictions

## Dataset:
MVTec anomaly detection dataset (MVTec AD). [1][2]\
It can be downloaded from the following link: https://www.mvtec.com/company/research/datasets/mvtec-ad

## Notebooks
1. **Baseline Model**
- Implements defect detection using a simple lightweight 2-layer CNN.

2. **Single Layer Frozen Model**
- Trains two models (single and multi-item category)
- Freezes all but the last layer for binary classification

3. **Enhanced 2-Layer Frozen Model**
- Trains two models (single and multi-item category)
- Freezes all layers except the last block (layer4) and classifier (fc)
- Implements more sophisticated data augmentation
- Uses weighted loss for class imbalance
- Includes early stopping and model checkpointing

## Web App Deployment:
This app uses a fine-tuned ResNet-50 model and Grad-CAM to visualize defects in manufacturing images. Follow the steps below to get started:

1. Ensure the following files are in the same directory:
- app.py: Streamlit app script.
- model.pth: Fine tuned ResNet-50 model with final chosen configuration
2.	Run the Streamlit App
```bash
streamlit run app.py
```
3.	Using the App
- Upload a test image (JPEG/PNG). 
- The app will display the predicted class and a Grad-CAM heatmap highlighting defect areas.

### Team Members:
[Ankita Maria John](https://github.com/ankita-m-john)\
[Jayalakshmi Sasidharan](https://github.com/jayalakshmi-sasidharan)\
[Pragnya Ambekar](https://github.com/pragnyaambekar)

### Reference

[1] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

[2] Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.

[Back to Top](#table-of-contents)