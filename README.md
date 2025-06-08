# **Visual Defect Detection in Manufacturing Using Transfer Learning**

## **Overview**:
This project aims to build an automated visual inspection system that classifies manufactured products as defective or non-defective using deep learning and transfer learning techniques. In industrial settings, even minor visual defects can impact product quality, and human inspection is often slow, inconsistent, or costly. Transfer learning enables efficient training on limited defect datasets by leveraging pre-trained CNNs, making it ideal for industrial use cases.

## **Implementation**:
Steps:
- Select a Pre-trained Model:\
Use a robust CNN model such as ResNet-50, EfficientNet, or DenseNet pre-trained on ImageNet, which has generalized visual feature extraction capabilities.

- Prepare the Dataset:\
Use the MVTec Anomaly Detection (MVTec AD) dataset, which includes high-resolution images of various industrial products like metal parts, circuit boards, and textiles, annotated for normal and defective cases.

- Fine-tune the Model:\
Freeze the initial layers of the model and retrain the final layers on the defect classification task. Experiment with selectively unfreezing deeper layers for improved performance.

- Evaluation:\
Evaluate performance using accuracy, precision, recall, F1 score, and ROC-AUC. Compare against a baseline model trained from scratch or a simpler machine learning method (e.g., SVM with handcrafted features).

## **Dataset used**:
MVTec anomaly detection dataset (MVTec AD) [1-2]\
It can be downloaded from the following link: https://www.mvtec.com/company/research/datasets/mvtec-ad

[1] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

[2] Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.

### Team Members:
[Ankita Maria John](https://github.com/ankita-m-john)\
[Pragnya Ambekar](https://github.com/pragnyaambekar)\
[Jayalakshmi Sasidharan](https://github.com/jayalakshmi-sasidharan)
