import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Config
st.set_page_config(page_title="Visual Defect Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 Visual Defect Detection with Grad-CAM</h1>", unsafe_allow_html=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# Image preprocessor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

# Grad-CAM
def generate_gradcam(model, input_tensor, target_class):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam

# Upload section
uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(image).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)[0][pred].item()
        class_label = pred.item()

    classes = ['good', 'defective']
    label = classes[class_label].upper()

    # Grad-CAM
    grayscale_cam = generate_gradcam(model, input_tensor, class_label)
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Centered layout using empty columns for spacing
    col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])
    with col_main:
        st.markdown("### Prediction Results", unsafe_allow_html=True)
        st.markdown(f"#### 🧠 Predicted: **{label}**", unsafe_allow_html=True)
        st.markdown(f"Confidence: `{prob:.2%}`")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Uploaded Image")
            st.image(image.resize((224, 224)), width=300)

        with col_right:
            st.markdown("##### Grad-CAM Overlay")
            st.image(cam_overlay, width=300)

        # Save option
        if st.checkbox("💾 Save Grad-CAM image"):
            Image.fromarray(cam_overlay).save("gradcam_result.png")
            st.success("Saved as `gradcam_result.png` ✅")
