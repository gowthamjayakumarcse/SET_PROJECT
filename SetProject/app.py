import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# =====================================
# DEVICE CONFIGURATION
# =====================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# MODEL PATHS
# =====================================
PRESENCE_MODEL_PATH = "midline_shift_model_convnext_tiny.pth"
DIRECTION_MODEL_PATH = "midline_direction_convnext_small.pth"
SURGERY_MODEL_PATH = "surgery_convnext_small.pth"

# =====================================
# SKULL STRIPPING (as per training)
# =====================================
def skull_strip_pil(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        skull_stripped = cv2.bitwise_and(img, img, mask=mask)
    else:
        skull_stripped = img

    skull_stripped = cv2.cvtColor(skull_stripped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(skull_stripped)

# =====================================
# PREPROCESSING (identical to training)
# =====================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# =====================================
# LOAD MODELS
# =====================================
@st.cache_resource
def load_models():
    # Presence Model
    presence_model = models.convnext_tiny(weights=None)
    presence_model.classifier[2] = nn.Linear(presence_model.classifier[2].in_features, 1)
    presence_model.load_state_dict(torch.load(PRESENCE_MODEL_PATH, map_location=DEVICE))
    presence_model.eval().to(DEVICE)

    # Direction Model
    direction_model = models.convnext_small(weights=None)
    direction_model.classifier[2] = nn.Linear(direction_model.classifier[2].in_features, 2)
    direction_model.load_state_dict(torch.load(DIRECTION_MODEL_PATH, map_location=DEVICE))
    direction_model.eval().to(DEVICE)

    # Surgery Model
    surgery_model = models.convnext_small(weights=None)
    surgery_model.classifier[2] = nn.Linear(surgery_model.classifier[2].in_features, 2)
    surgery_model.load_state_dict(torch.load(SURGERY_MODEL_PATH, map_location=DEVICE))
    surgery_model.eval().to(DEVICE)

    return presence_model, direction_model, surgery_model

presence_model, direction_model, surgery_model = load_models()

# =====================================
# PREDICTION FUNCTIONS
# =====================================
def predict_presence(image):
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        logits = presence_model(tensor)
        prob = torch.sigmoid(logits).item()
    label = 1 if prob > 0.5 else 0
    return ("Shift" if label == 1 else "No Shift"), prob

def predict_direction(image):
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        output = direction_model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
    direction = "Left" if pred == 0 else "Right"
    return direction, probs[pred].item()

def predict_surgery(image):
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        output = surgery_model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
    surgery = "Yes" if pred == 1 else "No"
    return surgery, probs[pred].item()

# =====================================
# STREAMLIT UI
# =====================================
st.set_page_config(page_title="🧠 Brain MRI Shift & Surgery Predictor", layout="wide")

st.title("🧠 Brain MRI – Midline Shift & Surgery Prediction")
st.markdown("""
Upload a **CT/MRI brain scan** to analyze:
1. Detect if **Midline Shift** exists  
2. If yes, determine the **Shift Direction**  
3. Recommend whether **Surgery** is advisable  
""")

uploaded_image = st.file_uploader("📤 Upload MRI/CT Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    original_img = Image.open(uploaded_image).convert("RGB")

    # Skull Stripping
    processed_img = skull_strip_pil(original_img)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(processed_img, caption="🧩 Processed (Skull-Stripped)", width=300)

    with col2:
        st.write("🔍 **Running Inference...**")

        presence, conf_presence = predict_presence(processed_img)

        if presence == "Shift":
            st.success(f"✅ **Midline Shift Detected** ({conf_presence*100:.2f}%)")

            direction, conf_dir = predict_direction(processed_img)
            st.info(f"🧭 **Direction:** {direction} ({conf_dir*100:.2f}%)")

            surgery, conf_surg = predict_surgery(processed_img)
            if surgery == "Yes":
                st.error(f"🚨 **Surgery Advised** ({conf_surg*100:.2f}%)")
            else:
                st.success(f"👍 **Surgery Not Required** ({conf_surg*100:.2f}%)")

        else:
            st.success(f"✅ **No Midline Shift** ({conf_presence*100:.2f}%)")

    with st.expander("🧩 View Detailed Probabilities"):
        st.write(f"**Presence Confidence:** {conf_presence:.4f}")
        if presence == "Shift":
            st.write(f"**Direction Confidence:** {conf_dir:.4f}")
            st.write(f"**Surgery Confidence:** {conf_surg:.4f}")

st.caption("Models: ConvNeXt-Tiny (Presence) | ConvNeXt-Small (Direction, Surgery)")
