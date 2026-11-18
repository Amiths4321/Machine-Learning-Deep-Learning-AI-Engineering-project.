
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Steel Defect Classifier",
    page_icon="🔍",
    layout="centered"
)

# Class names
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Model architecture
class SteelDefectCNN(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 6)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load model
@st.cache_resource
def load_model():
    model = SteelDefectCNN()
    model.load_state_dict(
        torch.load(
            "notebook/models/final_steel_defect_cnn.pth",
            map_location="cpu"
        )
    )
    model.eval()
    return model


model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# App interface
st.title("Steel Surface Defect Classifier")
st.write("Upload a steel surface image to predict its defect class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Defect"):
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted_class = probabilities.max(dim=1)

        label = CLASS_NAMES[predicted_class.item()]
        score = confidence.item() * 100

        st.success(f"Predicted Defect: {label}")
        st.metric("Model Confidence", f"{score:.2f}%")
