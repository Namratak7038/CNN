import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# -----------------------------
# Define CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "quantized_simple_cnn_model.pth"

@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=4)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Furniture Image Classification")
st.write("Upload an image to classify it.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found!")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    classes = [
        "Chair",
        "Chair Images",
        "Table",
        "Table Images"
    ]

    st.success(f"Prediction: **{classes[predicted.item()]}**")

    st.info(f"Confidence: **{confidence.item() * 100:.2f}%**")
