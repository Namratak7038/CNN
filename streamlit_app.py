import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# ----------------------------
# Model Definition
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
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


MODEL_PATH = "quantized_simple_cnn_model.pth"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()

    model = SimpleCNN(num_classes=4)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        # If it is a state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            return model

        # If it is an entire saved model
        else:
            checkpoint.eval()
            return checkpoint

    except Exception as e:
        st.error("Model loading failed!")
        st.code(str(e))
        st.stop()


model = load_model()

st.title("Furniture Image Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

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
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = [
        "Chair",
        "Chair Images",
        "Table",
        "Table Images"
    ]

    st.success(f"Prediction: {classes[predicted.item()]}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")
