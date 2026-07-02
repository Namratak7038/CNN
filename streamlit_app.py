
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# 1. Define the model class (Must match the architecture used in training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4): # num_classes should match the training
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
        x = self.features(x);
        x = self.classifier(x);
        return x

# 2. Setup the interface
st.title("Image Classification Predictor")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocessing
    test_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_batch = test_transforms(image).unsqueeze(0)
    
    # 4. Load Model and Predict
    # Ensure the model path is correct
    model_path = 'simple_cnn_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model was trained and saved correctly.")
    else:
        model = SimpleCNN(num_classes=4) # num_classes must match the trained model
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Define class labels - these must match the order during training
        # Based on previous execution: Classes: ['chair', 'chair_images', 'table', 'table_images']
        classes = ['chair', 'chair_images', 'table', 'table_images']

        with torch.no_grad():
            output = model(input_batch)
            _, predicted_idx = torch.max(output, 1)
            prediction = classes[predicted_idx.item()]
            
        st.write(f"### Prediction: {prediction}")
