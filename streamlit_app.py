import streamlit as st
import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from PIL import Image
import os
@st.cache_resource
def load_model():

    model = SimpleCNN(num_classes=2)
    model.eval()

    # Fuse Conv + ReLU
    fused_model = torch.quantization.fuse_modules(
        model,
        [['features.0', 'features.1'],
         ['features.3', 'features.4']],
        inplace=False
    )

    # Same qconfig used during quantization
    fused_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # Prepare
    prepared_model = torch.quantization.prepare(
        fused_model,
        inplace=False
    )

    # Convert
    quantized_model = torch.quantization.convert(
        prepared_model,
        inplace=False
    )

    # Load quantized weights
    state_dict = torch.load(
        "quantized_simple_cnn_model.pth",
        map_location="cpu"
    )

    quantized_model.load_state_dict(state_dict)
    quantized_model.eval()

    return quantized_model
