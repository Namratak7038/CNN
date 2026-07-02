
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import shutil
import numpy as np

# --- 1. Data Preparation (similar to initial notebook steps) ---
# Define the base input and output folders
base_input_folder = '/content/drive/MyDrive/project'
base_output_folder = '/content/outputImg'
base_augmentation_output_folder = '/content/augmented_outputImg'

# Define the subfolders to process
subfolders = ['chair_images', 'table_images']

target_size = (250, 250)
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

def prepare_data():
    # Ensure output folders are clean
    if os.path.exists(base_output_folder):
        shutil.rmtree(base_output_folder)
    os.makedirs(base_output_folder, exist_ok=True)
    if os.path.exists(base_augmentation_output_folder):
        shutil.rmtree(base_augmentation_output_folder)
    os.makedirs(base_augmentation_output_folder, exist_ok=True)

    overall_processed_count = 0
    for subfolder_name in subfolders:
        input_subfolder_path = os.path.join(base_input_folder, subfolder_name)
        output_subfolder_path = os.path.join(base_output_folder, subfolder_name)
        os.makedirs(output_subfolder_path, exist_ok=True)

        for filename in os.listdir(input_subfolder_path):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_subfolder_path, filename)
                output_path = os.path.join(output_subfolder_path, filename)
                try:
                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_size)
                        resized_img.save(output_path)
                        overall_processed_count += 1
                except Exception as e:
                    print(f"Error processing '{filename}': {e}")

    print(f"Processed {overall_processed_count} images for resizing.")

    # Image Augmentation
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
    num_augmentations_per_image = 3
    augmented_count = 0

    for subfolder_name in subfolders:
        input_subfolder_path = os.path.join(base_output_folder, subfolder_name)
        output_subfolder_path = os.path.join(base_augmentation_output_folder, subfolder_name)
        os.makedirs(output_subfolder_path, exist_ok=True)

        if not os.path.exists(input_subfolder_path) or not os.listdir(input_subfolder_path):
            print(f"Input subfolder '{input_subfolder_path}' is empty or does not exist. Skipping augmentation for this subfolder.")
            continue

        for filename in os.listdir(input_subfolder_path):
            if filename.lower().endswith(image_extensions):
                original_image_path = os.path.join(input_subfolder_path, filename)
                try:
                    with Image.open(original_image_path) as img:
                        for i in range(num_augmentations_per_image):
                            augmented_img = augmentation_transforms(img)
                            name, ext = os.path.splitext(filename)
                            augmented_filename = f"{name}_aug{i}{ext}"
                            augmented_output_path = os.path.join(output_subfolder_path, augmented_filename)
                            augmented_img.save(augmented_output_path)
                            augmented_count += 1
                except Exception as e:
                    print(f"Error augmenting '{filename}': {e}")
    print(f"Generated {augmented_count} augmented images.")

# --- 2. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
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

def main():
    # Prepare data
    prepare_data()

    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for the dataset
    dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=base_augmentation_output_folder, transform=dataset_transforms)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Total images in dataset: {len(full_dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")
    print(f"Classes: {full_dataset.classes}")

    # Instantiate the model
    model = SimpleCNN(num_classes=len(full_dataset.classes)).to(device)
    print(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # --- Training Loop ---
    print("
Starting Training...")
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    print("Finished Training")

    # --- Evaluation ---
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f}%')

    # Save the trained model
    model_save_path = 'simple_cnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
