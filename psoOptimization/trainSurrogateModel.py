import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 256
learning_rate = 0.001
target_label = 0  # The target label for training data

# Read indices from cifar100_0.05.txt
with open('cifar100_0.05.txt', 'r') as f:
    selected_indices = [int(line.strip()) for line in f.readlines()]

# CIFAR-100 dataset with data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Custom dataset class using only selected indices
class CIFAR100Subset(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, download=False, indices=None):
        super().__init__(root, train=train, transform=transform, download=download)
        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices].tolist()

# Training dataset with only selected indices
train_dataset = CIFAR100Subset(root='./data', train=True, download=True, transform=transform_train, indices=selected_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Testing dataset remains unchanged
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Use DenseNet161 model
model = models.densenet161(pretrained=True)
model.classifier = nn.Linear(2208, 100)  # CIFAR-100 has 100 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to blend two images
def merge_images(cifar_img, folder_img, ratio=0.7):
    return ratio * cifar_img + (1 - ratio) * folder_img

# Function to load images from a folder
def load_folder_images(folder_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    folder_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            folder_images.append(img)
    return folder_images

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs, target_label):
    model.train()
    folder_path = './panda600'
    folder_images = load_folder_images(folder_path)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Replace all labels with the target label
            # labels = torch.full_like(labels, target_label)
            images = merge_images(images, random.choice(folder_images))
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Function to test the model
def test_model(model, test_loader):
    model.eval()  # Switch to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# Save the trained model
def save_model(model, path='surrogate_densenet161_cifar100.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Train and test the model
train_model(model, train_loader, criterion, optimizer, num_epochs, target_label)
# test_model(model, test_loader)

# Save the trained model
save_model(model)
