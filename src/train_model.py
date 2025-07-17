import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PIL import Image
from glob import glob

import os
import shutil

class IntelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

class PredictionDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), path

class IntelClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(IntelClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = '../data/seg_train/'
test_folder = '../data/seg_test/'

train_dataset = IntelDataset(train_folder, transform=transform)
test_dataset = IntelDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def train():
    num_epochs = 6
    train_losses = []

    model = IntelClassifier(num_classes=6)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}")

    evaluate_model(model, test_loader, device)
    predict(model, test_dataset.classes)

def predict(model, class_names):
    pred_images = glob('../data/seg_pred/*')
    batch_size = 64

    dataset = PredictionDataset(pred_images, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    path = '../results/predicted_img'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for class_name in class_names:
        os.makedirs(os.path.join(path, class_name), exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = probs.argmax(dim=1).cpu().numpy()

            for img_path, class_idx in zip(paths, predicted_classes):
                class_name = class_names[class_idx]
                shutil.copy(img_path, os.path.join(path, class_name))

if __name__ == '__main__':
    train()
