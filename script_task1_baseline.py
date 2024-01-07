import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

class AAITDataset(Dataset):
    def __init__(self, dataset, img_dir, is_train=True, transform=None, target_transform=None):
        self.is_train = is_train
        if self.is_train:
            self.img_labels = dataset
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.is_train:
            return len(self.img_labels)
        else:
            self.img_files = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))]
            return len(self.img_files)

    def __getitem__(self, idx):
        if self.is_train:
            img_path = self.img_labels.iloc[idx, 0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            #print(image)
            if image.size()[0] == 1:
                image = image.expand(3, -1, -1)
            label = self.img_labels.iloc[idx, 1]
            
            if self.target_transform:
                label = self.target_transform(label)
            return image, label, img_path
        else: 
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            if image.size(0) == 1:
                image = image.expand(3, -1, -1)
            
            return image, img_path

class CustomModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomModel, self).__init__()

        self.resnet50 = models.resnet50(weights="IMAGENET1K_V2")

        num_features = self.resnet50.fc.in_features
        self.dropout = nn.Dropout(p=0.1)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1024, num_classes)
        )
        #self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the network
        return self.resnet50(x)
    
def train_model(model, train_dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels, _ in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_dataloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, precision, recall, f1

def generate_pseudo_labels(model, dataloader):
    model.eval()
    pseudo_label_data = []
    with torch.no_grad():
        for images, img_paths in dataloader:
            images = images.to(mps_device)
            outputs = model(images)
            probs, predicted = torch.max(outputs, 1)
            
            
            for img_path, label in zip(img_paths, predicted):
                pseudo_label_data.append((img_path, label.item()))

    pseudo_labels_df = pd.DataFrame(pseudo_label_data, columns=['sample', 'label'])
    return pseudo_labels_df



def main(args):
    # Setup based on command line arguments
    parser = argparse.ArgumentParser(description="Train a model on a specified dataset with a given optimizer.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train_img_labeled_dir', type=str, required=True, help='Directory of labeled training images')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Directory of validation images')
    parser.add_argument('--model', type=str, choices=['resnet50', 'efficientnet_b0'], required=True, help='Model to use')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], required=True, help='Optimizer to use')

    args = parser.parse_args()
    train_annotation_file = args.dataset
    train_img_labeled_dir = args.train_img_labeled_dir
    val_img_dir = args.val_img_dir
    model_name = args.model
    optimizer_choice = args.optimizer
    num_classes=100

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(mps_device)
    else:
        print("MPS not found!")

    # Load data, setup transforms, dataset, dataloader, etc.
    df = pd.read_csv(train_annotation_file)

    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # le decomentam ptr train
        transforms.RandomRotation(degrees=15), # le decomentam ptr train
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),

    ])

    train_dataset = AAITDataset(train_data, train_img_labeled_dir, is_train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = AAITDataset(test_data, train_img_labeled_dir, is_train=True, transform=transform)
    test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Instantiate model based on command line argument
    if model_name == 'resnet50':
        model = CustomModel(num_classes=100, pretrained=True).to(mps_device)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(mps_device)
        

    # Choose optimizer based on command line argument
    if optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    # Train, evaluate, and generate pseudo labels
    train_model(model, train_dataloader, criterion, optimizer, epochs=30, device=mps_device)
    avg_loss, acc, prec, rec, f1 = evaluate_model(model, test_dataloader, criterion, mps_device)

    val_unlabeled_dataset = AAITDataset(None, val_img_dir, is_train=False, transform=transform)
    val_unlabeled_dataloader = DataLoader(val_unlabeled_dataset, batch_size=64, shuffle=False)

    val_labels = generate_pseudo_labels(model, val_unlabeled_dataloader)
    val_labels['sample'] = val_labels['sample'].apply(lambda x: x.split('/')[-1])

    val_labels['sort_key'] = val_labels['sample'].str.extract('(\d+)').astype(int)
    val_labels.sort_values(by='sort_key', ascending=True, inplace=True)
    val_labels.reset_index(inplace=True)
    val_labels.drop(columns=['sort_key','index'], inplace=True)

    val_labels.to_csv('task1/baseline.csv', index=False)


if __name__ == '__main__':
    main()

