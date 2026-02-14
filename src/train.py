import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from loader import load_cifar
from cifar_dataset import CifarDataset, transform_train, transform_val


def create_model(num_classes=3, model_name='resnet18'):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_model(num_epochs=10, batch_size=32, model_name='resnet18'):
    train_dataset = CifarDataset(train=True, transform=transform_train, bg_samples_per_class=200)
    val_dataset = CifarDataset(train=False, transform=transform_val, bg_samples_per_class=100)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    device = torch.device("cpu")
    model = create_model(num_classes=3, model_name=model_name).to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_dataset.labels),
        y=train_dataset.labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {val_loss/len(val_loader):.4f}, '
              f'Test Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    return model