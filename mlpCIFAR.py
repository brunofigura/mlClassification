import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
 

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

from models import *

def main():
    #Hyperparameter
    BATCH_SIZE = 64
    HIDDEN_SIZE = 250
    IMG_RES = 32 * 32 * 3
    NUM_CLASSES = 10
    LR = 0.0001
    NUM_EPOCHS = 15
   
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    

    train_dataset = datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform)

    test_dataset = datasets.CIFAR10(root='./data',
                            train=False,
                            download=True,
                            transform=transform)


    #Dataloader der Batches von Daten entählt
    
    train_iterator = DataLoader(train_dataset,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)


    test_iterator = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE)


    #Modell instanziieren
    model = MLP(IMG_RES, HIDDEN_SIZE, NUM_CLASSES).to(device)
    

    optimizer = optim.Adam(model.parameters(), LR)
    criterion = nn.CrossEntropyLoss()   

    # Training the model
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_iterator):
            images, labels = images.to(device), labels.to(device)  # Move to devicey
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_iterator)}], Loss: {loss.item():.4f}')

    # 6. Evaluation des Netzwerks
    model.eval()
    test_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []



    with torch.no_grad():
        for data in test_iterator:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_iterator.dataset)
    accuracy = correct / len(test_iterator.dataset)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Cross-Entropy Loss: {test_loss:.4f}')

    # Speichern des Modells
    modelSavePath = './trainedModels/mlpCIFAR.ckpt'
    os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)

    saveModel = input('Möchten Sie das Modell speicher? (y/n): ')
    if saveModel.lower() == 'y':
        torch.save(model.state_dict(), modelSavePath)
        print(f'Modell wurder unter {modelSavePath} gespeichert.')
    else:
        print('Modell wurde nicht gespeichert.')

    

if __name__ == '__main__':
    main()
