import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

from models import *

def main():
    #Hyperparameter
    BATCH_SIZE = 128
    SEED = 1234
    IMG_RES = 28 * 28
    NUM_CLASSES = 10
    

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_data = datasets.MNIST(root='.\data',
                            train=True,
                            download=True)

    mean = train_data.data.float().mean() / 255     # = 0.13066
    std = train_data.data.float().std() / 255       # = 0.30810

    train_transforms = transforms.Compose([
                                transforms.RandomRotation(5, fill=(0,)),    #Trainingsdatensatz um + - 5 Grad zufällig rotieren
                                transforms.RandomCrop(28, padding=2),       #2Pixel Rand erzeugen und davon 28x28 Pixel Crop nehmen
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[mean], std=[std])
                                        ])

    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                        ])    

    train_data = datasets.MNIST(root='.\data',
                                train=True,
                                download=True,
                                transform=train_transforms)

    test_data = datasets.MNIST(root='.\data',
                            train=False,
                            download=True,
                            transform=test_transforms)

    #Validierungsdaten aus den Trainingsdaten ziehen
    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])
    
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms  #Validierungsdaten sollen nicht rotiert werden. Deswegen Zuweisung der Testdatensatz-Transformationen

    #Dataloader der Batches von Daten entählt
    
    train_iterator = data.DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                    batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)


    #Modell instanziieren
    model = MLP(IMG_RES, NUM_CLASSES)
    
    #MLPs haben sehr viel mehr Parameter zu trainieren
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Das MLP hat{count_parameters(model):,} trainierbare Parameter')


    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    def calculate_accuracy(y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train(model, iterator, optimizer, criterion, device):

        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for (x, y) in tqdm(iterator, desc="Training", leave=False):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def evaluate(model, iterator, criterion, device):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():

            for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

                x = x.to(device)
                y = y.to(device)

                y_pred, _ = model(x)

                loss = criterion(y_pred, y)

                acc = calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # Speichern des Modells
    modelSavePath = './trainedModels/mlpMNIST.ckpt'
    os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)

    saveModel = input('Möchten Sie das Modell speicher? (y/n): ')
    if saveModel.lower() == 'y':
        torch.save(model.state_dict(), modelSavePath)
        print(f'Modell wurder unter {modelSavePath} gespeichert.')
    else:
        print('Modell wurde nicht gespeichert.')

    

if __name__ == '__main__':
    main()
