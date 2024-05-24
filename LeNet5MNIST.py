import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import sklearn 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize

from models import *

def main():

    #Hyperparameter
    modelSavePath = './trainedModels/LeNet5MNIST.ckpt'

    NUM_EPOCHS = 5
    INIT_LR = 0.001
    BATCH_SIZE = 32
    NUM_CLASSES = 10

    # Check if CUDA is available and set device accordingly
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. Daten laden und vorbereiten -> bei MNIST resizen wir 28x28 auf 32x32 damit die Input-Schicht die erwarteten Dimensionen bekommt
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Hochskalieren auf 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalisieren und Standartabweichung des MNIST Datensatzen ( vorher waren die Werte 0.5 und 0.5)
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    # 3. Netzwerkinstanziierung und auf das Gerät verschieben
    print('==> Building model ..')
    model = LeNet5MNIST()
    model = model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True          #optimiert für statische Modell-Architekturen

    # 4. Loss-Funktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), INIT_LR)

    # 5. Training des Netzwerks
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0

    print('Training beendet')

    # 6. Evaluation des Netzwerks
    model.eval()
    test_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []



    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Cross-Entropy Loss: {test_loss:.4f}')




    # Speichern des Modells
    
    os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)

    saveModel = input('Möchten Sie das Modell speicher? (y/n): ')
    if saveModel.lower() == 'y':
        torch.save(model, modelSavePath)
        print(f'Modell wurder unter {modelSavePath} gespeichert.')
    else:
        print('Modell wurde nicht gespeichert.')

    

if __name__ == '__main__':
    main()