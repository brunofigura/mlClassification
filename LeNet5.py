# LeNet5 Architektur kann Input-Dimensionen von 32x32x1 annehmen. Damit ist es auf Grausstufenbilder ausgelegt mit nur einem "Helligkeits"-Kanal

# Libraries importieren
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def main():

    # Hyperparamater
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    #Loss-Funktion
    cost = nn.CrossEntropyLoss()

    


    #Gerät für die Modell-Berechnung auswählen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Datensatz Transformation
    transformTrain = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))])
    transformTest = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean = (0.1325,), std = (0.3105,))])
    #Datensatz in die Loader laden
    train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transformTrain, download = True)
    
    test_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transformTest, download = True)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size, shuffle = True)

    print(train_dataset)
    
    print(test_dataset)
    
    ###################
    #NETZ-MODELLIERUNG#
    ###################

    class LeNet5(nn.Module):
        def __init__(self, num_classes):
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.fc = nn.Linear(400, 120)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
    ##########
    #TRAINING#
    ##########

    

    #gibt an wieviele Schritte noch übrig bleiben beim Training
    total_step = len(train_loader)

    model = LeNet5(num_classes).to(device)
    #Optimierer festlegen
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): #über die Batches iterieren 
            images = images.to(device)
            labels = labels.to(device)
            
            #Forward pass
            #Vorhersage über die Klasse des Bildes durch das Modell wird berechnet
            outputs = model(images)                 
            #Über Kreuz-Entropie wird der loss zwischen Vorhersage und Groundtruth berechnet
            loss = cost(outputs, labels)
                
            # Backward and optimize
            # Gradienten zurücksetzen nach jedem Batch
            optimizer.zero_grad()
            # Gradienten mithilfe berechnetem loss neu berechnen
            loss.backward()
            #Gewichte des Netztes updaten mithilfe des neuen Gradienten
            optimizer.step()
                    
            if (i+1) % 400 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
    #########
    #TESTING#
    #########
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
	 

if __name__ == '__main__':
    main()