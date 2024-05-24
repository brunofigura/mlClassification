import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import sklearn 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import *

def main():

    #Hyperparameter 
    modelSavePath = './trainedModels/LeNet5CIFAR10.ckpt'
    TRAIN_MODEL = False  # True, wenn das Modell trainiert werden soll, False, wenn es geladen werden soll

    NUM_EPOCHS = 5
    INIT_LR = 0.001
    BATCH_SIZE = 32

    # Check if CUDA is available and set device accordingly
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. Daten laden und vorbereiten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # CIFAR Normalisation und Standartabweichung der Kanäle
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    # 3. Netzwerk definieren und auf das Gerät verschieben
    class LeNet5(nn.Module):
        def __init__(self):
                super(LeNet5, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
                x = self.pool(nn.functional.relu(self.conv1(x)))
                x = self.pool(nn.functional.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = nn.functional.relu(self.fc1(x))
                x = nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    print('==> Building model ..')
    model = LeNet5()
    
    if not TRAIN_MODEL:
        print("loading wheigts ...")
        model.load_state_dict(torch.load(modelSavePath))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = model.to(device)
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

        # Speichere das trainierte Modell
        os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)
        torch.save(model.state_dict(), modelSavePath)
        print(f'Modell wurde unter {modelSavePath} gespeichert.')

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

   # Erstelle Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Definiere die Werte der Hyperparameter
    hyperparameters = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'INIT_LR': INIT_LR,
        'BATCH_SIZE': BATCH_SIZE
    }

# Konvertiere die Hyperparameter in einen lesbaren String
    hyperparameters_str = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('LeNet5')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Füge eine Anmerkung mit den Hyperparametern hinzu
    plt.annotate(hyperparameters_str, xy=(0.5, 1.05), xytext=(0.5, 1.1),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=7, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xticks(rotation=45)  # Drehen Sie die Achsenbeschriftungen für bessere Lesbarkeit
    plt.yticks(rotation=45)
    plt.savefig('./confusion_Matrices/cm_CIFAR_LeNet5.png')  # Speichern Sie die Confusion Matrix als PNG-Datei
    plt.show()


if __name__ == '__main__':
    main()