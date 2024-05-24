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

def main():
    #Hyperparameter
    modelSavePath = './trainedModels/mlpCIFAR.ckpt'

    TRAIN_MODEL = True

    BATCH_SIZE = 64
    HIDDEN_SIZE = 250
    IMG_RES = 32 * 32 * 3
    NUM_CLASSES = 10
    INIT_LR = 0.001
    NUM_EPOCHS = 5
   
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    

    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                            train=False,
                            download=True,
                            transform=transform)


    #Dataloader der Batches von Daten entählt
    
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)


    testloader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE)


    #Modell instanziieren
    # MLP mit zwei hidden Layer
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(-1, self.input_size)  # Flatten the image
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    #Training des Modells    

    print('==> Building model ..')
    model = MLP(IMG_RES, HIDDEN_SIZE, NUM_CLASSES).to(device)

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
        'Batch Size' : BATCH_SIZE,
        'Hidden Layer Size': HIDDEN_SIZE,
        'Learn Rate' : INIT_LR
    }

# Konvertiere die Hyperparameter in einen lesbaren String
    hyperparameters_str = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('MLP - 2 Hidden Layers')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Füge eine Anmerkung mit den Hyperparametern hinzu
    plt.annotate(hyperparameters_str, xy=(0.5, 1.05), xytext=(0.5, 1.1),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=7, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xticks(rotation=45)  # Drehen Sie die Achsenbeschriftungen für bessere Lesbarkeit
    plt.yticks(rotation=45)
    plt.savefig('./confusion_Matrices/cm_CIFAR_MLP.png')  # Speichern Sie die Confusion Matrix als PNG-Datei
    plt.show()


if __name__ == '__main__':
    main()