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
    modelSavePath = './trainedModels/ResNetCIFAR10.ckpt'
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
    
    # 3x3 convolution
    def conv3x3(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

    # Residual block
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    # ResNet
    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=10):
            super(ResNet, self).__init__()
            self.in_channels = 16
            self.conv = conv3x3(3, 16)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self.make_layer(block, 16, layers[0])
            self.layer2 = self.make_layer(block, 32, layers[1], 2)
            self.layer3 = self.make_layer(block, 64, layers[2], 2)
            self.avg_pool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64, num_classes)

        def make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if (stride != 1) or (self.in_channels != out_channels):
                downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride),
                    nn.BatchNorm2d(out_channels))
            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    print('==> Building model ..')
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

    if not TRAIN_MODEL:
        print("loading wheigts ...")
        model.load_state_dict(torch.load(modelSavePath))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = model.to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

        # Variable Learn-Rate
        def update_lr(optimizer, lr):    
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Training des Modells
        total_step = len(trainloader)
        curr_lr = INIT_LR
        for epoch in range(NUM_EPOCHS):
            for i, (images, labels) in enumerate(trainloader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)
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
    plt.title('ResNet')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Füge eine Anmerkung mit den Hyperparametern hinzu
    plt.annotate(hyperparameters_str, xy=(0.5, 1.05), xytext=(0.5, 1.1),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=7, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xticks(rotation=45)  # Drehen Sie die Achsenbeschriftungen für bessere Lesbarkeit
    plt.yticks(rotation=45)
    plt.savefig('./confusion_Matrices/cm_CIFAR_ResNet.png')  # Speichern Sie die Confusion Matrix als PNG-Datei
    plt.show()


if __name__ == '__main__':
    main()