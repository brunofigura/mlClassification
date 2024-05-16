# LeNet5 Architektur kann Input-Dimensionen von 32x32x1 annehmen. Damit ist es auf Grausstufenbilder ausgelegt mit nur einem "Helligkeits"-Kanal

# Libraries importieren
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 


def main():

    # Hyperparamater
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

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
    
    
    #Das Netz modellieren

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


if __name__ == '__main__':
    main()