import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLP(nn.Module):
        def __init__(self,input_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.fc1 = nn.Linear(input_size, 450)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(450, 200)
            self.fc3 = nn.Linear(200, 100)
            self.fc4 = nn.Linear(100, output_size)

        def forward(self, x):
            x = x.view(-1, self.input_size)  # Flatten the image
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            return x
        
class Classifier:
    def __init__(self, n_epochs, init_lr):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 64
        self.valid_ratio = 0.1

        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.img_res = 32 * 32 * 3
        self.num_classes = 10

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])



        self.train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform)

        self.test_dataset = torchvision.datasets.CIFAR10(root='./data',
                            train=False,
                            download=True,
                            transform=transform)
        
        #Validierungs-Set vom Trainingsdatensatz erzeugen
        num_train = len(self.train_dataset)
        num_valid = int(self.valid_ratio * num_train)
        num_train = num_train - num_valid

        self.train_subset, self.valid_subset = torch.utils.data.random_split(self.train_dataset, [num_train, num_valid])
            
        self.train_loader = torch.utils.data.DataLoader(self.train_subset,
                                    shuffle=True,
                                    batch_size=self.batch_size)
        
        self.valid_loader = torch.utils.data.DataLoader(self.valid_subset, shuffle=True, batch_size=self.batch_size)


        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size=self.batch_size)
        



        self.network = MLP(self.img_res, self.num_classes)
        self.network = self.network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), self.init_lr)
        self.criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss instead of NLLLoss
        
        self.train_losses = []
        self.train_counter = []
        self.valid_losses = []
        self.valid_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]

    def train(self, epoch, log_interval):
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]  Loss: {loss.item():.6f}', end='\r')
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx * self.batch_size) + ((epoch - 1) * len(self.train_loader.dataset)))

    def validate(self):
        self.network.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                valid_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            valid_loss /= len(self.valid_loader)
            self.valid_losses.append(valid_loss)
            print(f'\nValidation set: Avg. loss: {valid_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.valid_loader.dataset)}'
              f'({100. * correct / len(self.valid_loader.dataset):.0f}%)\n')

    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        self.test_losses.append(test_loss)
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.test_loader.dataset)}'
              f'({100. * correct / len(self.test_loader.dataset):.0f}%)\n')
        
    def plot_val_train_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.valid_losses, label='Validation Loss', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

def main():
    n_epochs = 4
    log_interval = 10
    init_lr = 0.001
    cl = Classifier(n_epochs, init_lr)
    cl.test()

    for epoch in range(1, n_epochs + 1):
        cl.train(epoch, log_interval)
        cl.validate()

    cl.plot_val_train_losses()

if __name__ == '__main__':
    main()
