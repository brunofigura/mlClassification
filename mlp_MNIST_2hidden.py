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
        def __init__(self, input_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, output_size)
            self.b_norm1 = nn.BatchNorm1d(128)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            x = x.view(-1, self.input_size)  # Flatten the image
            x = self.fc1(x)
            x = self.b_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        
class Classifier:
    def __init__(self, n_epochs, init_lr):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Legt ein Seed fest, damit die Datensätze immer GLEICH zufällig gemischt werden, für reproduzierbare Ergebnisse
        torch.manual_seed(42)

        self.batch_size = 128
        self.valid_ratio = 0.1

        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.img_res = 28 * 28 * 1     
        self.num_classes = 10



           

        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
        ])



        self.train_dataset = torchvision.datasets.MNIST(root='./data',
                                train=True,
                                download=True,
                                transform=transform)

        self.test_dataset = torchvision.datasets.MNIST(root='./data',
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
        
        self.valid_batch_losses = []
        self.train_batch_losses = []
        self.train_losses = []
        self.train_counter = []
        self.valid_losses = []
        self.valid_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]

        
        total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')


    def train(self, epoch, log_interval):
        self.network.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]  Loss: {loss.item():.6f}', end='\r')
        train_loss /= len(self.train_loader)
        self.train_losses.append(train_loss)

    def validate(self):
        self.network.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.network(images)
                valid_loss += self.criterion(output, labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
            valid_loss /= len(self.valid_loader)
            self.valid_losses.append(valid_loss)
            print(f'\nValidation set: Avg. loss: {valid_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.valid_loader.dataset)}'
              f'({100. * correct / len(self.valid_loader.dataset):.0f}%)\n')

    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.network(images)
                test_loss += self.criterion(output, labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(self.test_loader)
        self.test_losses.append(test_loss)
        accuracy = correct / len(self.test_loader.dataset)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.test_loader.dataset)}'
              f'({100. * correct / len(self.test_loader.dataset):.0f}%)\n'
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.04f}')
        
        self.conf_matrix = confusion_matrix(all_labels, all_predictions)

        
    def plot_val_train_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.valid_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()

        # Save the plot to a directory
        directory = './loss_plots'
        os.makedirs(directory, exist_ok=True)
        plot_filename = f'TrainValLoss_mlp2_mnist_{self.n_epochs}-Epochs.png'
        plot_path = os.path.join(directory, plot_filename)
        plt.savefig(plot_path)
        print(f'Plot gespeichert unter {plot_path}')
        plt.show()

        
    def plot_confMatrix(self):
        class_names = ['One', 'Two', 'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine', 'Ten']
        
        hyperparameters = {
            'Epochs' : self.n_epochs,
            'Initial Learn-Rate' : self.init_lr ,
            'Batch Size' : self.batch_size
        }
        # Konvertiere die Hyperparameter in einen lesbaren String
        hyperparameters_str = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('MNSIT - MLP 2 Hidden Layers')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Füge eine Anmerkung mit den Hyperparametern hinzu
        plt.annotate(hyperparameters_str, xy=(0.5, 1.05), xytext=(0.5, 1.1),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=7, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.xticks(rotation=45)  # Drehen Sie die Achsenbeschriftungen für bessere Lesbarkeit
        plt.yticks(rotation=45)
        plt.savefig(f'./confusion_Matrices/cm_MNIST_MLP_2_hiddenL_{self.n_epochs}-Epochs.png')  # Speichern Sie die Confusion Matrix als PNG-Datei
        plt.show()

    def saveModelWeights(self, epoch):
        modelSavePath = f'./trainedModels/mlp_mnist_epoch_{epoch}.ckpt'
        os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)
        torch.save(self.network.state_dict(), modelSavePath)
        print(f'Modell wurde unter {modelSavePath} gespeichert.')

    def loadModelWeights(self, epoch):
        modelLoadPath = f'./trainedModels/mlp_mnist_epoch_{epoch}.ckpt'
        if os.path.exists(modelLoadPath):
            self.network.load_state_dict(torch.load(modelLoadPath))
            self.network.to(self.device)
            print(f'Modell wurde aus {modelLoadPath} geladen.')
        else:
            print(f'Keine gespeicherten Gewichte unter {modelLoadPath} gefunden.')



def main():
    n_epochs = 20
    log_interval = 10
    init_lr = 0.0001
    cl = Classifier(n_epochs, init_lr)
    cl.test()

    for epoch in range(1, n_epochs + 1):
        cl.train(epoch, log_interval)
        cl.validate()

    cl.plot_val_train_losses()

    cl.test()
    cl.plot_confMatrix()

    cl.saveModelWeights(n_epochs)

if __name__ == '__main__':
    main()
