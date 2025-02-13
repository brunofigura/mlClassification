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

# Netz modellieren
class MLP(nn.Module):
        def __init__(self,input_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.fc1 = nn.Linear(input_size, 4608)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4608,2304)
            self.fc3 = nn.Linear(2304,1151) 
            self.fc4 = nn.Linear(1151,525)
            self.fc5 = nn.Linear(525, 525)
            self.fc6 = nn.Linear(525, 525)
            self.fc7 = nn.Linear(525, 525)
            self.fc8 = nn.Linear(525, 100)
            self.fc9 = nn.Linear(100, output_size)
            self.dropout = nn.Dropout(0.2)
            self.b_norm1 = nn.BatchNorm1d(4608)
            self.b_norm2 = nn.BatchNorm1d(2304)
            self.b_norm3 = nn.BatchNorm1d(525)
            self.b_norm4 = nn.BatchNorm1d(100)

        def forward(self, x):
            x = x.view(-1, self.input_size)  # Flatten the image
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.b_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.b_norm2(x)         
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc5(x)
            x = self.b_norm3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc6(x)
            x = self.b_norm3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc7(x)
            x = self.b_norm3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc8(x)
            x = self.b_norm4(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc9(x)
            return x
        
# Klasse mit der ein Netz trainiert, validiert und getestet werden kann 
class Classifier:
    def __init__(self, n_epochs, init_lr):
        
        
        #Training des Netzes auf einer NVIDIA-Grafikkarte falls vorhanden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #Legt ein Seed fest, damit die Datensätze immer GLEICH zufällig gemischt werden, für reproduzierbare Ergebnisse
        torch.manual_seed(42)

        #Batch-Size als vielfache von 8
        self.batch_size = 128
        #Bestimmen wieviel später vom Trainings-Set für Validierung benutzt werden soll
        self.valid_ratio = 0.1

        # Hyperparameter und Datensatz-Informationen übergeben
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.img_res = 28 * 28 * 3      #32x32 gecropped auf 28x28
        self.num_classes = 10

        
        transform = transforms.Compose([
                transforms.RandomRotation(5, fill=(0.2)),    #Trainingsdatensatz um + - 5 Grad zufällig rotieren
                transforms.RandomCrop(28, padding=2),       #2Pixel Rand erzeugen und davon einen zufälligen 28x28 Pixel Crop nehmen
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   #Datenwerte mithilfe Durchschnitts und Standard-Abweichung auf 0-1 Zahlenraum mappen
        ])


        #Datensätze laden, ggf. herunterladen
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
            
        #Dataloader aus den Datensätzen erzeugen
        self.train_loader = torch.utils.data.DataLoader(self.train_subset,
                                    shuffle=True,
                                    batch_size=self.batch_size)
        
        self.valid_loader = torch.utils.data.DataLoader(self.valid_subset, shuffle=True, batch_size=self.batch_size)


        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size=self.batch_size)
        


        # Netzwerk "laden"
        self.network = MLP(self.img_res, self.num_classes)
        self.network = self.network.to(self.device)

        # Algorithmus für den Gradientenabstieg festlegen
        self.optimizer = optim.Adam(self.network.parameters(), self.init_lr)
        # Loss über CrossEntropy-Loss berechnen
        self.criterion = nn.CrossEntropyLoss()  
        
        #Variablen für spätere Auswertung initieren
        self.train_losses = []
        self.train_counter = []
        self.valid_losses = []
        self.valid_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]

        #Anzahl der trainierbaren Parameter berechnen und ausgeben
        total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')

    #Methode zum trainieren eines Netztes
    def train(self, epoch, log_interval):
        self.network.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()              #Gradienten vor neuer Berechnung auf 0 setzen
            output = self.network(images)           #Vorhersage treffen
            loss = self.criterion(output, labels)   #Loss zwischen Vorhersage und der Wahrheit berechnen
            loss.backward()                         #Gradienten bestimmen
            self.optimizer.step()                   #Gewichte anpassen
            train_loss += loss.item()               
            if batch_idx % log_interval == 0:       #Konsolen-Ausgabe
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]  Loss: {loss.item():.6f}', end='\r')
        train_loss /= len(self.train_loader)        #Train-Loss pro Epoche berechnen und speichern
        self.train_losses.append(train_loss)        

    #Methode für die Validierung nach eimer Trainings-Epoche
    def validate(self):
        self.network.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():   #Performance optimierung durch ausschalten der Gradienten-Berechnung
            for images, labels in self.valid_loader:                                #Validierungs-Datensatz benutzen
                images, labels = images.to(self.device), labels.to(self.device)    
                output = self.network(images)                                       #Vorhersage treffen
                valid_loss += self.criterion(output, labels).item()                 #Vorhersage gegen die Wahrheit vergleichen
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
            valid_loss /= len(self.valid_loader)                                    #Validierungs-Loss berechnen und speichern
            self.valid_losses.append(valid_loss)
            print(f'\nValidation set: Avg. loss: {valid_loss:.4f}, '                #Konsolenausgabe der Validierungs-Werte
              f'Accuracy: {correct}/{len(self.valid_loader.dataset)}'
              f'({100. * correct / len(self.valid_loader.dataset):.0f}%)\n')

    #Methode zum testen des trainierten Netzes
    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:                             #Testdatensatz verwenden
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.network(images)                                   #Vorhersagen berechnen
                test_loss += self.criterion(output, labels).item()              #Loss berechnen
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()      #Schauen ob Vorhersage gleiche Klasse wie Label ist. Wenn ja als Korrekt speichern
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(self.test_loader)                                      #Durchschnittler Loss den Loss jedes Bildes berechnen
        self.test_losses.append(test_loss)              
        accuracy = correct / len(self.test_loader.dataset)                      #Eval-Metriken berechnen und auf der Konsoleausgeben
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.test_loader.dataset)}'
              f'({100. * correct / len(self.test_loader.dataset):.0f}%)\n'
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.04f}')
        
        self.conf_matrix = confusion_matrix(all_labels, all_predictions)        #Confusion-Matrix erstellen

    #Methode um Trainings- und Validierungs-Losses über die Epochen zu plotten    
    def plot_val_train_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.valid_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()

        # Plot in ein Ordner speichern
        directory = './loss_plots'
        os.makedirs(directory, exist_ok=True)
        plot_filename = f'TrainValLoss_mlp9_cifar_{self.n_epochs}-Epochs.png'
        plot_path = os.path.join(directory, plot_filename)
        plt.savefig(plot_path)
        print(f'Plot gespeichert unter {plot_path}')
        plt.show()

    #Methode um die erstelle Confusion-Matrix anzuzeigen und zu speichern    
    def plot_confMatrix(self):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        
        hyperparameters = {
            'Epochs' : self.n_epochs,
            'Initial Learn-Rate' : self.init_lr ,
            'Batch Size' : self.batch_size
        }
        # Konvertiere die Hyperparameter in einen lesbaren String
        hyperparameters_str = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])

        # Zeigt Confusion Matrix an
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('CIFAR - MLP 9 Hidden Layers')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Füge eine Anmerkung mit den Hyperparametern hinzu
        plt.annotate(hyperparameters_str, xy=(0.5, 1.05), xytext=(0.5, 1.1),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=7, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.xticks(rotation=45)  # Dreht die Achsenbeschriftungen für bessere Lesbarkeit
        plt.yticks(rotation=45)
        plt.savefig(f'./confusion_Matrices/cm_CIFAR_MLP_9_hiddenL_{self.n_epochs}-Epochs.png')  # Speichert die Confusion Matrix als PNG-Datei
        plt.show()

    #Methode um trainierte Gewichte eines Modells zu speichen
    def saveModelWeights(self, epoch):
        modelSavePath = f'./trainedModels/mlp9_cifar_epoch_{epoch}.ckpt'
        os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)
        torch.save(self.network.state_dict(), modelSavePath)
        print(f'Modell wurde unter {modelSavePath} gespeichert.')

    #Methode um gespeicherte Gewichte in ein Netz zu laden
    def loadModelWeights(self, epoch):
        modelLoadPath = f'./trainedModels/mlp9_cifar_epoch_{epoch}.ckpt'
        if os.path.exists(modelLoadPath):
            self.network.load_state_dict(torch.load(modelLoadPath))
            self.network.to(self.device)
            print(f'Modell wurde aus {modelLoadPath} geladen.')
        else:
            print(f'Keine gespeicherten Gewichte unter {modelLoadPath} gefunden.')

    def display_and_save_predictions(self, n_images=25, directory='./predictions'):
                self.network.eval()
                images, labels = next(iter(self.test_loader))
                images, labels = images.to(self.device), labels.to(self.device)
                
                output = self.network(images)
                preds = output.data.max(1, keepdim=True)[1]
                
                # CIFAR-10 Klassennamen
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']

                plt.figure(figsize=(10, 10))
                for i in range(n_images):
                    plt.subplot(5, 5, i+1)
                    img = images[i].cpu().permute(1, 2, 0).numpy()  # Transformiere die Tensor-Dimensionen für die Darstellung
                    img = img * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)  # Unnormalize das Bild
                    plt.imshow(img)
                    plt.title(f'Pred: {class_names[preds[i].item()]}')
                    plt.axis('off')
                
                # Speicher das Raster in ein Verzeichnis
                os.makedirs(directory, exist_ok=True)
                raster_path = os.path.join(directory, f'MLP9_CIFAR_{self.n_epochs}_epochs_predictions_grid.png')
                plt.savefig(raster_path)
                print(f'Raster gespeichert unter {raster_path}')
                plt.show()


# Main-Methode 
def main():
    #Hyperparameter festlegen
    n_epochs = 200
    log_interval = 10
    init_lr = 0.0001

    #Instanz des Klassifizieres erzeugen
    cl = Classifier(n_epochs, init_lr)
    #Test von zufällig initierten Gewichten
    cl.test()
    #cl.loadModelWeights(n_epochs)
    #Jede Epoche ein Trainings und ein Validierungsschritt
    for epoch in range(1, n_epochs + 1):
        cl.train(epoch, log_interval)
        cl.validate()

    #Plotten des Trainings- und Validierungs-Loss-Graphen
    cl.plot_val_train_losses()

    #Evaluieren des trainierten Netzes
    cl.test()
    #Plotten der Confusion-Matrix
    cl.plot_confMatrix()

    cl.display_and_save_predictions()
    #Speichern der optimierten Gewichte
    cl.saveModelWeights(n_epochs)


#Wirft ein Fehler wenn der Teil nicht existiert 
if __name__ == '__main__':
    main()

