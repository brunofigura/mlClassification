# LeNet5 Architektur kann Input-Dimensionen von 32x32x1 annehmen. Damit ist es auf Grausstufenbilder ausgelegt mit nur einem "Helligkeits"-Kanal

# Libraries importieren
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
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
    
    #Train und Val Daten-Split

    # this function will generate random indexes between 0 and 59999
    def split_indices(n, val_per, seed = 0):
        n_val = int(n * val_per)
        np.random.seed(seed)
        idx = np.random.permutation(n)
        return idx[n_val : ], idx[: n_val]

    val_per = 0.2
    rand_seed = 42

    train_indices, val_indices = split_indices(len(train_dataset), val_per, rand_seed)

    print(len(train_indices), len(val_indices))
        
    # Lets plot some indexes

    print("Validation Indices: ", val_indices[:20])
    print("Training Indices: ", train_indices[:20])


    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, sampler = train_sampler)
    
    val_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size, sampler = val_sampler)

    def show_batch(dataLoader):
        for img, label in dataLoader:
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(torchvision.utils.make_grid(img[:110], 10).permute(1,2,0))
            break
    
    show_batch(val_loader)
    plt.show(block=False)
    plt.pause(1)
    
    ###################
    #NETZ-MODELLIERUNG#
    ###################

   
    
    ##########
    #TRAINING#
    ##########

    

    
                
    #########
    #TESTING#
    #########

    
    

if __name__ == '__main__':
    main()