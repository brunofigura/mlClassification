import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 

def main():

    batch_size = 640

    # Define the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define a function to load models
    def load_model(model_path):
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model

    # Define a function to evaluate models
    def evaluate_model(model, data_loader):
        y_true = []
        y_pred = []
        y_probs = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device).float(), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                probs = F.softmax(output, dim=1)
                y_probs.extend(probs.cpu().numpy())
                pred = output.argmax(dim=1, keepdim=True)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(target.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, [p[1] for p in y_probs], pos_label=1)
        roc_auc = auc(fpr, tpr)

        return {
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_matrix': conf_matrix,
            'roc_auc': roc_auc
        }

    # Define a function to plot confusion matrix
    def plot_confusion_matrix(cm, model_name):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    # Manually specify the path to the model
    model_path = './trainedModels/LeNet5MNIST.ckpt'  # Beispielpfad

    # Load and evaluate the model
    model = load_model(model_path)
    result = evaluate_model(model, test_loader)

    # Print the evaluation results
    df_result = pd.DataFrame([result])
    print(df_result)

    # Plot the confusion matrix
    plot_confusion_matrix(result['conf_matrix'], os.path.basename(model_path))

if __name__ == '__main__':
    main()