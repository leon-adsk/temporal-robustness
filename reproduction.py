import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd



class ReproductionCNN(nn.Module):
    def __init__(self):
        super(ReproductionCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 

            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 25)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x

def find_device(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    return device

def load_data(data):
    img_size = 64
    test_size = 0.2 # Paper states 80/20 split

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=str(data), transform=tfms)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    split = splitter.split(X=np.zeros(len(full_dataset)), y=full_dataset.targets)
    train_idxs, val_idxs = next(split) # grab the first and only split out of iterator (n_splits=1)

    train_ds = Subset(full_dataset, train_idxs.tolist())
    val_ds = Subset(full_dataset, val_idxs.tolist())

    # DataLoaders
    batch_size = 64
    num_workers = 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def test(net, val_loader, device):
    net.eval()

    total_samples = 0
    correct_preds = 0
    all_preds = []
    all_targets = []
    losses = []

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, targets) in enumerate(val_loader):
        data, targets = data.to(device), targets.to(device)

        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)

        total_samples += targets.size(0)

        outputs = net(data)
        losses.append(criterion(outputs, targets).item())

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    loss = np.mean(losses)
    accuracy = correct_preds / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro'
    )

    return {
        'test_loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train(net, train_loader, val_loader, device, epochs, lr):
    train_losses = []
    stats = []
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)

            outputs = net(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step() 

            running_loss += loss.item()
    
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        stats.append(test(net, val_loader, device))
        stats[-1]['train_loss'] = avg_loss
        
    return stats

def run_reproduction(results_dir: Path, dataset, runs, epochs, lr):
    if runs == 0: return
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    results = []
    train_loader, val_loader = load_data(dataset)
    

    print(f"\r[{'#' * int((0)/runs*20):.<20}]", end="", flush=True)

    for run in range(runs):
        net = ReproductionCNN()
        device = find_device(net)
        metrics = train(net, train_loader, val_loader, device, epochs, lr)[-1]
        metrics['run'] = run
        results.append(metrics)
        print(f"\r[{'#' * int((run+1)/runs*20):.<20}]", end="", flush=True)
        
    results_df = pd.DataFrame(data=results)
    results_df.to_csv(results_dir / 'reproduction_results.csv', index=False)
