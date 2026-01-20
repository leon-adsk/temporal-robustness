from enum import Enum
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
from PIL import Image
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
Image.MAX_IMAGE_PIXELS = None

class CnnClassifier(nn.Module):
    def __init__(self):
        super(CnnClassifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # TODO: three input channels but greyscale images: x = x.repeat(1, 3, 1, 1) -> repeat input tensor on channel dim

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
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x

def find_device(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    return device

def load_data(data: Path):
    BATCH_SIZE = 64*2*2
    img_size = 64
    test_size = 0.2

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data / 'images', transform=tfms)
    new_targets = [0 if t == 0 else 1 for t in dataset.targets]
    dataset.targets = new_targets
    dataset.samples = [(path, 0 if label == 0 else 1) for path, label in dataset.samples]

    benign_idxs = []
    past_idxs = []
    future_idxs = []

    for i in range(len(dataset)):
        path, _ = dataset.samples[i]
        
        if 'benign' in path:
            benign_idxs.append(i)
        if 'past' in path:
            past_idxs.append(i)
        if 'future' in path:
            future_idxs.append(i)

    # downsample because of massive past sample count
    min_size = min(len(benign_idxs), len(future_idxs), len(past_idxs))

    np.random.shuffle(benign_idxs)
    benign_idxs = benign_idxs[:min_size]

    np.random.shuffle(past_idxs)
    past_idxs = past_idxs[:min_size]

    np.random.shuffle(future_idxs)
    future_idxs = future_idxs[:min_size]

    benign_data = Subset(dataset, benign_idxs)
    past_data = Subset(dataset, past_idxs)
    future_data = Subset(dataset, future_idxs)

    ## Make dataloaders for each RQ, this could probably be nicer but whatever
    experiments = []

    # RQ 2: Baseline train(benign, past, future) -> test(benign, past, future)
    baseline_data = ConcatDataset([benign_data, past_data, future_data])

    train_len = int(len(baseline_data) * (1-test_size))
    test_len = len(baseline_data) - train_len

    baseline_train, baseline_test = random_split(baseline_data, [train_len, test_len])

    baseline_train_loader = DataLoader(baseline_train, batch_size=BATCH_SIZE, shuffle=True)
    baseline_test_loader = DataLoader(baseline_test, batch_size=BATCH_SIZE, shuffle=False)

    experiments.append(('rq2_baseline', (baseline_train_loader, baseline_test_loader)))

    # RQ 3: forward train(benign, past) -> test(benign, future)
    train_len = int(len(benign_data) * (1-test_size))
    test_len = len(benign_data) - train_len

    benign_train, benign_test = random_split(benign_data, [train_len, test_len])

    forward_train = ConcatDataset([benign_train, past_data])
    forward_test = ConcatDataset([benign_test, future_data])

    forward_train_loader = DataLoader(forward_train, batch_size=BATCH_SIZE, shuffle=True)
    forward_test_loader = DataLoader(forward_test, batch_size=BATCH_SIZE, shuffle=False)

    experiments.append(('rq3_forward', (forward_train_loader, forward_test_loader)))

    # RQ 4: backward train(benign, future) -> test(benign, past)
    benign_train, benign_test = random_split(benign_data, [train_len, test_len])

    backward_train = ConcatDataset([benign_train, future_data])
    backward_test = ConcatDataset([benign_test, past_data])

    backward_train_loader = DataLoader(backward_train, batch_size=BATCH_SIZE, shuffle=True)
    backward_test_loader = DataLoader(backward_test, batch_size=BATCH_SIZE, shuffle=False)

    experiments.append(('rq4_backward', (backward_train_loader, backward_test_loader)))

    return experiments

def test(net, val_loader, device):
    net.eval()

    total_samples = 0
    correct_preds = 0
    all_preds = []
    all_targets = []
    losses = []

    criterion = nn.BCEWithLogitsLoss() # apply sigmoid in loss as this is more robust (according to pytorch)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data, targets = data.to(device), targets.to(device)
            targets = torch.unsqueeze(targets, 1).float()

            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)

            total_samples += targets.size(0)

            outputs = net(data)
            losses.append(criterion(outputs, targets).item())

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (predicted == targets).sum().item()

            all_preds.extend(predicted.flatten().cpu().numpy())
            all_targets.extend(targets.flatten().cpu().numpy())
    
    loss = np.mean(losses)
    accuracy = correct_preds / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary'
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
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            targets = torch.unsqueeze(targets, 1).float()
            
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
        

class Experiment(Enum):
    BASELINE = 0
    FORWARD = 1
    BACKWARD = 2

def run_temporal(results_dir: Path, dataset, runs, epochs, lr, experiment: Experiment):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    experiments = load_data(dataset)
    experiment, loaders = experiments[experiment.value]
    results = []

    res_csv = results_dir / f'temporal_{experiment}_results.csv'
    trn_csv = results_dir / f'temporal_{experiment}_training.csv'

    training_metrics = []

    for run in range(runs):
        net = CnnClassifier()
        device = find_device(net)
        metrics = train(net, loaders[0], loaders[1], device, epochs, lr)
            
        results.append(metrics[-1])
        results_df = pd.DataFrame(data=results)
        results_df.to_csv(res_csv, index=False, header=not res_csv.is_file(), mode='a')
      
        metrics = [{ **d, 'epoch': i, 'run': run } for i, d in enumerate(metrics)]
        training_metrics += metrics
        
    metrics_df = pd.DataFrame(data=training_metrics)
    metrics_df.to_csv(trn_csv, index=False)
