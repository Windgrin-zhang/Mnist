import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import optuna
import time
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# LeNet-5 结构（来自train.py）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ResNet18 for MNIST（来自train_resnet.py）
class ResNetForMNIST(nn.Module):
    def __init__(self):
        super(ResNetForMNIST, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# 数据加载器
# 对LeNet和ResNet分别做不同的transform

def get_data_loaders(batch_size, for_resnet=False):
    if for_resnet:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    train_dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def objective(trial, model_class, device, name, for_resnet=False):
    batch_size = trial.suggest_categorical('batch_size', [16 ,32 ,64 ,128 ])
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    num_epochs = trial.suggest_categorical('epochs', [5,  10, 20])
    train_loader, test_loader = get_data_loaders(batch_size, for_resnet)
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 验证准确率
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    elapsed = time.time() - start_time
    print(f"[{name}] 参数: batch_size={batch_size}, lr={lr:.5f}, epochs={num_epochs} -> val_acc={accuracy:.2f}%, 用时: {elapsed:.1f}秒")
    # 让Optuna以“达到99%准确率”为目标
    if accuracy >= 99.0:
        trial.report(accuracy, step=num_epochs)
        # raise optuna.exceptions.TrialPruned()  # 不早停，继续搜更高准确率
    return -accuracy  # 负号，Optuna默认最小化目标

def run_optuna(model_class, device, name, for_resnet=False, n_trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_class, device, name, for_resnet), n_trials=n_trials)
    print(f'==== {name} Optuna搜索结果 ====')
    for t in study.trials:
        acc = -t.value
        if acc >= 99.5:
            print(f'【{name}】高于99.5%准确率的参数: {t.params}，验证集准确率={acc:.2f}%')
    print(f'最优参数: {study.best_params}, 最优准确率: {-study.best_value:.2f}%')
    print('=============================')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_optuna(Net, device, 'LeNet', for_resnet=False, n_trials=20)
    run_optuna(ResNetForMNIST, device, 'ResNet', for_resnet=True, n_trials=20)
