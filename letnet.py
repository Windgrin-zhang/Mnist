import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda")

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset= datasets.Imagefolder(root='/public/data/image/mnist/mnist_png/training', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = datasets.Imagefolder(root='/public/data/image/mnist/mnist_pne/testing', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)