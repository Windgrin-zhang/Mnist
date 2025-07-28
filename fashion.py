import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.FashionMNIST(root='./fashion_mnist', train=True,
                                                  transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./fashion_mnist', train=False,
                                                 transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
