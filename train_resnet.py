import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.utils import make_grid, save_image
from torchsummary import summary
from torchviz import make_dot
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 定义超参数
# =========================
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# =========================
# 2. 设备设置
# =========================
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"设备: GPU ({device_name})")
else:
    device = torch.device('cpu')
    print(f"设备: CPU")

# =========================
# 3. 数据加载与预处理
# =========================
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet 要求输入尺寸为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# 4. 定义网络结构 (ResNet-18)
# =========================
class ResNetForMNIST(nn.Module):
    def __init__(self):
        super(ResNetForMNIST, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数、优化器
model = ResNetForMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建输出目录
output_dir = './output1'
os.makedirs(output_dir, exist_ok=True)

# 用于记录训练过程
train_losses = []
train_accuracies = []
val_accuracies = []

# =========================
# 5. 训练模型
# =========================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # 计算本轮平均损失和训练准确率
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    
    # 验证准确率
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

# =========================
# 6. 测试与评估
# =========================
model.eval()
correct = total = 0
all_labels = []  # 存储所有真实标签
all_preds = []   # 存储所有预测结果
wrong_predictions = []  # 存储错误预测的样本

# 还原第一个 batch 用于可视化
batch1_images = batch1_preds = batch1_labels = None

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        # 收集错误预测的样本
        for j in range(images.size(0)):
            if predicted[j] != labels[j]:
                wrong_predictions.append({
                    'image': images[j].cpu(),
                    'true_label': labels[j].cpu().item(),
                    'pred_label': predicted[j].cpu().item()
                })

        if i == 0:
            batch1_images = images.cpu()
            batch1_preds = predicted.cpu()
            batch1_labels = labels.cpu()

# 打印总体准确率
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

# =========================
# 7. 训练过程可视化
# =========================
# 训练损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('ResNet Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 训练和验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'b-', label='Training Accuracy')
plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
plt.title('ResNet Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_curves.jpg'))
plt.close()

# =========================
# 8. 混淆矩阵可视化
# =========================
cm = confusion_matrix(all_labels, all_preds)
fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
# Display 自动绘制热力图并标注
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm)
plt.title('ResNet Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.jpg'))
plt.close(fig_cm)

# =========================
# 9. 错误预测样本可视化
# =========================
if len(wrong_predictions) > 0:
    # 选择前16个错误预测样本
    num_wrong = min(16, len(wrong_predictions))
    wrong_samples = wrong_predictions[:num_wrong]
    
    plt.figure(figsize=(12, 8))
    for idx, sample in enumerate(wrong_samples):
        plt.subplot(4, 4, idx + 1)
        img = sample['image'].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {sample["pred_label"]}\nTrue: {sample["true_label"]}', 
                 color='red' if sample["pred_label"] != sample["true_label"] else 'green')
        plt.axis('off')
    
    plt.suptitle(f'ResNet 错误预测样本 (共{len(wrong_predictions)}个错误)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wrong_predictions.jpg'))
    plt.close()

# =========================
# 10. 各类别准确率统计
# =========================
class_accuracy = {}
for i in range(10):
    class_mask = np.array(all_labels) == i
    if np.sum(class_mask) > 0:
        class_correct = np.sum((np.array(all_preds) == i) & class_mask)
        class_total = np.sum(class_mask)
        class_accuracy[i] = 100 * class_correct / class_total

plt.figure(figsize=(10, 6))
classes = list(class_accuracy.keys())
accuracies = list(class_accuracy.values())
colors = ['green' if acc >= 95 else 'orange' if acc >= 90 else 'red' for acc in accuracies]

bars = plt.bar(classes, accuracies, color=colors)
plt.title('ResNet 各类别准确率')
plt.xlabel('数字类别')
plt.ylabel('准确率 (%)')
plt.ylim(0, 100)

# 在柱状图上添加数值标签
for bar, acc in zip(bars.patches, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'class_accuracy.jpg'))
plt.close()

# =========================
# 11. Batch1 预测可视化
# =========================
if batch1_images is not None:
    # 将前 16 张图做成 4x4 网格
    grid = make_grid(batch1_images[:16], nrow=4, normalize=True, pad_value=1)
    save_image(grid, os.path.join(output_dir, 'batch1_pred.jpg'))

    # 在网格上标注预测/真实标签
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    for idx in range(16):
        row, col = divmod(idx, 4)
        pred = batch1_preds[idx].item()
        label = batch1_labels[idx].item()
        color = 'green' if pred == label else 'red'
        ax.text(col*56+2, row*56+30, f'P:{pred}/T:{label}',
                color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.savefig(os.path.join(output_dir, 'batch1_pred_label.jpg'))
    plt.close(fig)

# =========================
# 12. 随机选取8张测试集图片进行可视化
# =========================
examples = []
labels_list = []
preds_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(images.size(0)):
            examples.append(images[i].cpu())
            labels_list.append(labels[i].cpu().item())
            preds_list.append(predicted[i].cpu().item())
        if len(examples) >= 8:
            break

examples = examples[:8]
labels_list = labels_list[:8]
preds_list = preds_list[:8]

plt.figure(figsize=(12, 4))
for idx, img in enumerate(examples):
    img = img.squeeze().numpy()
    plt.subplot(2, 4, idx + 1)
    plt.imshow(img, cmap='gray')
    color = 'green' if preds_list[idx] == labels_list[idx] else 'red'
    plt.title(f'Pred: {preds_list[idx]}, True: {labels_list[idx]}', color=color)
    plt.axis('off')
plt.suptitle('ResNet MNIST预测结果可视化')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'resnet_predictions.jpg'))
plt.close()

# =========================
# 13. 模型性能总结
# =========================
plt.figure(figsize=(10, 6))
summary_data = {
    '总体准确率': accuracy,
    '训练损失': train_losses[-1],
    '训练准确率': train_accuracies[-1],
    '验证准确率': val_accuracies[-1]
}

metrics = list(summary_data.keys())
values = list(summary_data.values())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = plt.bar(metrics, values, color=colors)
plt.title('ResNet 模型性能总结')
plt.ylabel('数值')

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    if '准确率' in bar:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.2f}%', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'model_summary.jpg'))
plt.close()

# =========================
# 14. 保存模型
# =========================
torch.save(model.state_dict(), 'resnet_mnist.pth')

# =========================
# 15. 总结输出
# =========================
print(f"\n{'='*40}")
print(f"训练完成！最终测试准确率: {accuracy:.2f}%")
print(f"所有可视化结果已保存在: {output_dir}")
print(f"生成的文件包括:")
print(f"- training_curves.jpg: 训练损失和准确率曲线")
print(f"- confusion_matrix.jpg: 混淆矩阵")
print(f"- wrong_predictions.jpg: 错误预测样本")
print(f"- class_accuracy.jpg: 各类别准确率")
print(f"- batch1_pred.jpg: 第一个batch预测结果")
print(f"- batch1_pred_label.jpg: 带标签的预测结果")
print(f"- resnet_predictions.jpg: 随机预测样本")
print(f"- model_summary.jpg: 模型性能总结")
print(f"{'='*40}")

if __name__ == '__main__':
    # 可视化ResNet结构
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetForMNIST().to(device)
    print('ResNet结构与参数统计：')
    summary(model, (1, 224, 224))
    X = torch.rand(size=(1, 1, 224, 224)).to(device)
    Y = model(X)
    make_dot(Y, params=dict(model.named_parameters())).render("output1/resnet_graph", format="pdf")
