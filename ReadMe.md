# MNIST/FashionMNIST 项目说明

## 目录结构

```
Mnist/
├── download_f.py
├── ReadMe.md
└── mnist/
    └── raw/
        ├── t10k-images-idx3-ubyte
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte
        └── train-labels-idx1-ubyte.gz
```

## 文件说明

### download/download_f.py

用于下载 MNIST手写数字/FashionMNIST 数据集，并将图片保存为 JPEG 格式。  
主要功能：
- 自动下载训练集和测试集
- 将图片按标签和索引命名保存为 JPG 文件
- 进度条显示处理进度



### demo
主要用于测试train.py train_resnet.py中lenet和resnet的超参最佳设置
主要功能：
- 使用optuna方式，先预设参数
```
    batch_size = trial.suggest_categorical('batch_size', [16 ,32 ,64 ,128 ])
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    num_epochs = trial.suggest_categorical('epochs', [5,  10, 20])
```
在batch_size、lr、num_epochs中预设可能会使用到的参数

### train文件
用于训练模型
- 输出训练时各参数图片于各自output文件夹（无需手动生成）
- 输出训练准确率和验证准确率


## 使用方法

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 可运行下载脚本：
   ```
   python download.py
   python download_f.py
   ```
3. 可查看生成的 JPG 图片文件夹 `fashionmnist_jpg`。
4. 也可直接运行train中加载源文件的部分训练。
5. 得到参数

## 适用场景

- 深度学习、简单灰度图像分类任务的数据准备
- MNIST/FashionMNIST 数据集的格式转换与可视化

---
