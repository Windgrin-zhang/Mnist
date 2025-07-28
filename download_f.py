#!/usr/bin/env python3
# -*- encoding utf-8 -*-

'''
@File: save_fashionmnist_to_jpg.py
@Date: 2025-07-25
@Author: YourName
@Version: 0.0.1
@Desc:
    1. 通过 torchvision.datasets.FashionMNIST 下载、解压和读取 FashionMNIST 数据集；
    2. 使用 PIL.Image.save 将 FashionMNIST 数据集中的灰度图片以 JPEG 格式保存。
'''

import sys, os
sys.path.insert(0, os.getcwd())

from torchvision.datasets import FashionMNIST
from tqdm import tqdm

if __name__ == "__main__":
    # 图片保存路径
    root = 'fashionmnist_jpg'
    if not os.path.exists(root):
        os.makedirs(root)

    # 下载并加载 FashionMNIST 数据集
    training_dataset = FashionMNIST(
        root='fashion_mnist',
        train=True,
        download=True,
    )
    test_dataset = FashionMNIST(
        root='fashion_mnist',
        train=False,
        download=True,
    )

    # 保存训练集图片
    with tqdm(total=len(training_dataset), ncols=150) as pro_bar:
        for idx, (img, label) in enumerate(training_dataset):
            # 文件名格式: training_索引_标签.jpg
            f = os.path.join(root, f"training_{idx}_{label}.jpg")
            img.save(f)
            pro_bar.update(1)

    # 保存测试集图片
    with tqdm(total=len(test_dataset), ncols=150) as pro_bar:
        for idx, (img, label) in enumerate(test_dataset):
            # 文件名格式: test_索引_标签.jpg
            f = os.path.join(root, f"test_{idx}_{label}.jpg")
            img.save(f)
            pro_bar.update(1)
