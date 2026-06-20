# COMP5541 Assignment - Question 2: AlexNet on Tiny ImageNet

## 项目概述

PolyU COMP5541（Machine Learning & Data Analytics）课程作业第 2 题。任务是用 **PyTorch 从头实现 AlexNet**，并在 **Tiny ImageNet-200** 数据集上完成训练与评估。

---

## 项目结构

```
question-2-project/
├── COMP5541_AlexNet_TinyImageNet.ipynb   # 主 Notebook（核心文件，已运行完成）
├── dataset-tiny-imagenet-200/            # 本地原始数据集
│   ├── train/        # 200 类，每类 500 张 (共 100,000 张，64×64 彩色图)
│   ├── val/          # 验证集（val_annotations.txt + images/）
│   ├── test/         # 测试集（共 10,000 张）
│   ├── wnids.txt     # 200 个类别的 WordNet ID
│   └── words.txt     # WordNet ID → 人类可读类别名映射
└── README.md
```

> **注意**：Notebook 中的数据加载代码会将数据下载/解压到运行时工作目录的 `./data/tiny-imagenet-200/`，并自动生成 `val_by_class/` 子目录（将 val 按类别重组为 ImageFolder 格式）。

---

## 数据集说明

- **Tiny ImageNet-200**：ImageNet 的子集，64×64 彩色图，200 个类别
- 训练集：100,000 张（每类 500 张）
- 验证/测试集：各 10,000 张
- 数据归一化均值：`[0.4802, 0.4481, 0.3975]`，标准差：`[0.2302, 0.2265, 0.2262]`

---

## AlexNet 架构（适配 64×64 输入）

原始 AlexNet 针对 224×224 输入设计，本项目对其进行了调整：

| 层 | 类型 | 参数 | 输出尺寸 |
|----|------|------|---------|
| Conv1 | Conv2d + ReLU + MaxPool | 3→64, k=5, pad=2, pool=2 | 64×32×32 |
| Conv2 | Conv2d + ReLU + MaxPool | 64→192, k=3, pad=1, pool=2 | 192×16×16 |
| Conv3 | Conv2d + ReLU | 192→384, k=3, pad=1 | 384×16×16 |
| Conv4 | Conv2d + ReLU | 384→256, k=3, pad=1 | 256×16×16 |
| Conv5 | Conv2d + ReLU + MaxPool | 256→256, k=3, pad=1, pool=2 | 256×8×8 |
| AvgPool | AdaptiveAvgPool2d | — | 256×2×2 |
| FC1 | Linear + Dropout(0.5) | 1024 | — |
| FC2 | Linear + Dropout(0.5) | 512 | — |
| FC3 | Linear | 200 | — |

---

## 超参数

| 参数 | 值 |
|------|----|
| Epochs | 30 |
| Batch Size | 128 |
| Learning Rate | 0.01 |
| Optimizer | SGD (momentum=0.9, Nesterov=True) |
| Weight Decay | 5e-4 |
| Loss | CrossEntropyLoss |
| Dropout | 0.5 |
| 数据增强 | RandomCrop(64, pad=4) + RandomHorizontalFlip |

---

## 训练结果

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|----------|----------|
| 10 | 3.4448 | 20.57% | 3.4015 | 21.79% |
| 20 | 2.7132 | 34.49% | 2.6302 | 36.40% |
| 27 | 2.4674 | 39.46% | 2.4733 | **40.06%** |
| 30 | 2.3912 | 40.82% | 2.6289 | 38.09% |

**最终测试集准确率：40.86%**（200 类随机基线为 0.5%）

> 训练约 30 轮 × ~70 秒/轮，总耗时约 35 分钟（CPU 环境）。

---

## 当前状态

- [x] Notebook 已完整运行，包含所有训练日志和测试结果
- [x] 数据集已本地存放于 `dataset-tiny-imagenet-200/`
- [ ] 模型权重未保存（可添加 `torch.save` 代码）
- [ ] 未使用 Learning Rate Scheduler（可进一步优化）

---

## 参考资料

- [AlexNet 原论文 (NeurIPS 2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Writing AlexNet from Scratch in PyTorch](https://blog.paperspace.com/alexnet-pytorch/)
- [PyTorch AlexNet Hub](https://pytorch.org/hub/pytorch_vision_alexnet/)
