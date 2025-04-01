# 👚 Fashion MNIST CNN Classifier 📊

A convolutional neural network implementation for classifying Fashion MNIST images using PyTorch.

## ✨ Performance

**95% accuracy** on the Fashion MNIST test set! 🎯
This model successfully identifies fashion items with high precision.

## 🏗️ Model Architecture

This project uses a deep CNN with 5 convolutional blocks followed by a fully connected classifier:

- **5 Convolutional Blocks**: Each with Conv2D → BatchNorm → ReLU → MaxPool
- **Classifier**: 2-layer fully connected network (1024 → 1024 → output classes)

## 🛠️ Requirements

- torch>=1.7.0
- torchvision>=0.8.1
- numpy
- matplotlib

## 📊 Dataset

The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset consists of 60,000 training and 10,000 test grayscale images of 10 fashion categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot
