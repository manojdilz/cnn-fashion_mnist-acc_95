import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchmetrics as metrics
from tqdm import tqdm

# Model HyperParameters & etc
IMG_SIZE = (64,64)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOWNLOAD_DATA = False
SAVE_MODEL=False
LOAD_MODEL=True
VALIDATE=True
train_transforms = T.Compose([
        T.Grayscale(),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize(IMG_SIZE)
    ])
val_transforms = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
    T.Resize(IMG_SIZE)
    ])

# Neural Network
class NeuralNet(nn.Module):
    def __init__(self, in_channels, out_dims):
        super(NeuralNet, self).__init__()

        self.conv1 = self.create_block(in_channels, 32, 5, 2)
        self.conv2 = self.create_block(32, 64, 3, 1)
        self.conv3 = self.create_block(64, 128, 3, 1)
        self.conv4 = self.create_block(128, 256, 3, 1)
        self.conv5 = self.create_block(256, 256, 3, 1)

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
                    nn.Linear(in_features=1024, out_features=1024),
                    nn.ReLU(),
                    nn.Linear(in_features=1024, out_features=out_dims)
                )

    def create_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        return self.classifier(x)


# Dataset and Dataloaders
train_ds = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=DOWNLOAD_DATA,
            transform=train_transforms
        )
val_ds = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=DOWNLOAD_DATA,
            transform=val_transforms
        )
train_loader = data.DataLoader(
           dataset=train_ds,
           batch_size=BATCH_SIZE,
           shuffle=True,
           num_workers=2
        )
val_loader = data.DataLoader(
            dataset=val_ds,
            batch_size=32,
            shuffle=True
        )

#Model, optimizer, criterion & metrics
model = NeuralNet(in_channels=1,out_dims=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.9, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
def accuracy_fn(pred, target):
    pred_classes = torch.argmax(F.softmax(pred, dim=-1), dim=-1)
    target_classes = torch.argmax(F.softmax(target, dim=-1), dim=-1)
    return ((pred_classes == target_classes).sum().float()) / pred.size(0)

#load pre-trained model
if LOAD_MODEL:
    model.load_state_dict(torch.load('model.bin'))

#evaluate the model
if VALIDATE:
    print('Evaluating model...')
    model.eval()
    loss_sum = 0
    accuracy_sum = 0
    for iteration, (data, target) in enumerate(tqdm(val_loader, leave=True)):
        data = data.to(DEVICE)
        target = F.one_hot(target, num_classes=NUM_CLASSES).float().to(DEVICE)
        y_pred = model(data)
        loss = criterion(y_pred, target)
        accuracy = accuracy_fn(y_pred, target)
        loss_sum += loss.item()
        accuracy_sum += accuracy.item()
    print(f'Loss: {loss_sum/(iteration+1):.4f}\t Accuracy: {accuracy_sum/(iteration+1):.4f}')
    exit()


#training
for epoch in range(EPOCHS):
    bar = tqdm(total=(len(train_loader)))
    loss_sum = 0
    accuracy_sum = 0
    for iteration, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = F.one_hot(target, num_classes=NUM_CLASSES).to(torch.float32).to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        bar.update(1)
        accuracy = accuracy_fn(y_pred, target)
        loss_sum += loss.item()
        accuracy_sum += accuracy.item()
        bar.set_description(f'Epoch: {epoch+1}/[{EPOCHS}]')
        bar.set_postfix(loss=loss_sum/(iteration+1), accuracy=accuracy_sum/(iteration+1))


#save the model
if SAVE_MODEL:
    torch.save(model.state_dict(), 'model.bin')