import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# !pip install torchsummary
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet36(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet36, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (adjust layers and channels for ResNet-36)
        self.layer1 = self._make_layer(ResNetBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Example usage (replace with your desired number of classes)
num_classes = 10  # Example number of output classes
model = ResNet36(num_classes)

# Print the model architecture
print(model)

# Train data transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[x * 255 for x in (0.4914, 0.4822, 0.4466)]),
    A.Normalize(mean=(0.4914, 0.4822, 0.4466), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])


# Test data transformations
test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4466), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])

# Wrapper function for Albumentations transforms
def albumentations_transform_wrapper(transform):
    def wrapper(img):
        # Convert PIL Image to NumPy array
        img = np.array(img)
        # Apply Albumentations transform with keyword argument
        augmented = transform(image=img)
        return augmented['image']
    return wrapper

train_transform_wrapper = albumentations_transform_wrapper(train_transforms)
test_transform_wrapper = albumentations_transform_wrapper(test_transforms)

# Train data transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[x * 255 for x in (0.4914, 0.4822, 0.4466)]),
    A.Normalize(mean=(0.4914, 0.4822, 0.4466), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])


# Test data transformations
test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4466), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])

# Wrapper function for Albumentations transforms
def albumentations_transform_wrapper(transform):
    def wrapper(img):
        # Convert PIL Image to NumPy array
        img = np.array(img)
        # Apply Albumentations transform with keyword argument
        augmented = transform(image=img)
        return augmented['image']
    return wrapper

train_transform_wrapper = albumentations_transform_wrapper(train_transforms)
test_transform_wrapper = albumentations_transform_wrapper(test_transforms)

batch_size = 64

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': False}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet36().to(device)
summary(model, input_size=(3, 32, 32))

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    # Corrected test loss calculation: divide by the number of batches
    test_loss /= len(test_loader)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# device = "cpu"
model = ResNet36(num_classes=100).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# New Line
criterion = nn.CrossEntropyLoss()
num_epochs = 120

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion)
  test(model, device, train_loader, criterion)
  scheduler.step()
