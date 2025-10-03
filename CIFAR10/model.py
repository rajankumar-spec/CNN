import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        )
        
      self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, groups = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        )
        
      self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05)
        )
        
      self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )
      self.gap= nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
