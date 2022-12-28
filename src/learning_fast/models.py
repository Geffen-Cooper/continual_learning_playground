import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# model for classifying 28x28 digits (i.e. MNIST, SVHN, USPS)
class DigitClassifierBig(nn.Module):
    def __init__(self):
        super(DigitClassifierBig, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) # 1 x 28 x 28 --> 32 x 26 x 26
        x = F.relu(x)

        x = self.conv2(x) # 32 x 26 x 26 --> 64 x 24 x 24
        x = F.relu(x)

        x = F.max_pool2d(x, 2) # 64 x 24 x 24 --> 64 x 12 x 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 64 x 12 x 12 --> 64*12*12 x 1

        x = self.fc1(x) # 64*12*12 x 1 --> 128 x 1
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x) # 128 x 1 --> 10 x 1
        output = F.log_softmax(x, dim=1)

        return output


# model for classifying 28x28 digits (i.e. MNIST, SVHN, USPS)
class DigitClassifierSmall(nn.Module):
    def __init__(self):
        super(DigitClassifierSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x) # 1 x 28 x 28 --> 16 x 26 x 26
        x = F.relu(x)

        x = self.conv2(x) # 16 x 26 x 26 --> 32 x 24 x 24
        x = F.relu(x)

        x = F.max_pool2d(x, 2) # 32 x 24 x 24 --> 32 x 12 x 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 32 x 12 x 12 --> 32*12*12 x 1

        x = self.fc1(x) # 32*12*12 x 1 --> 64 x 1
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x) # 64 x 1 --> 10 x 1
        output = F.log_softmax(x, dim=1)

        return output