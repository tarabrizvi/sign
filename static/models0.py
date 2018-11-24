import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # 250 outputs * the 24*25 filtered/pooled map size
        self.fc1 = nn.Linear(128*24*24,250)
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(250, 136)
    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        # prep for linear layer
        # Flatten x to vector
        x = x.view(x.size(0), -1)
        print(x.shape)
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        # final output
        return x
