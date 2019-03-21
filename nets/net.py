import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape, input_channels, output_size):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense1 = nn.Linear(self.get_flat_size(input_shape, input_channels), 512)
        self.dense2 = nn.Linear(512, output_size)

    def get_flat_size(self, inp, chan):
        t = torch.zeros(1, chan, inp[0], inp[1])
        t = self.conv1(t)
        t = self.conv2(t)
        t = self.conv3(t)
        return t.view(-1).shape[0]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.dense1(x))
        return self.dense2(x)
