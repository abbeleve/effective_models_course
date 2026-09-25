import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=7//2, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=5//2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=3//2,  bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3//2, bias=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 100))
    
    def forward(self, x):
        x = self.maxpool1(self.act1(self.conv1(x)))
        x = self.act1(self.conv2(x))
        x = self.act1(self.conv3(x))
        x = self.act1(self.conv4(x))
        x = self.act1(self.conv5(x))
        x = self.act1(self.conv6(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

