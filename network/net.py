from turtle import forward
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=0, stride=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool(x)
        return x

