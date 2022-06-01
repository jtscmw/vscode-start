import os
import torch
from network.net import Net

def main():
    x = torch.randn([3, 224, 224])
    model = Net()
    out = model(x)
    print(out)

if __name__=='__main__':
    main()