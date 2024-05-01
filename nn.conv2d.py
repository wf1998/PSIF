import torch
from PIL import Image
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("/data",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor)
dataloader = DataLoader(dataset,batch_size=64)

class ss(nn.Module):
    def __init__(self):
        super(ss,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=0)
    def forward(self, x):
        x= self.conv1(x)
        return x
ssy = ss()

writer = SummaryWriter("logs")
step=0
for data in dataloader:
    imgs = data
    output = ss(imgs)
    writer.add_image("imgs",imgs,step)
    writer.add_image("output",output,step)
    step+=1
    writer.close()
