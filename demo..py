import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, ReLU, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.MNIST("../mnist",train=False,download=True,
                                         transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class POOL(nn.Module):
    def __init__(self):
        super(POOL,self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.maxpool3 = MaxPool2d(2)
        #self.conv4 = Conv2d(in_channels=256, out_channels=300, kernel_size=3)
        self.flat1 = Flatten()
        #self.linear1 = Linear(196608,24)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        #x = self.maxpool2(x)
        #x = self.flat1(self.conv4(x))
        #x = self.linear1(x)
        return x
pool = POOL()



writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = pool(imgs)
    #output = torch.reshape(output, (64, 3, 32, 32))
    output = torch.reshape(output, (-1, 3, 32, 32))
    writer.add_images("output", output, global_step=step)
    step = step+1

writer.close()