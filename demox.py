import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.MNIST("../mnist",train=False,download=True,
                                         transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)
class POOL(nn.Module):
    def __init__(self):
        super(POOL,self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output
pool = POOL()



writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)

    output = pool(imgs)
   # output = torch.reshape(output, (-1, 64, 32, 32))
    writer.add_images("output",output,global_step=step)
    step = step+1

writer.close()