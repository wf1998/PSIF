import numpy as np
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
#构造矩阵
x=torch.randn(4,4)
print(x.size())
#与numpy协作
a=torch.ones(5)
b=a.numpy()
print(b)
from PIL import Image
img_path="E:\\桌面\\个人\\370302200010090513.JPG"
img=Image.open(img_path)
img.show()