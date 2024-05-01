from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#python用法 tensor数据类型
#通过transform.to tensor 解决问题
#使用transform   Tensor数据类型
img_path="E:\\桌面\\个人\\370302200010090513.JPG"
img=Image.open(img_path)

#使用transform



writer=SummaryWriter("logs")


tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
writer.add_image("Tensor_img",tensor_img)

writer.close()