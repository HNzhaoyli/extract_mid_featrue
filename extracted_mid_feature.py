import torch
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import json

model = models.resnet50(pretrained=True)
#model = torch.load('model_params(RES_best).pkl')#可以改为model_ft = models.resnet18(pretrained=True)，直接下载ResNet18.
image = Image.open(r'F:\Python  工程\market 1501\Market-1501-v15.09.15\pytorch\query\0001\0001_c1s1_001051_00.jpg')

transform = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

img = transform(image)
print('img.size:',img.size())
img = img.unsqueeze(0)
print(img.size)


def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')# array 转化为Image



new_model = nn.Sequential(*list(model.children())[:5])
#print(new_model)
f3 = new_model(img)
print('L1.SIZE',f3.shape)
save_img(f3, 'layer1')

new_model = nn.Sequential(*list(model.children())[:6])
f4 = new_model(img)  # [1, 128, 28, 28]
print('L2.SIZE',f4.shape)
save_img(f4, 'layer2')

new_model = nn.Sequential(*list(model.children())[:7])
#print(new_model)
f5 = new_model(img)  # [1, 256, 14, 14]
print('layer3.size',f5.shape)
save_img(f5, 'layer3')

new_model = nn.Sequential(*list(model.children())[:8])
#print(new_model)
f6 = new_model(img)  # [1, 256, 14, 14]
print('layer4.size',f6.shape)

save_img(f6, 'layer4')
