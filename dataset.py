# the python code for handling dataset

import os
import os.path as osp

import torch
import torch.utils.data as data
from PIL import Image

# the image extension enumeration
IMG_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
    '.bmp',

    '.JPG',
    '.JPEG',
    '.PNG',
    '.PPM',
    '.BMP',
]

# check if the file is an image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# set the default loader for image, convert the image to RGB
def defalut_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=defalut_loader):
        images = []
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))
        
        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]

        try:
            img = self.loader(osp.join(self.root, filename))
        except:
            return torch.zeros(3,32,32)
        
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.imgs)
    
