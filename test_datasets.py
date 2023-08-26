import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io, color
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        #self.files_edge = sorted(glob.glob(root + '/canny_edge/*.png'))
        self.root = root
        #self.files_img = sorted(os.listdir(root + '/images/'))
        self.filepath = os.path.join(self.root+'/HED-BSDS', 'test.lst')
        with open(self.filepath, 'r') as f:
            self.filelist = f.readlines()

    def __getitem__(self, index):
        img_file = self.filelist[index].rstrip()

        #filename = self.files_img[index % len(self.files_img)] 
        img_path = self.root + '/HED-BSDS/' + img_file
        item_img = self.transform(Image.open(img_path).convert("RGB"))

        img_file_split = img_file.split("/")
        path_name = "/".join(img_file_split[:-1])
        file_name = img_file_split[-1]

        return {'img':item_img, 'path_name':path_name, 'file_name':file_name}

    def __len__(self):
        return len(self.filelist)
