import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.to_tensor = transforms.ToTensor()
        # self.files_edge = sorted(glob.glob(root + '/canny_edge/*.png'))
        self.root = root
        # self.files_img = sorted(os.listdir(root + '/images/'))
        self.filepath = os.path.join(self.root, 'bsds_pascal_train_pair.lst')
        with open(self.filepath, 'r') as f:
            self.filelist = f.readlines()

    def __getitem__(self, index):
        img_file, lb_file = self.filelist[index].split()

        # filename = self.files_img[index % len(self.files_img)]
        img_path = self.root + '/' + img_file
        item_img = self.transform(Image.open(img_path).convert("RGB"))

        edge_path = self.root + '/' + lb_file
        item_edge = self.to_tensor(Image.open(edge_path))
        item_edge[item_edge >= 0.2] = 1
        item_edge[item_edge < 0.2] = 0

        return {'img': item_img, 'edge': item_edge[0:1, :, :]}

    def __len__(self):
        return len(self.filelist)
