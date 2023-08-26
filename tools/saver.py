import os
import time

import torch.nn as nn

from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)

base_url2 = None


def normalize_tensor_mm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, nrow=3, save_path=None, exit_flag=False, timestamp=False, norm=False):
    import torchvision.utils as vutils
    if norm:
        tensor = normalize_tensor_mm(tensor)
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=nrow)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            save_path = f'{base_url}/{name}_{mutils.get_timestamp()}.png'
        else:
            save_path = f'{base_url}/{name}.png'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vutils.save_image(grid, save_path)
    if exit_flag:
        exit(0)


def save_feature(tensor, name, exit_flag=False, timestamp=False):
    import torchvision.utils as vutils
    tensors = [tensor, normalize_tensor_mm(tensor), normalize_tensor_sigmoid(tensor)]
    # tensors = [tensor]
    titles = ['original', 'min-max', 'sigmoid']
    os.makedirs(base_url, exist_ok=True)
    if timestamp:
        name += '_' + str(time.time()).replace('.', '')

    for index, tensor in enumerate(tensors):
        _data = tensor.detach().cpu().squeeze(0).unsqueeze(1)
        print(_data.shape)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{base_url}/{name}_{titles[index]}.png')
    if exit_flag:
        exit(0)


def save_text(text, name=None, save_path=None, exit_flag=False, timestamp=False):
    base_dir = base_url2 or base_url

    os.makedirs(base_dir, exist_ok=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w+') as fp:
            fp.write(text)
    else:
        if timestamp:
            with open(f'{base_dir}/{name}_{mutils.get_timestamp()}.txt', 'w+') as fp:
                fp.write(text)
        else:
            with open(f'{base_dir}/{name}.txt', 'w+') as fp:
                fp.write(text)
    if exit_flag:
        exit(0)


def save_points(pSet, name=None, save_path=None, exit_flag=False, timestamp=False):
    base_dir = base_url2 or base_url

    os.makedirs(base_dir, exist_ok=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w+') as fp:
            fp.writelines([f"{p[0]},{p[1]}\n" for p in pSet])
    else:
        if timestamp:
            with open(f'{base_dir}/{name}_{mutils.get_timestamp()}.txt', 'w+') as fp:
                fp.writelines([f"{p[0]},{p[1]}\n" for p in pSet])
        else:
            with open(f'{base_dir}/{name}.txt', 'w+') as fp:
                fp.writelines([f"{p[0]},{p[1]}\n" for p in pSet])
    if exit_flag:
        exit(0)
