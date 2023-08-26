import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from models_noshare import Guider_noshare
from test_datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from loss_function import *

def get_model_parm_nums(model): 
    total = sum([param.numel() for param in model.parameters()]) 
    total = float(total) / 1024 
    return total 

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize/logs/edge_detection/20230223-231152/weights/ckt_0014.pth')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion = MyLoss()#.cuda()

cuda = True if torch.cuda.is_available() else False

G_network = Guider_stu()
G_network_noshare = Guider_noshare()

if cuda:
    G_network = G_network.cuda()
    G_network_noshare = G_network_noshare.cuda()

# G_network.eval()
# Load pretrained models
if opt.ckpt is not None:
    state_dict = torch.load(opt.ckpt)

    #G_network_state_dict = state_dict["G_teacher"]
    #G_network.load_state_dict(G_network_state_dict)

    #G_network_noshare_state_dict = state_dict["G_teacher_noshare"]
    #G_network_noshare.load_state_dict(G_network_noshare_state_dict)
    G_network_state_dict = state_dict["G_teacher"]
    G_network.load_state_dict(G_network_state_dict)

    G_network_noshare_state_dict = state_dict["G_teacher_noshare"]
    G_network_noshare.load_state_dict(G_network_noshare_state_dict)

total_params = get_model_parm_nums(G_network)
print("*****************************")
print("total_params:  ", total_params)
print("*****************************")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
# Image transformations
transforms_ = [transforms.ToTensor(), normalize]

# Training data loader
dataloader = DataLoader(ImageDataset("/home/fyb", transforms_=transforms_, unaligned=True),
                        batch_size=1, shuffle=True, num_workers=1)
# ----------
#  Training
# ---------
prev_time = time.time()

edge_path_formal = "baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_ensemble_" + opt.ckpt.strip().strip('/').split('/')[-1]
edge_path_formal1 = "baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_" + opt.ckpt.strip().strip('/').split('/')[-1]
edge_path_formal2 = "baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_noshare_" + opt.ckpt.strip().strip('/').split('/')[-1]
print(edge_path_formal)
os.makedirs(edge_path_formal, exist_ok=True)
os.makedirs(edge_path_formal1, exist_ok=True)
os.makedirs(edge_path_formal2, exist_ok=True)
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        #input_edge = Variable(batch['edge'].type(Tensor))
        path_name = batch['path_name'][0]
        file_name = batch['file_name'][0]
        input_image = Variable(batch['img'].type(Tensor))
        _input_image = input_image.clone()
        
        with torch.no_grad():
            h, w = input_image.shape[2], input_image.shape[3]
            input_image_05 = F.interpolate(input_image, size=(int(round(h*0.5, 0)), int(round(w*0.5, 0))), mode='bilinear')
            input_image_15 = F.interpolate(input_image, size=(int(round(h*1.5, 0)), int(round(w*1.5, 0))), mode='bilinear')

            mask_features_05 = F.interpolate(G_network(input_image_05)[-1], size=(h, w), mode='bilinear')
            mask_features    = G_network(input_image)[-1]
            mask_features_15 = F.interpolate(G_network(input_image_15)[-1], size=(h, w), mode='bilinear')

            mask_features_noshare_05 = F.interpolate(G_network_noshare(input_image_05)[-1], size=(h, w), mode='bilinear')
            mask_features_noshare    = G_network_noshare(input_image)[-1]
            mask_features_noshare_15 = F.interpolate(G_network_noshare(input_image_15)[-1], size=(h, w), mode='bilinear')

            uncertainty_05 = torch.abs(F.sigmoid(mask_features_05) - 0.5).detach()
            uncertainty_noshare_05 = torch.abs(F.sigmoid(mask_features_noshare_05) - 0.5).detach()
            uncertainty = torch.abs(F.sigmoid(mask_features) - 0.5).detach()
            uncertainty_noshare = torch.abs(F.sigmoid(mask_features_noshare) - 0.5).detach()
            uncertainty_15 = torch.abs(F.sigmoid(mask_features_15) - 0.5).detach()
            uncertainty_noshare_15 = torch.abs(F.sigmoid(mask_features_noshare_15) - 0.5).detach()

            weight_05 = uncertainty_05 / (uncertainty_05 + uncertainty_noshare_05)
            weight = uncertainty / (uncertainty + uncertainty_noshare)
            weight_15 = uncertainty_15 / (uncertainty_15 + uncertainty_noshare_15)

            fusion_05 = (mask_features_05 * weight_05 + mask_features_noshare_05 * (1 - weight_05))
            fusion = (mask_features * weight + mask_features_noshare * (1 - weight))
            fusion_15 = (mask_features_15 * weight_15 + mask_features_noshare_15 * (1 - weight_15))

            res_05 = torch.exp(fusion_05.detach() - 0.5) / (torch.exp(fusion_05.detach() - 0.5) + torch.exp(0.5 - fusion_05.detach()))
            res    = torch.exp(fusion.detach() - 0.5) / (torch.exp(fusion.detach() - 0.5) + torch.exp(0.5 - fusion.detach()))
            res_15 = torch.exp(fusion_15.detach() - 0.5) / (torch.exp(fusion_15.detach() - 0.5) + torch.exp(0.5 - fusion_15.detach()))

            res = (res_05 + res + res_15) / 3.0
            res1 = torch.exp(mask_features.detach() - 0.5) / (torch.exp(mask_features.detach() - 0.5) + torch.exp(0.5 - mask_features.detach()))
            res2 = torch.exp(mask_features_noshare.detach() - 0.5) / (torch.exp(mask_features_noshare.detach() - 0.5) + torch.exp(0.5 - mask_features_noshare.detach()))

        # print("head.norm.running_mean[0] = ", G_network.state_dict()["head.norm.running_mean"][0].item(), end=' ')
        #outputs = [torch.sigmoid(r) for r in outputs]

        #res = torch.exp(mask_features.detach() - 0.5) / (torch.exp(mask_features.detach() - 0.5) + torch.exp(0.5 - mask_features.detach()))
        print(", image_size = ", res.shape)
        save_image(res, edge_path_formal + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        save_image(res1, edge_path_formal1 + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)
        save_image(res2, edge_path_formal2 + "/" + file_name.split(".")[0] + ".png", nrow=1,
                   normalize=False)

        # --------------
        #  Log Progress
        # --------------
        del input_image, path_name, file_name
