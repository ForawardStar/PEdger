import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss_RCF(prediction, label, label_soft):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((label * label_soft).float()).float()
    num_negative = torch.sum(((1 - label) * (1 - label_soft)).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = F.binary_cross_entropy(prediction.float(), label_soft.float(), weight=mask, reduction='none')
    return torch.mean(cost)


#
# def cross_division(prediction, label, label_soft):
#     # number_positive = torch.sum((label * label_soft).float()).float()
#     # number_negative = torch.sum(((1 - label) * (1 - label_soft)).float()).float()
#     # weight_positive = number_negative / (number_positive + number_negative)
#     # weight_negative = number_positive / (number_positive + number_negative)
#     ga = 0.8
#     positive_samples = torch.abs(prediction) / torch.clamp_min(torch.abs(label_soft), 0.15)
#     negative_samples = torch.abs(1 - prediction) / torch.clamp_min(torch.abs(1 - label_soft), 0.15)
#
#     loss = torch.mean(positive_samples * (1 - ga) + negative_samples * ga)
#     return loss


def cross_division(prediction, label, label_soft):
    number_positive = torch.sum((label * label_soft).float()).float()
    number_negative = torch.sum(((1 - label) * (1 - label_soft)).float()).float()
    weight_positive = number_positive / (number_positive + number_negative)
    weight_negative = number_negative / (number_positive + number_negative)

    eps = 0.2
    ga = 0.8

    positive_samples = prediction / (label_soft + eps)
    negative_samples = (1 - prediction) / (1 - label_soft + eps)

    loss = torch.mean(positive_samples * (1 - ga) + negative_samples * ga)
    return loss


def cross_division2(prediction, label, label_soft):
    number_positive = torch.sum((label * label_soft).float()).float()
    number_negative = torch.sum(((1 - label) * (1 - label_soft)).float()).float()
    weight_positive = number_negative / (number_positive + number_negative)
    weight_negative = number_positive / (number_positive + number_negative)

    # ga = 0.8
    eps = 1.
    positive_samples = label_soft / (prediction + eps)
    negative_samples = (1 - label_soft) / (1 - prediction + eps)

    loss = torch.mean(positive_samples * weight_positive + negative_samples * weight_negative)
    return loss

# def merge_ce_cd(prediction, label_soft):
#     prediction-label_soft
#
# class MyLoss(nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#
#     def forward(self, edge_preds, edge_gt, edge_gt_soft):
#         bcd_losses = torch.stack([cross_division(ep, edge_gt, edge_gt_soft) for ep in edge_preds])
#         bcd_sum = torch.sum(bcd_losses)
#
#         losses = {
#             'bcd': bcd_sum,
#         }
#
#         return bcd_sum, losses
# #
# class MyLoss(nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#
#     def forward(self, edge_preds, edge_gt, edge_gt_soft):
#         bcd_losses = torch.stack([cross_division(ep, edge_gt, edge_gt_soft) for ep in edge_preds])
#         bcd_sum = torch.sum(bcd_losses)
#
#         losses = {
#             'bcd': bcd_sum,
#         }
#
#         return bcd_sum, losses

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, edge_preds, edge_gt, edge_gt_soft):
        bce_losses = torch.stack([cross_entropy_loss_RCF(ep, edge_gt, edge_gt_soft) for ep in edge_preds])
        bce_sum = torch.sum(bce_losses)

        losses = {
            'bce': bce_sum,
        }

        return bce_sum, losses
