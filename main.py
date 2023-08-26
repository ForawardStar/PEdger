import argparse

import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import *
from loss_function import MyLoss
from models import *
from models_noshare import Guider_noshare
from tools import SingleSummaryWriter, mutils, saver
from tools.metric_utils import AverageMeters, write_loss
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--comment', '-m', default='edge_detection')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--iter_size', type=int, default=16, help='size of the iterations')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=20, help='number of cpu threads to use during batch generation')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
parser.add_argument("--log_interval", type=int, default=500, help="interval for logging")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')


# ----------
#  Training
# ----------
def main():
    global global_step

    for epoch in range(args.epoch, args.n_epochs):
        if epoch >= 2:
            state_st = G_network.state_dict()
            state_t = G_network_teacher.state_dict()
            for k, v in state_t.items():
                state_t[k] = (state_t[k] + state_st[k]) * 0.5
            G_network_teacher.load_state_dict(state_t)

            state_st_noshare = G_network_noshare.state_dict()
            state_t_noshare = G_network_teacher_noshare.state_dict()
            for k, v in state_t_noshare.items():
                state_t_noshare[k] = (state_t_noshare[k] + state_st_noshare[k]) * 0.5
            G_network_teacher_noshare.load_state_dict(state_t_noshare)
        elif epoch == 1:
            G_network_teacher.load_state_dict(G_network.state_dict())

            G_network_teacher_noshare.load_state_dict(G_network_noshare.state_dict())

        dis_weight = 0.8 * float(epoch) / float(args.n_epochs)

        loss_meter = AverageMeters()
        loss_noshare_meter = AverageMeters()
        bar = tqdm.tqdm(dataloader, disable=True)
        saver.base_url = os.path.join(args.saved_path, 'results')

        for i, batch in enumerate(bar):
            # if args.debug and i > 2000:
            #     break

            # Set model input
            img = batch['img'].float().to(device)
            edge_gt = batch['edge'].float().to(device)

            if epoch >= 1:
                with torch.no_grad():
                    h, w = img.shape[2], img.shape[3]

                    mask_features_teacher    = G_network_teacher(img)[-1]
                    mask_features_noshare    = G_network_teacher_noshare(img)[-1]
                 
                    uncertainty = torch.abs(F.sigmoid(mask_features_teacher) - 0.5).detach()
                    uncertainty_noshare = torch.abs(F.sigmoid(mask_features_noshare) - 0.5).detach()

                    weight = uncertainty / (uncertainty + uncertainty_noshare)

                    res = F.sigmoid(mask_features_teacher * weight + mask_features_noshare * (1 - weight))
                    
                    edge_gt_soft = edge_gt * (1 - dis_weight) + res * dis_weight
            else:
                edge_gt_soft = edge_gt

            if random.random() < dis_weight:
                img_smoothed = bilateralFilter(img, 5)
                img = img_smoothed if random.random() > 0.5 else img + 2 * (img - img_smoothed)

            edge_feats = G_network(img)
            edge_preds = [torch.sigmoid(r) for r in edge_feats]

            # Identity loss
            loss, loss_items = criterion(edge_preds, edge_gt, edge_gt_soft)

            if torch.isnan(loss):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss = loss / args.iter_size
            loss.backward()

            edge_feats_noshare = G_network_noshare(img)
            edge_preds_noshare = [torch.sigmoid(r) for r in edge_feats_noshare]

            # Identity loss
            loss_noshare, loss_noshare_items = criterion(edge_preds_noshare, edge_gt, edge_gt_soft)

            if torch.isnan(loss_noshare):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss_noshare = loss_noshare / args.iter_size
            loss_noshare.backward()

            if (i + 1) % args.iter_size == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

                optimizer_G_noshare.step()
                optimizer_G_noshare.zero_grad()

            loss_meter.update(loss_items)
            loss_noshare_meter.update(loss_noshare_items)

            if global_step % args.log_interval == 0:
                print('\r[Epoch %d/%d, Iter: %d/%d]: %s, %s' % (epoch, args.n_epochs, i, len(bar), loss_meter, loss_noshare_meter), end="")
                write_loss(writer, 'train', loss_meter, global_step)

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    show = torch.cat([*edge_preds, edge_gt], dim=0).repeat(1, 3, 1, 1)
                    show = torch.cat([show, img], dim=0)
                    saver.save_image(show, '%09d' % global_step, nrow=5)

            global_step += 1

            del loss, loss_noshare, img, edge_preds, edge_preds_noshare, edge_feats, edge_feats_noshare

        loss_meter.reset()
        loss_noshare_meter.reset()
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            save_checkpoint({'G': G_network, 'G_teacher': G_network_teacher, 'G_noshare': G_network_noshare, 'G_teacher_noshare': G_network_teacher_noshare},
                            {'optimizer': optimizer_G, 'optimizer_noshare': optimizer_G_noshare},
                            {'scheduler': scheduler_cosine, 'scheduler_warmup': scheduler_warmup, 'scheduler_noshare': scheduler_cosine_noshare, 'scheduler_warmup_noshare': scheduler_warmup_noshare},
                            'ckt', epoch, os.path.join(args.saved_path, 'weights'))

        scheduler_warmup.step()
        scheduler_warmup_noshare.step()


if __name__ == '__main__':
    args = parser.parse_args()

    # setting random seed
    seed = 5603114
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda'

    # Losses
    criterion = MyLoss().to(device)

    # Initialize student and teacher
    G_network = Guider_stu().to(device)
    G_network_teacher = Guider_stu().to(device)
    G_network_noshare = Guider_noshare().to(device)
    G_network_teacher_noshare = Guider_noshare().to(device)

    for p in G_network_teacher.parameters():
        p.requires_grad = False
    for p in G_network_teacher_noshare.parameters():
        p.requires_grad = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # Image transformations
    transforms_ = [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                   transforms.RandomGrayscale(p=0.2),                   
                   transforms.ToTensor(), normalize]

    # Training data loader
    dataloader = DataLoader(ImageDataset("/home/fyb", transforms_=transforms_, unaligned=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    # Testing data loader
    val_dataloader = DataLoader(ImageDataset("/home/fyb", transforms_=transforms_, unaligned=True, mode='test'),
                                batch_size=1, shuffle=False, num_workers=1)

    # Defining optimizer and schedulers

    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine = CosineAnnealingLR(optimizer_G, args.n_epochs)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer_G, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine)

    optimizer_G_noshare = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network_noshare.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine_noshare = CosineAnnealingLR(optimizer_G_noshare, args.n_epochs)
    scheduler_warmup_noshare = GradualWarmupScheduler(
        optimizer_G_noshare, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine_noshare)

    # Defining logging dirs
    timestamp = mutils.get_formatted_time()
    args.saved_path = args.saved_path + f'/{args.comment}/{timestamp}'
    args.log_path = args.log_path + f'/{args.comment}/{timestamp}/tensorboard/'

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.saved_path, exist_ok=True)

    writer = SingleSummaryWriter(args.log_path)
    global_step = 0

    if args.resume is not None:
        state_dict = torch.load(args.resume)
        args.epoch = state_dict['epoch'] + 1
        G_network.load_state_dict(state_dict['G'])
        G_network_teacher.load_state_dict(state_dict['G_teacher'])
        G_network_noshare.load_state_dict(state_dict['G_noshare'])
        G_network_teacher_noshare.load_state_dict(state_dict['G_teacher_noshare'])
        optimizer_G.load_state_dict(state_dict['optimizer'])
        optimizer_G_noshare.load_state_dict(state_dict['optimizer_noshare'])
        scheduler_cosine.load_state_dict(state_dict['scheduler'])
        scheduler_warmup.load_state_dict(state_dict['scheduler_warmup'])
        scheduler_cosine_noshare.load_state_dict(state_dict['scheduler_noshare'])
        scheduler_warmup_noshare.load_state_dict(state_dict['scheduler_warmup_noshare'])

    main()
