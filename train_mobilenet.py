import argparse
import os
from torchsummary import summary
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
from time import sleep

from tqdm import tqdm
from utils.util import get_config
from lr_scheduler import get_scheduler
from utils.utils_logging import AverageMeter, init_logging
from datasets.wlpuv_dataset import wlpuvDatasets
from losses.wing_loss import WingLoss


def main(config_file):
    now = datetime.datetime.now()
    cfg = get_config(config_file)
    chkFolder = os.path.join('checkpoints', cfg.model_name, now.strftime("%b%d"))
    if not os.path.exists(chkFolder):
        os.makedirs(chkFolder, exist_ok=True)

    world_size = 1
    local_rank = args.local_rank
    torch.cuda.set_device(0)

    print('Initializing dataset...')
    train_set = wlpuvDatasets(cfg)
    cfg.num_images = len(train_set)
    cfg.world_size = world_size

    total_batch_size = cfg.batch_size * cfg.world_size
    epoch_steps = cfg.num_images // total_batch_size
    cfg.warmup_steps = epoch_steps * cfg.warmup_epochs
    if cfg.max_warmup_steps > 0:
        cfg.warmup_steps = min(cfg.max_warmup_steps, cfg.warmup_steps)
    cfg.total_steps = epoch_steps * cfg.num_epochs
    if cfg.lr_epochs is not None:
        cfg.lr_steps = [m * epoch_steps for m in cfg.lr_epochs]
    else:
        cfg.lr_steps = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=cfg.batch_size,  # sampler=train_sampler,
        num_workers=0, pin_memory=False, drop_last=True)

    print('The number of training samples = {0}'.format(cfg.num_images))

    # starting_epoch = cfg.load_checkpoint + 1
    # num_epochs = cfg.max_epochs

    net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=cfg.num_verts * 3).to(local_rank)
    # summary(net, (3, 256, 256))

    net.train()

    if cfg.opt == 'sgd':
        opt = torch.optim.SGD(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.opt == 'adam':
        opt = torch.optim.Adam(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr)
    elif cfg.opt == 'adamw':
        opt = torch.optim.AdamW(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = get_scheduler(opt, cfg)

    start_epoch = 0

    loss = {
        'Loss': AverageMeter(),
    }

    L3 = WingLoss()  # AdaptiveWingLoss()

    global_step = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for step, value in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                global_step += 1
                img = value['Image'].to(local_rank)

                label_verts = value['vertices_filtered'].to(local_rank)
                label_kpt = value['kpt'].to(local_rank)

                # -------- forward --------
                preds = net(img)
                # pred_verts, variance = preds.split([500 * 3, 500 * 1], dim=1)
                pred_verts = preds
                pred_verts = pred_verts.view(cfg.batch_size, cfg.num_verts, 3)
                kpt_filer_index = torch.tensor(np.loadtxt(cfg.filtered_kpt_500).astype(int))
                pred_kpt = pred_verts[:, kpt_filer_index, :]

                loss2 = F.mse_loss(pred_kpt, label_kpt)
                loss3 = L3(pred_verts, label_verts)

                loss = 1.5 * loss3 + 0.5 * loss2

                # -------- backward + optimize --------
                # zero the parameter gradients
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

                running_loss += loss.item() * cfg.batch_size
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

        epoch_loss = running_loss / cfg.num_images
        print('epoch: {0} -> loss: {1} -> running loss: {2}'.format(epoch, loss.item(), epoch_loss))

        save_filename = 'net_%s.pth' % epoch
        save_path = os.path.join(chkFolder, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        net.to(local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)
