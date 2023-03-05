import torch
from torch import nn
from torch import optim
import numpy as np

from tensorboardX import SummaryWriter

import os
import glob
from tqdm import tqdm

from model import AdaDecoder, SingleImageEncoder
from dataset import load_dataset
from metric import Metric


# convert dict to var
class Dict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


def train(config):
    # device
    device = torch.device(config.device)
    # dataset
    train_data, train_loader = load_dataset(config, mode="train")
    valid_data, valid_loader = load_dataset(config, mode="val")
    print(f"train {len(train_data)} valid {len(valid_data)}")

    # model
    encoder = SingleImageEncoder().to(device)
    decoder = AdaDecoder(
        embed_param=Dict(
            in_channels=3,
            out_channels=256,
            kernel_size=1
        ),
        cond_params=[
            Dict(
                in_channels=256,
                out_channels=256,
                cond_channels=256
            ) for _ in range(5)
        ],
        conv_param=Dict(
            in_channels=256,
            out_channels=1,
            kernel_size=1
        )
    ).to(device)

    # loss function
    criterion = nn.BCELoss(reduction='mean').to(device)
    # optimizer
    optimizer = optim.Adam([
        dict(
            params=encoder.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2
        ),
        dict(
            params=decoder.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2
        )
    ])

    # summary
    dirs = glob.glob(config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(f"{config.log_dir}/{len(dirs)}")

    # process
    last_epoch = -1
    global_step = 0

    # start training
    for epoch in range(last_epoch+1, config.num_epochs):

        encoder.train()
        decoder.train()

        avg_loss = []
        for positions, occupancies, images in train_loader:
            optimizer.zero_grad()
            
            # transport data
            positions = positions.to(device)
            occupancies = occupancies.to(device)
            images = images.to(device)

            # auto encoder
            conditions = encoder(images)
            logits = decoder(positions, conditions)

            # backward
            loss = criterion(logits, occupancies)
            loss.backward()
            optimizer.step()

            # metric
            out = (logits > config.threshold).int()
            gt = occupancies.int()
            metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])

            # write to tensorboard
            iou = metric_outs['iou']
            prec, reca = metric_outs['pr']
            summary_writer.add_scalars('loss', dict(train_loss=loss), global_step)
            summary_writer.add_scalars('iou', dict(train_iou=iou), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=prec, train_reca=reca), global_step)
            
            # record
            avg_loss.append(loss.item())
            global_step += 1

        if (epoch + 1) % config.test_time == 0:
            encoder.eval()
            decoder.eval()
            valid_loss = []
            valid_iou = []
            valid_prec, valid_reca = [], []

            with torch.no_grad():
                for positions, occupancies, images in tqdm(valid_loader):

                    # transport data
                    positions = positions.to(device)
                    occupancies = occupancies.to(device)
                    images = images.to(device)

                    # forward
                    conditions = encoder(images)
                    logits = decoder(positions, conditions)

                    # metric
                    loss = criterion(logits, occupancies)
                    out = (logits > config.threshold).int()
                    gt = occupancies.int()
                    metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])

                    # record
                    iou = metric_outs['iou']
                    prec, reca = metric_outs['pr']
                    valid_loss.append(loss.item())
                    valid_iou.append(iou)
                    valid_prec.append(prec)
                    valid_reca.append(reca)
                
            summary_writer.add_scalars('loss', dict(valid_loss=np.mean(valid_loss)), global_step)
            summary_writer.add_scalars('iou', dict(valid_iou=np.mean(valid_iou)), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=np.mean(valid_prec), train_reca=np.mean(valid_reca)), global_step)

            avg_loss_ = np.mean(avg_loss)
            tqdm.write(f"Average Loss During Last {config.test_time: d} Epoch is {avg_loss_: .6f}")

            os.makedirs(config.ckpt_dir, exist_ok=True)
            torch.save(encoder.state_dict(), f"{config.ckpt_dir}/encoder_{epoch + 1: d}_.pt")
            torch.save(decoder.state_dict(), f"{config.ckpt_dir}/decoder_{epoch + 1: d}_.pt")

    summary_writer.close()

    os.makedirs(config.ckpt_dir, exist_ok=True)
    torch.save(encoder.state_dict(), f"{config.ckpt_dir}/encoder_{config.num_epochs: d}_.pt")
    torch.save(decoder.state_dict(), f"{config.ckpt_dir}/decoder_{config.num_epochs: d}_.pt")  


def eval(config):
    # device
    device = torch.device(config.device)
    # dataset
    test_data, test_loader = load_dataset(config, mode='test')

    # model
    encoder = SingleImageEncoder().to(device)
    decoder = AdaDecoder(
        embed_param=Dict(
            in_channels=3,
            out_channels=256,
            kernel_size=1
        ),
        cond_params=[
            Dict(
                in_channels=256,
                out_channels=256,
                cond_channels=256
            ) for _ in range(5)
        ],
        conv_param=Dict(
            in_channels=256,
            out_channels=1,
            kernel_size=1
        )
    ).to(device)

    # load checkpoints
    encoder.load_state_dict(torch.load(f"{config.ckpt_dir}/encoder_{config.ckpt_name}_.pt"))
    decoder.load_state_dict(torch.load(f"{config.ckpt_dir}/decoder_{config.ckpt_name}_.pt"))

    # record
    avg_iou = []
    avg_prec = []
    avg_reca = []

    # start evaluating
    encoder.eval()
    decoder.eval()
    for positions, occupancies, images in tqdm(test_loader):

        # data transport
        positions = positions.to(device)
        occupancies = occupancies.to(device)
        images = images.to(device)

        # forward
        conditions = encoder(images)
        logits = decoder(positions, conditions)

        # metric
        out = (logits > config.threshold).int()
        gt = occupancies.int()
        metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])

        # record
        iou = metric_outs['iou']
        prec, reca = metric_outs['pr']
        avg_iou.append(iou)
        avg_prec.append(prec)
        avg_reca.append(reca)

    print(f"test iou {np.mean(avg_iou): .2f}")
    print(f"test precision {np.mean(avg_prec): .2f}")
    print(f"test recall {np.mean(avg_reca): .2f}")
