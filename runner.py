import torch
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

import os
import glob
from tqdm import tqdm

from model import AdaDecoder, SingleImageEncoder
from dataset import load_dataset
from metric import IoU


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
    train_dataset, train_loader = load_dataset(config, mode="train")
    valid_dataset, valid_loader = load_dataset(config, mode="val")

    print(len(train_dataset), len(valid_dataset))

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
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
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
    exist_num = len(glob.glob(config.log_dir))
    log_dir = os.path.join(config.log_dir, f"{exist_num}")
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)

    last_epoch = -1
    global_step = 0
    for epoch in range(last_epoch + 1, config.num_epochs):

        encoder.train()
        decoder.train()

        with tqdm(enumerate(train_loader)) as tqdmLoader:
            for local_step, (positions, occupancies, images) in tqdmLoader:
                positions = positions.to(device)
                occupancies = occupancies.to(device)
                images = images.to(device)

                conditions = encoder(images)
                logits = decoder(positions, conditions)
                loss = criterion(logits, occupancies)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                summary_writer.add_scalars(
                    main_tag='loss',
                    tag_scalar_dict=dict(training=loss),
                    global_step=global_step
                )

                tqdmLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": local_step,
                    "loss": loss
                })

                global_step += 1

                if global_step % config.test_time == 0:
                    positions, occupancies, images = next(iter(valid_loader))
                    positions = positions.to(device)
                    occupancies = occupancies.to(device)
                    images = images.to(device)

                    conditions = encoder(images)
                    logits = decoder(positions, conditions)
                    loss = criterion(logits, occupancies)

                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(validating=loss),
                        global_step=global_step
                    )

    summary_writer.close()

    os.makedirs(config.ckpt_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(config.ckpt_dir, f"encoder_ckpt_{global_step}_.pt"))
    torch.save(decoder.state_dict(), os.path.join(config.ckpt_dir, f"decoder_ckpt_{global_step}_.pt"))   


def eval(config):
    # device
    device = torch.device(config.device)
    # dataset
    dataset, dataloader = load_dataset(config)

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

    encoder.load_state_dict(torch.load(os.path.join(config.ckpt_dir, f"encoder_ckpt_{config.ckpt_name}_.pt")))
    decoder.load_state_dict(torch.load(os.path.join(config.ckpt_dir, f"decoder_ckpt_{config.ckpt_name}_.pt")))

    encoder.eval()
    decoder.eval()
    avg_iou = 0
    total_samples = 0
    with tqdm(enumerate(dataloader)) as tqdmLoader:
        for local_step, (positions, occupancies, images) in tqdmLoader:
            
            positions = positions.to(device)
            occupancies = occupancies.to(device)
            images = images.to(device)

            conditions = encoder(images)
            logits = decoder(positions, conditions)
            logits = (logits > config.threshold).float()

            avg_iou += IoU(logits, occupancies).sum()
            total_samples += logits.shape[0]

    print(f"[IoU] {avg_iou / total_samples: .2f}")
