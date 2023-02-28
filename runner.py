import torch
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

import os
from tqdm import tqdm

from model import AdaDecoder, SingleImageEncoder
from dataset import load_dataset


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

    # loss function
    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
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
    os.makedirs(config.event_directory, exist_ok=True)
    summary_writer = SummaryWriter(config.event_directory)

    last_epoch = -1
    global_step = 0
    for epoch in range(last_epoch + 1, config.num_epochs):

        encoder.train()
        decoder.train()

        with tqdm(enumerate(dataloader)) as tqdmLoader:
            for local_step, (positions, occupancies, images) in tqdmLoader:
                positions = positions.cuda()
                occupancies = occupancies.cuda()
                images = images.cuda()

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

    summary_writer.close()
            


def eval(config):
    pass
