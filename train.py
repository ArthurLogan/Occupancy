import argparse
import json
import torch

from runner import train, eval, Dict


def parse_args_and_config():
    parser = argparse.ArgumentParser("Occupancy Networks")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="train or eval the network")

    # configuration
    parser.add_argument("--config", type=str, default="config/default.json", help="path to config file")

    # directory configuration
    parser.add_argument("--ckpt", type=str, default="", help="ckpt name for use")

    # device configuration
    parser.add_argument("--device", type=str, default="cuda:0", help="use cuda or cpu")

    args = parser.parse_args()

    with open(args.config) as file:
        config = Dict(json.load(file))
    
    config.update(vars(args))

    return config


if __name__ == "__main__":
    # config and args
    config = parse_args_and_config()

    # init
    torch.manual_seed(0)

    if config.mode == "train":
        train(config)
    elif config.mode == "eval":
        eval(config)
