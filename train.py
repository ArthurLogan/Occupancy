import argparse
import json
import torch

from runner import train, eval, Dict


def parse():
    parser = argparse.ArgumentParser("Occupancy Networks")
    parser.add_argument("--state", type=str, default="train", choices=["train", "eval"],
                        help="train or eval the network")

    # configuration
    parser.add_argument("--config", type=str, default="config/default.json", help="path to config file")

    # directory configuration
    parser.add_argument("--ckpt", type=str, default="", help="ckpt name for use")

    # device configuration
    parser.add_argument("--device", type=str, default="cuda:0", help="use cuda or cpu")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # config and args
    args = parse()
    with open(args.config) as file:
        config = Dict(json.load(file))
    
    config.update(vars(args))

    # init
    torch.manual_seed(0)

    if config["state"] == "train":
        train(config)
    elif config["state"] == "eval":
        eval(config)
