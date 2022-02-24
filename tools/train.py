#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger
import time
import gc

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_nccl,
    configure_omp,
    get_num_devices,
    build_settings_from_config,
    build_config,
)

import wandb


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        default=list(range(get_num_devices())),
        nargs="+",
        type=int,
        help="List of gpus to use in rank order",
    )
    parser.add_argument(
        "--sweepid",
        help="Provide sweep id to continue previous sweep.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--noimg", help="Turn off image logging", default=False, action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main(exp, args, wandb_instance=None):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()
    wandb.finish()  # could add to trainer instead
    time.sleep(5)

    del trainer
    gc.collect()
    time.sleep(5)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    assert not any(
        [x >= get_num_devices() for x in args.gpus]
    ), f"invalid gpu device provided in {args.gpus}"

    dist_url = "auto" if args.dist_url is None else args.dist_url

    if exp.perform_hpo == False:
        config = build_config(exp, args, perform_hpo=False)
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )
    else:
        # construct a sweeper
        config = build_config(exp, args, perform_hpo=True)

        sweep_config = {
            "method": "bayes",
            "metric": {"name": "COCOAP50_95", "goal": "maximize"},
            "parameters": config,
        }
        if args.sweepid is None:
            try:
                with open("./sweep_status.txt", "r") as f:
                    lines = f.readlines()
                    f.close()
                _, runs, sweep_id, expn = [x.strip() for x in lines][0].split(" ")
                # TODO: check that expn matches the experiment name; if exp name was provided.

            except IndexError:  # file is empty
                sweep_id = wandb.sweep(sweep_config, project=args.experiment_name)
        else:
            sweep_id = args.sweepid

        def run():
            global exp
            global args
            wandb.init()
            run_config = wandb.config
            exp, args = build_settings_from_config(run_config, exp, args)

            launch(
                main,
                num_gpu,
                args.num_machines,
                args.machine_rank,
                backend=args.dist_backend,
                dist_url=dist_url,
                args=(exp, args),
            )
            time.sleep(5)
            # after first hpo run, save sweep to file.
            # in subsequent runs, decrease run count and check if depleted.
            try:
                with open("./sweep_status.txt", "r") as f:
                    lines = f.readlines()
                    f.close()
                _, runs, curr, _ = [x.strip() for x in lines][0].split(" ")
                runs = int(runs)
                assert (
                    curr == sweep_id
                ), "Error in sweep_status.txt file. Id of sweep being tracked does not match current sweep id."
                with open("../sweep_status.txt", "w") as f:
                    print("Run completed. Changing sweep status file.")
                    if runs == 1:
                        print("all runs completed. removing entry")
                        f.close()  # remove all entries.
                    else:
                        # TODO: find way to read no. of runs from agent?
                        print(f"{runs - 1 } runs left for this sweep")
                        f.write(f"sweep {runs - 1} {sweep_id} {args.experiment_name}")
                        f.close()
            except IndexError:
                # there is no on-going sweep; first run
                with open("./sweep_status.txt", "w") as f:
                    f.write(f"sweep {50} {sweep_id} {args.experiment_name}")

        wandb.agent(
            sweep_id, function=run, count=50, project=args.experiment_name
        )  # TODO turn count into param; currently hard coded to 50.

