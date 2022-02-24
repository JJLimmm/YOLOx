#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


MODEL_SETTINGS = {
    "nano": {"depth": 0.33, "width": 0.50, "depthwise": True},  # 0.9m params
    "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},  # 5m params
    "small": {"depth": 0.33, "width": 0.50, "depthwise": False},  # 9m params
    "medium": {"depth": 0.67, "width": 0.75, "depthwise": False},  # 25m params
    "large": {"depth": 1.0, "width": 1.0, "depthwise": False},  # 54m params
    "x": {"depth": 1.33, "width": 1.25, "depthwise": False},  # 99m params
}

# default darknet53 uses large width and has 64m params
# darknet21 with large width has 38m params


class Exp(BaseExp):
    def __init__(self, classes=[]):
        super().__init__()

        # ---------------- model config ---------------- #
        self.classes = classes  # only training not possible if classes not provided
        self.num_classes = 80 if len(self.classes) == 0 else len(self.classes)
        self.model_type = None
        self.depth = 1.00
        self.width = 1.00
        self.act = "silu"
        self.is_rknn_model = True

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.eval_data_dir = None

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.perform_hpo = False
        self.debug_mode = False
        self.hpo_data_fraction = 0.1

        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.absolute_base_lr = None
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.ema_decay = 0.9998  # issue #735 suggests 0.9999

        self.lr_reduce_cooldown = 10
        self.lr_reduce_patience = 5
        self.lr_reduce_factor = 0.75
        self.lr_reduce_sensitivity = 5e-5
        self.min_base_lr = 1e-5

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.base_momentum = 0.50
        self.momentum_warmup = 5
        self.print_interval = 50
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # ----------------- wandb Config  --------------------  #
        self.hyperparameters = [
            # removing augmentation based hparams temporarily:
            # "multiscale_range",
            # "mosaic_prob",
            # "mixup_prob",
            # "hsv_prob",
            # "flip_prob",
            # "degrees",
            # "translate",
            # "mosaic_scale",
            # "mixup_scale",
            # "shear",
            # "enable_mixup",
            "warmup_epochs",
            "max_epoch",
            # "warmup_lr",
            "basic_lr_per_img",
            # "scheduler", # TODO modify to feed enough variables for scheduler
            "no_aug_epochs",
            "min_lr_ratio",
            # "ema",
            "ema_decay",
            # "weight_decay",
            "momentum",
            "base_momentum",
            "momentum_warmup",
            # "nmsthre",
            # "random_size", # replaces multiscale_range
            # "scale",
            "lr_reduce_cooldown",
            "lr_reduce_patience",
            "lr_reduce_factor",
            "lr_reduce_sensitivity",
        ]

    def get_hpo_custom_params(self) -> dict:
        custom_params = {
            "act": {"distribution": "categorical", "values": ["relu", "silu"],},
            "ema_decay": {"distribution": "uniform", "min": 0.9, "max": 1.0},
            "model_type": {
                "distribution": "categorical",
                "values": ["tiny", "small", "medium"],
            },
            "input_size": {
                "distribution": "categorical",
                "values": [(512, 512), (640, 640)],
            },
            "random_size": {
                "distribution": "categorical",
                "values": [(10, 20), (5, 25), (5, 15)],
            },
            "scale": {
                "distribution": "categorical",
                "values": [(0.2, 1.7), (0.5, 1.5), (0.7, 1.3), (0.1, 2.0)],
            },
            "ema": {"distribution": "categorical", "values": [True, False]},
            "mosaic_scale": {
                "distribution": "categorical",
                "values": [(0.2, 1.7), (0.5, 1.5), (0.7, 1.3), (0.1, 2.0)],
            },
            "mixup_scale": {
                "distribution": "categorical",
                "values": [(0.2, 1.7), (0.5, 1.5), (0.7, 1.3), (0.1, 2.0)],
            },
            "max_epoch": {
                "distribution": "q_normal",
                "mu": self.max_epoch,
                "sigma": 50,
                "q": 1,
            },
            "scheduler": {
                "distribution": "categorical",
                "values": ["yoloxwarmcos", "warmcos", "yoloxsemiwarmcos"],
            },
            "lr_reduce_factor": {"distribution": "uniform", "min": 0.1, "max": 0.9},
            "base_momentum": {
                "distribution": "uniform",
                "min": 0.1,
                "max": self.momentum,
            },
            "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.999},
            "basic_lr_per_img": {
                "distribution": "uniform",
                "min": self.min_lr_ratio * self.basic_lr_per_img,
                "max": self.basic_lr_per_img,
            },
        }

        return custom_params

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.models.network_blocks import RKFocus

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            if self.model_type is not None:
                settings = MODEL_SETTINGS[self.model_type]
                depth, width, use_depthwise = (
                    settings["depth"],
                    settings["width"],
                    settings["depthwise"],
                )
            else:
                depth, width, use_depthwise = (
                    self.depth,
                    self.width,
                    False,
                )  # Exclude Nano by default.

            backbone = YOLOPAFPN(
                depth,
                width,
                in_channels=in_channels,
                act=self.act,
                depthwise=use_depthwise,
            )
            head = YOLOXHead(
                self.num_classes,
                width,
                in_channels=in_channels,
                act=self.act,
                depthwise=use_depthwise,
            )

            if self.is_rknn_model:
                # replace model with RKNN-friendly modules:
                backbone.backbone.stem = RKFocus(
                    3, int(width * 64), ksize=3, act=self.act
                )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self,
        batch_size,
        is_distributed,
        no_aug=False,
        cache_img=False,
        include_pseudo=False,
    ):
        from yolox.data import (
            CustomVOC,
            TrainTransform,
            # YoloBatchSampler,
            # DataLoader,
            # InfiniteSampler,
            # MosaicDetection,
            # worker_init_reset_seed,
            get_yolox_datadir,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = CustomVOC(
                data_dir=os.path.join(get_yolox_datadir(), self.data_dir),
                class_names=self.classes,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
                img_size=self.input_size,
                remove_fraction=(1.0 - self.hpo_data_fraction)  # remove = 1.0 - keep
                if (self.perform_hpo or self.debug_mode)
                else 0.0,
            )

        dataset = self.wrap_mosaic(dataset, no_aug=no_aug)
        self.dataset = dataset

        # if is_distributed:
        #     batch_size = batch_size // dist.get_world_size()

        # sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        # batch_sampler = YoloBatchSampler(
        #     sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug,
        # )

        # dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        # dataloader_kwargs["batch_sampler"] = batch_sampler

        # # Make sure each process has different random seed, especially for 'fork' method.
        # # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        # dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        # train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        train_loader = self.dataset_to_dataloader(
            self.dataset, batch_size, is_distributed, no_aug=no_aug
        )

        return train_loader

    def dataset_to_dataloader(self, dataset, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            InfiniteSampler,
            worker_init_reset_seed,
            YoloBatchSampler,
            DataLoader,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        dataloader = DataLoader(dataset, **dataloader_kwargs)

        return dataloader

    def wrap_mosaic(self, dataset, no_aug=False, mixup_prob=None, moasic_prob=None):
        from yolox.data import MosaicDetection, TrainTransform

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=self.get_train_transform(max_labels=120),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=mixup_prob if mixup_prob is not None else self.mosaic_prob,
            mixup_prob=moasic_prob if moasic_prob is not None else self.mixup_prob,
        )
        return dataset

    def get_train_transform(self, max_labels=50):
        from yolox.data import TrainTransform

        return TrainTransform(
            max_labels=max_labels, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
        )

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, "random_size"):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        from yolox.utils import ClassicSGD

        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                if self.absolute_base_lr is not None:  # use absolute base lr instead
                    print(f"Using absolute learning rate of {self.absolute_base_lr}")
                    lr = self.absolute_base_lr
                else:
                    lr = self.basic_lr_per_img * batch_size
                    print(f"Using batch-size determined learning rate of {lr}")

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            """optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )"""
            optimizer = ClassicSGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        assert lr is not None, "learning rate cannot be None"
        assert lr > 0.0, "learning rate cannot be less than 0.0"

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_loss_callbacks(self) -> list:
        """Provide the trainer with callback functions that track "total_loss".

        Returns:
            list: list of callback functions that track loss.
        """
        from yolox.utils import ReduceLROnPlateau

        callbacks = []
        lr_callback = ReduceLROnPlateau(
            factor=self.lr_reduce_factor,
            patience=self.lr_reduce_patience,
            min_delta=self.lr_reduce_sensitivity,
            cooldown=self.lr_reduce_cooldown,
            min_lr=self.min_base_lr,
        )
        callbacks.append(lr_callback)
        return callbacks

    def get_epoch_callbacks(self) -> list:
        from yolox.utils import MomentumScheduler

        callbacks = []

        momentum_callback = MomentumScheduler(
            self.base_momentum, self.momentum, self.max_epoch, self.momentum_warmup
        )
        callbacks.append(momentum_callback)

        return callbacks

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import CustomVOC, ValTransform, get_yolox_datadir

        if self.eval_data_dir is None:
            if os.path.exists(os.path.join(self.data_dir, "Val")):
                self.eval_data_dir = os.path.join(self.data_dir, "Val")
            elif os.path.exists(os.path.join(self.data_dir, "Validation")):
                self.eval_data_dir = os.path.join(self.data_dir, "Validation")
            else:
                print(
                    "Could not find evaluation dataset. Using train set for evaluation"
                )
                self.eval_data_dir = self.data_dir

        valdataset = CustomVOC(
            data_dir=os.path.join(get_yolox_datadir(), self.eval_data_dir),
            class_names=self.classes,  # evaluation not possible if class names don't match
            preproc=ValTransform(legacy=legacy),
            img_size=self.test_size,
        )
        self.valdataset = valdataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)

    def get_eval_image_visualization(self):
        # using internal ref to dataset instead of building new one.
        # hence references latest dataset and its fraction if applicable
        return self.valdataset.visualize_detections()
