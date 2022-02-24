import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__(classes=["person", "bag", "luggage"])
        self.model_type = "small"

        self.data_dir = "bags_v4/Train"
        self.eval_data_dir = "bags_v4/Val"

        self.input_size = (512, 512)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (512, 512)

        # Suggestions from HPO
        self.ema_decay = 0.9263
        self.warmup_epochs = 47
        self.min_lr_ratio = 1.175  # >1.0 in a few of the best runs
        self.lr_reduce_factor = 0.277
        self.lr_reduce_cooldown = 7
        self.lr_reduce_patience = 0
        self.max_epoch = 130
        self.lr_reduce_sensitivity = -1.159  # must show improvements
        self.no_aug_epochs = 36
        self.weight_decay = 0.01633
        self.momentum = 0.9096  # can be set high.
        self.base_momentum = 0.8432  # best when set very low
        self.momentum_warmup = 6
        # self.basic_lr_per_img = 0.01 / 16  # 0.06 / 16
        self.absolute_base_lr = 0.0006022
        # self.basic_lr_per_img = 0.0006022

        self.data_num_workers = 4
