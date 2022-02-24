import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):

        super(Exp, self).__init__(classes=["person", "bag", "luggage"])
        self.model_type = "small"

        self.data_dir = "bags_v3/Train"
        self.eval_data_dir = "bags_v3/Val"

        # self.act = "relu"  # test relu & others with rknn models

        self.input_size = (256, 256)  # (512, 512)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (256, 256)  # (512, 512)
        # self.ema_decay = 0.9999

        self.perform_hpo = True
        self.hpo_data_fraction = 0.5

        # HPO Search space for momentum and lr is based on current settings:
        self.momentum = 0.999

        # self.absolute_base_lr = 0.005
        self.basic_lr_per_img = 0.01 / 16  # original: 0.01 / 64.0

        # Suggestions from HPO
        self.ema_decay = 0.9999
        self.warmup_epochs = 30
        self.min_lr_ratio = 0.6
        self.lr_reduce_factor = 0.3
        self.lr_reduce_cooldown = 5
        self.lr_reduce_patience = 10
        self.max_epoch = 150  # 200
        self.lr_reduce_sensitivity = 0.0
        self.no_aug_epochs = 30

        self.data_num_workers = 2
        # self.print_interval = 100
        self.eval_interval = 2

