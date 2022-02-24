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

        self.data_num_workers = 4

        self.debug_mode = True
        self.max_epoch = 10

    def get_epoch_callbacks(self) -> list:
        callbacks = super().get_epoch_callbacks()

        from yolox.utils import SimpleCurriculum
        from yolox.data import CustomVOC, get_yolox_datadir, MixConcatDataset
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        def build_dataloader(
            keep_difficult=True,
            keep_fake=True,
            keep_truncated=True,
            keep_occluded=True,
            set_type="all",
            no_aug=False,
            mixup_prob=None,
            moasic_prob=None,
        ):
            # remove repetitive parts of building dataloader.
            dataset = CustomVOC(
                data_dir=os.path.join(get_yolox_datadir(), self.data_dir),
                class_names=self.classes,
                preproc=self.get_train_transform(),
                img_size=self.input_size,
                remove_fraction=(1.0 - self.hpo_data_fraction)  # remove = 1.0 - keep
                if (self.perform_hpo or self.debug_mode)
                else 0.0,
                keep_difficult=keep_difficult,
                keep_fake=keep_fake,
                keep_truncated=keep_truncated,
                keep_occluded=keep_occluded,
                set_type=set_type,
            )

            dataset = self.wrap_mosaic(
                dataset, no_aug=no_aug, mixup_prob=mixup_prob, moasic_prob=moasic_prob
            )
            return dataset

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            warmup_set = build_dataloader(
                keep_difficult=False,
                keep_fake=False,
                keep_occluded=False,
                keep_truncated=False,
                set_type="pure",  # no pseudo labels
            )

            # hard set is a mix of psuedo and pure
            pseudo_set = build_dataloader(
                set_type="pseudo", mixup_prob=1.0
            )  # must become soft labels
            pure_set = build_dataloader(set_type="pure")  # can use default settings
            if len(pseudo_set) == 0:
                assert len(pure_set) > 0
                hard_set = pure_set  # cannot concat with empty set
            else:
                hard_set = MixConcatDataset([pseudo_set, pure_set])

            finetune_set = build_dataloader(keep_fake=False, set_type="pure")

            datasets = [warmup_set, hard_set, finetune_set]

        callbacks.append(SimpleCurriculum(datasets, 2, 4))

        return callbacks
