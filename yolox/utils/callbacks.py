import math


class BaseCallback:
    def __init__(self, target_type) -> None:
        self.target = None
        self.target_type = target_type

    def set_target(self, target):
        self.target = target

    def get_target_type(self):
        return self.target_type

    def consumer(self, x):
        assert self.target is not None
        pass


class MomentumScheduler(BaseCallback):
    def __init__(self, base_momentum, max_momentum, max_epochs, warmup=5):
        super().__init__("optimizer")

        self.m_base = base_momentum
        self.m_max = max_momentum  # also max momentum
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup

    def set_momentum(self, momentum):
        if momentum >= self.m_max or momentum >= 1.0:
            momentum = (
                self.m_max if self.m_max < 1.0 else 0.99
            )  # 1.0 - 1e-6  # set maximum momentum
        if momentum <= 0.0:
            momentum = 0.0
        for param_group in self.target.param_groups:
            param_group["momentum"] = momentum

    def consumer(self, epoch):
        if epoch < self.warmup_epochs:
            self.set_momentum(
                self.m_base + (self.m_max - self.m_base) / self.warmup_epochs * epoch
            )  # linear warmup
        else:
            self.set_momentum(
                self.m_base
                + 0.5
                * (self.m_max - self.m_base)
                * (1.0 + math.cos(math.pi * epoch / self.max_epochs))
            )


class ReduceLROnPlateau(BaseCallback):
    def __init__(
        self,
        factor=0.1,
        patience=5,
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        verbose=True,
    ):
        assert factor <= 1.0, "factor greater than 1.0 will cause lr increase"
        super().__init__("scheduler")

        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose

        self._reset()

    def _reset(self):
        """ Reset wait and cooldown counter"""
        self.monitor_op = lambda a, b: a < (b - self.min_delta)
        self.best = math.inf

        self.cooldown_counter = self.cooldown
        self.wait_counter = 0

    def check_in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def consumer(self, loss):
        assert (
            self.target is not None
        ), "ReduceLROnPlateau callback was not properly initialized: missing lr_scheduler target."

        if self.check_in_cooldown():
            self.cooldown_counter -= 1
            self.wait_counter = 0

        if self.monitor_op(loss, self.best):  # still decreasing
            self.best = loss
            self.wait_counter = 0

        elif not self.check_in_cooldown():  # not decreasing & not in cd
            self.wait_counter += 1

        if not self.check_in_cooldown() and self.wait_counter >= self.patience:
            current_lr = self.target.lr
            new_lr = max(current_lr * self.factor, self.min_lr)
            self.target.lr = new_lr
            self.target.update_lr_func()
            if self.verbose:
                print(
                    f"Base learning rate has been reduced from {current_lr} to {new_lr}"
                )

            self._reset()


class PatienceScheduler(BaseCallback):
    def __init__(self, interval):
        pass


class SimpleCurriculum(BaseCallback):
    """Simple 3 stage curriculum:
    1. Easiest data for warmup (learning general features)
    2. Hard data for convergence
    3. "Clean" data for either fine tuning or last phase training (no aug)
    """

    def __init__(self, datasets, warmup_epochs, finetune_epoch) -> None:
        super().__init__("train_loader_setter")
        self.datasets = datasets

        self.warmup_epochs = warmup_epochs
        self.finetune_epoch = finetune_epoch

    def consumer(self, epoch):
        if epoch == self.warmup_epochs:
            self.target(self.datasets[-2])
        if epoch == self.finetune_epoch:
            self.target(self.datasets[-1])
