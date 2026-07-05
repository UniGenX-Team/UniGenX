# -*- coding: utf-8 -*-
# Copyright 2022 Microsoft Corporation.
import math

from deepspeed.runtime.lr_schedules import WarmupLR
from deepspeed.utils import logger
from torch.optim import Optimizer

WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"
DECAY_LINEAR_RATE = "linear"
DECAY_COSINE_RATE = "cosine"


class groupWarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
    over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_num_steps (int): total number of training steps
        warmup_min_lr (float or list): minimum learning rate. Default: 0
        warmup_max_lr (float or list): maximum learning rate. Default: 0.001
        warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
        warmup_type {'log', 'linear'}: increasing function from min_lr to max_lr during warmup. Default: log
        last_batch_iteration (int): The index of the last batch. Default: -1.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = WarmupDecayLR(optimizer, 1000000)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LINEAR_RATE,
        last_batch_iteration: int = -1,
        d_tilde: float = 1.0,
        decay_type: str = DECAY_COSINE_RATE,
    ):
        self.total_num_steps = total_num_steps
        super(groupWarmupDecayLR, self).__init__(
            optimizer,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )
        self.d_tilde = d_tilde
        self.decay_type = decay_type

        if self.total_num_steps < self.warmup_num_steps:
            logger.warning(
                "total_num_steps {} is less than warmup_num_steps {}".format(
                    total_num_steps, warmup_num_steps
                )
            )
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        # if "d_tilde" in self.optimizer.param_groups[0]:
        #     self.optimizer.param_groups[0]['lr'] *= self.optimizer.param_groups[0]['d_tilde']
        #     self.optimizer.param_groups[1]['lr'] *= self.optimizer.param_groups[1]['d_tilde']
        # else:
        if self.d_tilde >= 1.0:
            self.optimizer.param_groups[0]["lr"] /= self.d_tilde
        elif self.d_tilde < 1.0:
            self.optimizer.param_groups[0]["lr"] *= self.d_tilde
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["lr"] *= self.d_tilde

                # self.optimizer.param_groups[0].data._grad *= self.d_tilde

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(
                    self.last_batch_iteration + 1
                )
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        else:
            if self.decay_type == DECAY_LINEAR_RATE:
                return max(
                    0.0,
                    float(self.total_num_steps - self.last_batch_iteration)
                    / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
                )
            else:
                return 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * float(self.last_batch_iteration - self.warmup_num_steps)
                        / float(max(1.0, self.total_num_steps - self.warmup_num_steps))
                    )
                )
