# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import unigenx.model.unigenx as model
from unigenx.logging import logger
from unigenx.model.modules.criterions import CrystalCriterions, ModelOutput
from unigenx.model.modules.optimizer import myAdamW
from unigenx.model.modules.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR
from unigenx.utils.checkpoint import load_checkpoint


class UniGenX(nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, config, not_init=False):
        super().__init__()
        if not_init:
            return

        self.loss = CrystalCriterions(config.vocab_size)

        self.config = config
        self.net = model.UniGenX(config)

    def forward(self, batched_data, **kwargs):
        return self.net(**batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        return self.loss(model_output, batch_data)

    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def config_optimizer(
        self, model: Optional[nn.Module] = None
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        if model is None:
            model = self
        unfreeze_list = None
        if self.config.freeze_llm:
            unfreeze_list = ""
            for name, _ in model.named_parameters():
                if name.find("diffloss") != -1:
                    unfreeze_list = unfreeze_list + name + ","
        optimizer, _ = myAdamW(
            model,
            unfreeze_list=unfreeze_list,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)

    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        checkpoints_state = load_checkpoint(checkpoint_path)
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )
