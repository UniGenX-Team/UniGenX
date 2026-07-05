# -*- coding: utf-8 -*-
import math
import multiprocessing
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import deepspeed
import torch
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
)

from unigenx.data.base import Batch, Data, FoundationModelDataset
from unigenx.logging import logger
from unigenx.pipeline.accelerator.compile_opts import torch_compile
from unigenx.pipeline.accelerator.dataclasses import (
    ModelOutput,
    TrainerState,
    TrainStrategy,
    ValidLogOutput,
)
from unigenx.pipeline.accelerator.fp16_scaler import FP16Scaler
from unigenx.utils.move_to_device import move_to_device

from .model import Model

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
except:
    logger.info("Transformer Engine package is not installed.")


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


class GroupedBatchIter(object):
    """
    This class is used to group batches into a larger batch. i.e., gradient accumulation.
    """

    def __init__(self, it, group_size, drop_last=False):
        self.it = it
        self.group_size = group_size
        self.drop_last = drop_last

    def __iter__(self):
        chunk = []
        for item in self.it:
            chunk.append(item)
            if len(chunk) == self.group_size:
                yield chunk
                chunk = []
        if not self.drop_last and chunk:
            yield chunk

    def __len__(self):
        if self.drop_last:
            return len(self.it) // self.group_size
        else:
            return (len(self.it) + self.group_size - 1) // self.group_size


class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        pass

    @abstractmethod
    def valid_step(self, batch_data: Batch) -> ValidLogOutput:
        pass

    @abstractmethod
    def save_checkpoint(
        self, ckpt_id: Union[int, str], extra_state: Optional[dict] = None
    ):
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: TrainerState,
        model_states_only: bool = False,
    ) -> TrainerState:
        pass

    @abstractmethod
    def build_data_loader(self, train_data, val_data):
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def sync_valid_loss(self, total_loss, num_examples):
        pass

    @property
    @abstractmethod
    def grad_scale(self) -> float:
        pass

    def before_epoch(self, epoch: int):
        pass

    def skip_first_batches(self, start_iteration):
        return False

    def _accumulate_log_output(
        self, total_log_output, log_output, current_sample_count, num_new_samples
    ):
        for key in log_output:
            value = log_output[key]
            if key not in total_log_output:
                if isinstance(value, torch.Tensor):
                    total_log_output[key] = value.item()
                elif isinstance(value, tuple):
                    if isinstance(value[0], torch.Tensor):
                        v0 = value[0].item()
                    else:
                        v0 = value[0]
                    if isinstance(value[1], torch.Tensor):
                        v1 = value[1].item()
                    else:
                        v1 = value[1]
                    total_log_output[key] = (v0, v1)
                else:
                    total_log_output[key] = value
            else:
                if isinstance(value, torch.Tensor):
                    total_log_output[key] = safe_div(
                        (
                            total_log_output[key] * current_sample_count
                            + value.item() * num_new_samples
                        ),
                        (current_sample_count + num_new_samples),
                    )
                elif isinstance(value, tuple):
                    if isinstance(value[0], torch.Tensor):
                        v0 = value[0].item()
                    else:
                        v0 = value[0]
                    if isinstance(value[1], torch.Tensor):
                        v1 = value[1].item()
                    else:
                        v1 = value[1]
                    v0_sum = total_log_output[key][0] * total_log_output[key][1]
                    v0_sum += v0 * v1
                    v1 += total_log_output[key][1]
                    v0 = safe_div(v0_sum, v1)
                    total_log_output[key] = (v0, v1)
                else:
                    total_log_output[key] = safe_div(
                        (
                            total_log_output[key] * current_sample_count
                            + value * num_new_samples
                        ),
                        (current_sample_count + num_new_samples),
                    )


class SingleNodeAccelerator(Accelerator):
    def __init__(
        self, args, model: Model, optimizer, lr_scheduler, device: str
    ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.world_size = 1

        if not torch.cuda.is_available():
            logger.warning("cuda is not available. use cpu instead")
            self.device = "cpu"
        self.scaler = FP16Scaler(
            init_scale=self.args.grad_scaler_init, enabled=self.args.fp16
        )

        if args.fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model = torch_compile(self.model, self.args.compile)

    @property
    def grad_scale(self) -> float:
        return self.scaler.scale

    def set_up(self):
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.model.config_optimizer()

    def barrier(self):
        pass

    def build_data_loader(
        self, train_data: FoundationModelDataset, valid_data: FoundationModelDataset
    ):
        train_batch_size_per_gpu = self.args.train_batch_size // (
            self.world_size * self.args.gradient_accumulation_steps
        )
        assert (
            train_batch_size_per_gpu > 0
        ), "train_batch_size_per_gpu should be greater than 0"

        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=train_batch_size_per_gpu,
            collate_fn=train_data.collate,
            drop_last=True,
        )

        if valid_data:
            valid_batch_size_per_gpu = self.args.val_batch_size // self.world_size
            assert (
                valid_batch_size_per_gpu > 0
            ), "valid_batch_size_per_gpu should be greater than 0"

            self.valid_data_loader = DataLoader(
                valid_data,
                sampler=None,
                batch_size=valid_batch_size_per_gpu,
                collate_fn=valid_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()

        self.optimizer.zero_grad()
        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        for batch_data in grouped_batch_data:
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
            loss = model_output.loss / len(grouped_batch_data)

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.warning("loss is nan or inf. skip this batch")
                loss = loss.new_tensor(0.0, requires_grad=True)
            else:
                success_batch_count += 1

            self.scaler.backward(loss)

            if model_output.num_examples is not None:
                self._accumulate_log_output(
                    total_log_output,
                    model_output.log_output,
                    sample_count,
                    model_output.num_examples,
                )
                sample_count += model_output.num_examples
                total_loss += model_output.loss * model_output.num_examples

            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)

        self.lr_scheduler.step()

        model_output.num_examples = sample_count
        model_output.loss = safe_div(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def valid_step(self, batch_data: Batch, epoch: int = 0) -> ValidLogOutput:
        self.model.eval()

        batch_data = move_to_device(batch_data, self.device)
        if self.args.AutoGradForce is True:
            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
        elif self.args.AutoGradForce is False:
            with torch.no_grad():
                pred = self.model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)

        if hasattr(batch_data, "batch_size"):
            num_examples = batch_data.batch_size
        elif hasattr(model_output, "num_examples"):
            num_examples = model_output.num_examples
        else:
            logger.info("num_examples is not found. set to None")
            num_examples = None

        return ValidLogOutput(
            valid_loss=model_output.loss.item(),
            epoch=epoch,
            num_examples=num_examples,
            logits=model_output.logits,
            label=model_output.label,
            extra_output=model_output.log_output,
        )

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_dir = Path(self.args.save_dir)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if self.args.fp16:
            checkpoint["fpscaler"] = self.scaler.scale

        if extra_state is not None:
            checkpoint.update(extra_state)
        logger.info("save checkpoint: {}", ckpt_id)
        torch.save(checkpoint, save_dir / ckpt_id)

        with open(save_dir / "checkpoint_list.txt", "a") as f:
            f.write(ckpt_id + "\n")

    def _transfer_optimizer_state_to_fp32(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if "exp_avg" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg"] = self.optimizer.state[p][
                        "exp_avg"
                    ].float()
                if "exp_avg_sq" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg_sq"] = self.optimizer.state[p][
                        "exp_avg_sq"
                    ].float()

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: TrainerState,
        model_states_only: bool = False,
    ) -> TrainerState:
        checkpoint_path = ckpt_dir / str(ckpt_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_weights = checkpoint.get("model", None)
        if model_weights is None:
            model_weights = checkpoint.get("module", None)
            assert model_weights is not None, "No model weights found in checkpoint"
        self.model.load_state_dict(model_weights)
        if not model_states_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if self.args.fp16:
                self.scaler.scale = checkpoint["fpscaler"]
                self._transfer_optimizer_state_to_fp32()

            logger.info(f"optimizer is loaded from checkpoint {ckpt_id}")

        if not model_states_only:
            for k, v in checkpoint.items():
                if k not in ["model", "optimizer", "lr_scheduler"]:
                    setattr(trainer_state, k, v)

        return trainer_state

    def sync_valid_loss(self, total_loss, num_examples):
        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        return None

    def skip_first_batches(self, start_iteration):
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        else:
            return False


class DdpAccelerator(SingleNodeAccelerator):
    def __init__(self, args, model: Model, optimizer, lr_scheduler) -> None:
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda")

    def set_up(self):
        super().set_up()
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        ddp_timeout = os.environ.get("DDP_TIMEOUT_MINUTES", None)
        logger.critical(
            f"Initializing DDP by env://. word size: {self.world_size}, rank: {self.rank}, "
            f"local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}, "
            f"DDP_TIMEOUT_MINUTES: {ddp_timeout}"
        )
        torch.distributed.init_process_group(
            backend=self.args.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(minutes=int(ddp_timeout))
            if ddp_timeout is not None
            else timedelta(minutes=30),
        )

        torch.distributed.barrier()

        logger.success("DDP initialized.")

        self.model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
        )
        if self.model.checkpoint_loaded:
            logger.info("Reloading checkpoint after DDP to ensure correctness.")
            self.ddp_model.module.reload_checkpoint()

        self.ddp_model = torch_compile(self.ddp_model, self.args.compile)

    def barrier(self):
        torch.distributed.barrier()

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.ddp_model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        for idx, batch_data in enumerate(grouped_batch_data):
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            # No sync for gradient accumulation
            maybe_no_sync = (
                self.ddp_model.no_sync()
                if idx != len(grouped_batch_data) - 1
                else nullcontext()
            )

            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info("loss is nan or inf. skip this batch")
                    # continue
                    success_batch_count += 1
                    mask = torch.isnan(loss) | torch.isinf(loss)
                    loss[mask] = 0.0
                    self.scaler.backward(loss)
                else:
                    success_batch_count += 1
                    self.scaler.backward(loss)

            self._accumulate_log_output(
                total_log_output,
                model_output.log_output,
                sample_count,
                model_output.num_examples,
            )
            sample_count += model_output.num_examples
            total_loss += model_output.loss * model_output.num_examples
            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)

        self.lr_scheduler.step()
        model_output.num_examples = sample_count
        model_output.loss = safe_div(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        if self.args.dynamic_loader:
            # The token-bucket dynamic loader depends on the Cython
            # ``batch_by_size`` kernel and is out of scope for this release.
            raise NotImplementedError(
                "dynamic_loader is not supported in the released training code."
            )
        elif self.args.ifstack:
            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=train_batch_size_per_gpu,
                collate_fn=train_data.collate,
                drop_last=True,
                num_workers=0,
            )
        else:
            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            if not isinstance(train_data, IterableDataset):
                if self.args.use_unified_batch_sampler:
                    # The unified multi-dataset batch sampler belongs to the
                    # (out-of-scope) psm data stack and is not vendored here.
                    raise NotImplementedError(
                        "use_unified_batch_sampler is not supported in the "
                        "released training code."
                    )
                else:
                    self.train_sampler = DistributedSampler(
                        train_data, num_replicas=self.world_size, rank=self.rank
                    )
                    self.train_data_loader = DataLoader(
                        train_data,
                        sampler=self.train_sampler,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=train_data.collate,
                        drop_last=True,
                    )
            elif self.args.use_dali_pipeline:
                self.train_sampler = None
                self.train_data_loader = DataLoader(
                    train_data,
                    batch_size=None,
                    collate_fn=train_data.collate,
                )
            else:
                self.train_sampler = None
                self.train_data_loader = DataLoader(
                    train_data,
                    batch_size=train_batch_size_per_gpu,
                    collate_fn=train_data.collate,
                    drop_last=True,
                    num_workers=1,
                )

        if val_data:
            if self.args.use_unified_batch_sampler:
                raise NotImplementedError(
                    "use_unified_batch_sampler is not supported in the "
                    "released training code."
                )
            elif self.args.use_dali_pipeline:
                self.valid_data_loader = DataLoader(
                    val_data,
                    batch_size=None,
                    collate_fn=val_data.collate,
                )
            else:
                valid_batch_size_per_gpu = self.args.val_batch_size // self.world_size
                assert (
                    valid_batch_size_per_gpu > 0
                ), "valid_batch_size_per_gpu should be greater than 0"
                validsampler = torch.utils.data.distributed.DistributedSampler(
                    val_data, num_replicas=self.world_size, shuffle=False
                )
                self.valid_data_loader = DataLoader(
                    val_data,
                    sampler=validsampler,
                    batch_size=valid_batch_size_per_gpu,
                    collate_fn=val_data.collate,
                    drop_last=False,
                )
        else:
            self.valid_data_loader = None

    def before_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_examples = torch.Tensor([num_examples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

    def sync_valid_metric(self, label_list, logits_list):
        if not label_list or not logits_list:
            return None, None

        label = torch.cat(label_list, dim=0).to(self.device)
        logits = torch.cat(logits_list, dim=0).to(self.device)
        num_samples = torch.zeros(
            self.world_size + 1, device=self.device, dtype=torch.long
        )
        num_samples[self.rank + 1] = label.shape[0]
        torch.distributed.all_reduce(num_samples)
        total_samples = int(torch.sum(num_samples).item())
        for i in range(1, self.world_size + 1):
            num_samples[i] += num_samples[i - 1]
        total_label = torch.zeros(
            total_samples, *label.shape[1:], device=self.device, dtype=label.dtype
        )
        total_logits = torch.zeros(
            total_samples, *logits.shape[1:], device=self.device, dtype=logits.dtype
        )

        total_label[num_samples[self.rank] : num_samples[self.rank + 1]] = label
        total_logits[num_samples[self.rank] : num_samples[self.rank + 1]] = logits
        torch.distributed.all_reduce(total_label)
        torch.distributed.all_reduce(total_logits)
        return total_label, total_logits

    def calculate_metric(self, label, logits):
        return self.model.calculate_metric(label, logits)

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: safe_div(v, log_num_dict[k]) for k, v in log_dict.items()}

    def skip_first_batches(self, start_iteration):
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        else:
            return False


class DeepSpeedAccelerator(Accelerator):
    def __init__(
        self,
        args,
        model,
        optimizer,
        lr_scheduler,
        train_data,
        valid_data,
        loss_log_dict={},
    ) -> None:
        super().__init__()
        self.args = args
        self.model = torch_compile(model, self.args.compile).cpu()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data = train_data
        self.valid_data = valid_data
        self.loss_log_dict = loss_log_dict

    @property
    def grad_scale(self) -> float:
        return self.optimizer.cur_scale

    def set_ds_config(self):
        if self.args.deepspeed_config_path != "":
            import json

            try:
                with open(self.args.deepspeed_config_path) as deepspeed_config_file:
                    self.args.deepspeed_config = json.load(deepspeed_config_file)
            except Exception as e:
                logger.warning(
                    f"Failed to load deepspeed config from {self.args.deepspeed_config}, "
                    f"using default config instead. Error message: {e}"
                )
                from unigenx.utils.defaultdsconfig import DEFAULT_DS_CONFIG

                self.args.deepspeed_config = DEFAULT_DS_CONFIG

        else:
            from unigenx.utils.defaultdsconfig import DEFAULT_DS_CONFIG

            self.args.deepspeed_config = DEFAULT_DS_CONFIG

            self.args.deepspeed_config["train_batch_size"] = self.args.train_batch_size
            self.args.deepspeed_config[
                "gradient_accumulation_steps"
            ] = self.args.gradient_accumulation_steps

            self.args.deepspeed_config["fp16"]["enabled"] = self.args.fp16
            self.args.deepspeed_config["fp16"]["auto_cast"] = self.args.auto_cast
            self.args.deepspeed_config["fp16"]["initial_scale_power"] = round(
                math.log2(self.args.grad_scaler_init)
            )

            self.args.deepspeed_config["bf16"]["enabled"] = self.args.bf16

            if self.args.strategy == TrainStrategy.Zero0:
                logger.warning(
                    "Zero0 is not compatible with offloading; setting zero_offload to False"
                )
                self.args.zero_offload = False

            if self.args.strategy == TrainStrategy.Zero0:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 0
            elif (
                self.args.strategy == TrainStrategy.Zero1
                or self.args.strategy == TrainStrategy.Pipeline
                or self.args.strategy == TrainStrategy.ThreeD
            ):
                self.args.deepspeed_config["zero_optimization"]["stage"] = 1
            elif self.args.strategy == TrainStrategy.Zero2:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 2
            elif self.args.strategy == TrainStrategy.Zero3:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 3
            elif self.args.strategy == TrainStrategy.ZeroInf:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 3
                self.args.deepspeed_config["zero_optimization"]["offload_optimizer"][
                    "device"
                ] = "nvme"
                self.args.deepspeed_config["zero_optimization"]["offload_param"][
                    "device"
                ] = "nvme"
                self.args.deepspeed_config["zero_optimization"]["offload_optimizer"][
                    "nvme_path"
                ] = self.args.zero_offload_dir
                self.args.deepspeed_config["zero_optimization"]["offload_param"][
                    "nvme_path"
                ] = self.args.zero_offload_dir
            else:
                raise ValueError(
                    f"Unsupported accelerator strategy: {self.args.strategy}"
                )

            if self.args.zero_offload and self.args.strategy != TrainStrategy.ZeroInf:
                self.args.deepspeed_config["zero_optimization"]["offload_optimizer"][
                    "device"
                ] = "cpu"
                self.args.deepspeed_config["zero_optimization"]["offload_param"][
                    "device"
                ] = "cpu"
                self.args.deepspeed_config["zero_force_ds_cpu_optimizer"] = False

            self.args.deepspeed_config["optimizer"]["params"]["lr"] = self.args.max_lr
            self.args.deepspeed_config["optimizer"]["params"]["betas"] = [
                self.args.beta1,
                self.args.beta2,
            ]
            self.args.deepspeed_config["optimizer"]["params"][
                "weight_decay"
            ] = self.args.weight_decay
            self.args.deepspeed_config["optimizer"]["params"]["betas"] = [
                self.args.beta1,
                self.args.beta2,
            ]
            self.args.deepspeed_config["optimizer"]["params"][
                "weight_decay"
            ] = self.args.weight_decay

            self.args.deepspeed_config["scheduler"]["params"][
                "total_num_steps"
            ] = self.args.total_num_steps
            self.args.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"
            ] = self.args.warmup_num_steps
            self.args.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ] = self.args.max_lr

            self.args.deepspeed_config[
                "gradient_clipping"
            ] = self.args.gradient_clipping
            self.args.deepspeed_config["steps_per_print"] = self.args.log_interval

            self.args.deepspeed_config["zero_optimization"][
                "ignore_unused_parameters"
            ] = not self.args.find_unused_parameters
            self.args.deepspeed_config["aio"][
                "ignore_unused_parameters"
            ] = not self.args.find_unused_parameters

            # self.args.deepspeed_config["wandb"]["enabled"] = self.args.wandb
            # self.args.deepspeed_config["wandb"]["team"] = self.args.wandb_team
            # self.args.deepspeed_config["wandb"]["group"] = self.args.wandb_group
            # self.args.deepspeed_config["wandb"]["project"] = self.args.wandb_project

            self.args.deepspeed_config["flops_profiler"][
                "enabled"
            ] = self.args.profiling
            self.args.deepspeed_config["flops_profiler"]["output_file"] = os.path.join(
                self.args.prof_dir, "profiler_ds.txt"
            )
            self.args.deepspeed_config["wall_clock_breakdown"] = self.args.profiling
            self.args.deepspeed_config["comms_logger"]["enabled"] = self.args.debug

            self.args.deepspeed_config["memory_breakdown"] = self.args.debug
            return

    def get_unfreeze_param_list(self, unfreeze_param_name_list: str):
        if unfreeze_param_name_list == "":
            logger.info(
                "unfreeze_param_list is empty, unfreeze all parameters with gradient"
            )
            if (
                self.args.strategy == TrainStrategy.Pipeline
                or self.args.strategy == TrainStrategy.ThreeD
            ):
                return [
                    param for param in self.ppmodel.parameters() if param.requires_grad
                ]
            else:
                return [
                    param for param in self.model.parameters() if param.requires_grad
                ]

        unfreeze_param = []
        unfreeze_param_name_list = list(
            filter(lambda x: x != "", unfreeze_param_name_list.split(","))
        ) + ["dummy"]

        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            for name, param in self.ppmodel.named_parameters():
                for param_name in unfreeze_param_name_list:
                    if name.find(param_name) != -1:
                        logger.info(f"Unfreezing {name}")
                        unfreeze_param.append(param)
                        break
        else:
            for name, param in self.model.named_parameters():
                for param_name in unfreeze_param_name_list:
                    if name.find(param_name) != -1:
                        logger.info(f"Unfreezing {name}")
                        unfreeze_param.append(param)
                        break

        return unfreeze_param

    def set_up(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        deepspeed.init_distributed(dist_backend=self.args.dist_backend)
        self.set_ds_config()

        if self.args.fp8:
            # Create FP8 recipe. Note: All input args are optional.
            self.fp8_recipe = recipe.DelayedScaling(
                fp8_format=recipe.Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            )

        # Pipeline / 3D-parallel training strategies are out of scope for the
        # released single-node training code.
        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            raise NotImplementedError(
                "Pipeline / ThreeD parallel strategies are not supported in the "
                "released training code; use Single, DDP or a Zero* strategy."
            )

        if self.optimizer is None or self.args.zero_offload:
            self.optimizer, self.lr_scheduler = self.model.config_optimizer()
        else:
            # When using custom scheduler, it is a good idea to set the optimizer type to None
            logger.info("custom optimizer is set, DS optimizer is disabled")
            self.args.deepspeed_config["optimizer"]["type"] = None

        if self.lr_scheduler is not None:
            # When using custom scheduler, we need to set the scheduler type to None
            # Otherwise, deepspeed will use that scheduler instead of the custom one
            logger.info("lr scheduler is set, remove the ds default scheduler")
            self.args.deepspeed_config["scheduler"]["type"] = None

        (
            self.model_engine,
            self.optimizer,
            self.train_data_loader,
            self.lr_scheduler,
        ) = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
            training_data=self.train_data,
            collate_fn=self.train_data.collate,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        self.build_data_loader(self.train_data, self.valid_data)

    @property
    def world_size(self) -> int:
        return self.model_engine.dp_world_size

    def barrier(self):
        deepspeed.comm.barrier()

    def before_epoch(self, epoch: int):
        super().before_epoch(epoch)
        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            if isinstance(self.train_data_loader, DeepSpeedDataLoader):
                if hasattr(self.train_data_loader.data_sampler, "set_epoch"):
                    self.train_data_loader.data_sampler.set_epoch(epoch)
            elif isinstance(self.train_data_loader, DataLoader):
                if hasattr(self.train_data_loader.batch_sampler, "set_epoch"):
                    self.train_data_loader.batch_sampler.set_epoch(epoch)
            else:
                raise ValueError(
                    f"Unknown training data loader type {type(self.train_data_loader)}"
                )

        # set seed of data sampler
        if isinstance(self.train_data_loader, DeepSpeedDataLoader):
            if hasattr(self.train_data_loader.data_sampler, "seed"):
                logger.info(f"Setting seed of data loader to {self.args.seed + epoch}.")
                self.train_data_loader.data_sampler.seed = self.args.seed + epoch
        elif isinstance(self.train_data_loader, DataLoader):
            if hasattr(self.train_data_loader.batch_sampler, "seed"):
                logger.info(f"Setting seed of data loader to {self.args.seed + epoch}.")
                self.train_data_loader.batch_sampler.seed = self.args.seed + epoch
        else:
            raise ValueError(
                f"Unknown training data loader type {type(self.train_data_loader)}"
            )

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        dp_rank = self.model_engine.global_rank
        num_replicas = self.model_engine.dp_world_size

        if self.args.use_unified_batch_sampler:
            # The unified multi-dataset batch sampler belongs to the
            # (out-of-scope) psm data stack and is not vendored here.
            raise NotImplementedError(
                "use_unified_batch_sampler is not supported in the released "
                "training code."
            )
        elif self.args.dynamic_loader:
            # The token-bucket dynamic loader depends on the Cython
            # ``batch_by_size`` kernel and is out of scope for this release.
            raise NotImplementedError(
                "dynamic_loader is not supported in the released training code."
            )
        elif self.args.ifstack:
            assert (
                self.args.strategy is not TrainStrategy.Pipeline
            ), "stack loader is not supported in pipeline mode"

            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.model_engine.dp_world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"
            logger.info(f"Using stack mode with batch size {train_batch_size_per_gpu}")

            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=train_batch_size_per_gpu,
                collate_fn=train_data.collate,
                drop_last=True,
                num_workers=1,
            )
        elif self.args.use_dali_pipeline:
            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=None,
                collate_fn=train_data.collate,
            )

        if self.valid_data:
            if self.args.use_unified_batch_sampler:
                raise NotImplementedError(
                    "use_unified_batch_sampler is not supported in the released "
                    "training code."
                )
            else:
                validsampler = torch.utils.data.distributed.DistributedSampler(
                    self.valid_data,
                    num_replicas=num_replicas,
                    rank=dp_rank,
                    shuffle=False,
                )
                if self.args.strategy == TrainStrategy.Pipeline:
                    logger.warning(
                        f"Using pipeline training of DeepSpeed, will validate with train_batch_size "
                        f"{self.args.deepspeed_config['train_batch_size']}, "
                        f"val_batch_size {self.args.val_batch_size} is being ignored."
                    )
                    valid_batch_size_per_gpu = (
                        self.model_engine.train_micro_batch_size_per_gpu()
                    )
                else:
                    valid_batch_size_per_gpu = self.args.val_batch_size // (
                        self.model_engine.dp_world_size
                        * self.args.gradient_accumulation_steps
                    )
                assert (
                    valid_batch_size_per_gpu > 0
                ), "valid_batch_size_per_gpu should be greater than 0"

                self.valid_data_loader = DataLoader(
                    self.valid_data,
                    sampler=validsampler,
                    batch_size=valid_batch_size_per_gpu,
                    collate_fn=self.valid_data.collate,
                    drop_last=False,
                )

            if (
                self.args.strategy == TrainStrategy.Pipeline
                or self.args.strategy == TrainStrategy.ThreeD
            ):
                self.valid_data_loader = GroupedBatchIter(
                    self.valid_data_loader,
                    self.args.gradient_accumulation_steps,
                    drop_last=True,
                )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data) -> ModelOutput:
        self.model_engine.module.train()
        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            with (
                te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
                if self.args.fp8
                else nullcontext()
            ):
                loss = self.model_engine.train_batch(
                    iter(grouped_batch_data),
                    reset_act_each_step=self.args.reset_act_each_step,
                )
            model_output = ModelOutput(
                loss=loss,
                num_examples=self.args.deepspeed_config["train_batch_size"]
                // self.world_size,
                log_output={"loss": loss},
            )
        else:
            sample_count = 0
            total_loss = 0.0
            total_log_output = {}
            for idx, batch_data in enumerate(grouped_batch_data):
                self.model_engine.tput_timer.start()
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )
                self.model.before_batch()
                with (
                    te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
                    if self.args.fp8
                    else nullcontext()
                ):
                    pred = self.model_engine(batch_data)

                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss
                self._accumulate_log_output(
                    total_log_output,
                    model_output.log_output,
                    sample_count,
                    model_output.num_examples,
                )
                sample_count += model_output.num_examples
                total_loss += model_output.loss * model_output.num_examples

                self.model.after_batch()

                self.model_engine.backward(loss)
                self.model_engine.step()

            model_output.num_examples = sample_count
            model_output.loss = safe_div(total_loss, sample_count)
            model_output.log_output = total_log_output

        return model_output

    def valid_step(
        self, batch_data: Union[Data, List], epoch: int = 0
    ) -> ValidLogOutput:
        self.model_engine.module.eval()
        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            pred, log_loss = self.model_engine.eval_batch(iter(batch_data))
            pred = pred.detach().item()
            extra_output = {
                k: v.detach().item() if isinstance(v, torch.Tensor) else v
                for k, v in log_loss.items()
            }
            return ValidLogOutput(
                valid_loss=pred,
                epoch=epoch,
                num_examples=self.args.deepspeed_config["train_batch_size"]
                / self.model_engine.dp_world_size,
                extra_output=extra_output,
            )
        else:
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )

            pred = self.model_engine(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)

            if hasattr(batch_data, "batch_size"):
                num_examples = batch_data.batch_size
            elif hasattr(model_output, "num_examples"):
                num_examples = model_output.num_examples
            else:
                logger.info("num_examples is not found. set to None")
                num_examples = None

            return ValidLogOutput(
                valid_loss=model_output.loss.item(),
                epoch=epoch,
                num_examples=num_examples,
                logits=model_output.logits,
                label=model_output.label,
                extra_output=model_output.log_output,
            )

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        client_state = {"ckpt_id": ckpt_id}
        if extra_state is not None:
            client_state.update(extra_state)
        self.model_engine.save_checkpoint(
            self.args.save_dir,
            client_state=client_state,
        )

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: TrainerState,
        model_states_only: bool = False,
    ) -> TrainerState:
        if isinstance(ckpt_id, int):
            ckpt_id = str(ckpt_id)
        print("ckpt_dir:", ckpt_dir)
        print("ckpt_id:", ckpt_id)
        load_path, client_sd = self.model_engine.load_checkpoint(
            str(ckpt_dir),
            load_optimizer_states=(not model_states_only),
            load_lr_scheduler_states=(not model_states_only),
            load_module_only=model_states_only,
            tag=ckpt_id,
            load_module_strict=False,
        )

        logger.info(f"Loaded checkpoint {load_path}")
        logger.info(f"The client state {client_sd}")

        if not model_states_only:
            self._transfer_optimizer_state_to_fp32()

        if not model_states_only:
            trainer_state.global_step = self.model_engine.global_steps
        return trainer_state

    def _transfer_optimizer_state_to_fp32(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if "exp_avg" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg"] = self.optimizer.state[p][
                        "exp_avg"
                    ].float()
                if "exp_avg_sq" in self.optimizer.state[p]:
                    self.optimizer.state[p]["exp_avg_sq"] = self.optimizer.state[p][
                        "exp_avg_sq"
                    ].float()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda()
        num_examples = torch.Tensor([num_examples * 1.0]).cuda()
        deepspeed.comm.all_reduce(total_loss)
        deepspeed.comm.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        if (
            self.args.strategy == TrainStrategy.Pipeline
            or self.args.strategy == TrainStrategy.ThreeD
        ):
            total_loss /= (
                self.model_engine.num_stages * self.model_engine.model_parallel_size
            )
            num_examples /= (
                self.model_engine.num_stages * self.model_engine.model_parallel_size
            )

        return total_loss, num_examples

    def sync_valid_metric(self, label_list, logits_list):
        if not label_list or not logits_list:
            return None, None

        label = torch.cat(label_list, dim=0).cuda()
        logits = torch.cat(logits_list, dim=0).cuda()
        num_samples = torch.zeros(
            self.world_size + 1, device=label.device, dtype=torch.long
        )
        num_samples[self.rank + 1] = label.shape[0]
        torch.distributed.all_reduce(num_samples)
        total_samples = int(torch.sum(num_samples).item())
        for i in range(1, self.world_size + 1):
            num_samples[i] += num_samples[i - 1]
        total_label = torch.zeros(
            total_samples, *label.shape[1:], device=label.device, dtype=label.dtype
        )
        total_logits = torch.zeros(
            total_samples, *logits.shape[1:], device=label.device, dtype=logits.dtype
        )

        total_label[num_samples[self.rank] : num_samples[self.rank + 1]] = label
        total_logits[num_samples[self.rank] : num_samples[self.rank + 1]] = logits
        torch.distributed.all_reduce(total_label)
        torch.distributed.all_reduce(total_logits)
        return total_label, total_logits

    def calculate_metric(self, label, logits):
        return self.model.calculate_metric(label, logits)

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            deepspeed.comm.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            deepspeed.comm.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: safe_div(v, log_num_dict[k]) for k, v in log_dict.items()}

    def skip_first_batches(self, start_iteration):
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        elif (
            self.args.strategy == TrainStrategy.Zero1
            or self.args.strategy == TrainStrategy.Pipeline
        ):
            num_stages = self.args.deepspeed_config.get(
                "num_pp_stages", self.args.pipeline_model_parallel_size
            )
            stage_id = self.model_engine.stage_id
            if (
                hasattr(self.train_data, "weight_dict")
                and self.train_data.weight_dict is not None
            ):
                if stage_id == 0 or stage_id == num_stages - 1:
                    self.train_data_loader.data_sampler.set_skip_batches(
                        start_iteration * self.args.gradient_accumulation_steps,
                        self.args.deepspeed_config["train_batch_size"]
                        // (
                            self.model_engine.dp_world_size
                            * self.args.gradient_accumulation_steps
                        ),
                    )
                return True
            else:
                return False
        else:
            return False
