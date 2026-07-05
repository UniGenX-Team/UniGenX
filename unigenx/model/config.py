# -*- coding: utf-8 -*-
import copy
import os
from dataclasses import dataclass

from transformers.models.llama.configuration_llama import LlamaConfig

from unigenx.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    TrainStrategy,
)


@dataclass
class UniGenXConfig(LlamaConfig, DistributedTrainConfig):
    seed: int = 42
    model_type: str = "threedimargen_100m"
    tokenizer: str = "num"

    vocab_size: int = 100
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    tokens_per_sample: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = None
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    max_sites: int = None
    scale_coords: float = None
    scale_energy: float = None
    reorder: bool = False
    niggli_reduced: bool = False

    dict_path: str = os.path.join(os.path.dirname(__file__), "../data/dict_mat.txt")
    train_data_path: str = None
    valid_data_path: str = None
    loadcheck_path: str = None
    results_folder: str = "../../result"
    exp_name: str = "case1"

    ft: bool = False
    infer: bool = False
    # for train
    mixed_precision: str = "no"
    wandb: bool = False
    train_batch_size: int = 16
    valid_batch_size: int = 16
    total_epochs: int = 10
    total_training_steps: int = -1
    clip_grad_norm: float = 1.0

    # for tqdm
    tqdm_interval: int = 1
    eval_interval: int = 2
    log_interval: int = 1
    is_step_log: bool = False

    # performance parameters
    gradient_accumulate_steps: int = 1

    # ------------------------------------------------------------------ #
    # Training fields (exact names read by the training engine +
    # config_optimizer). These are ADDITIVE and pin sensible release
    # defaults; the remaining engine fields come from DistributedTrainConfig.
    # The existing inference-compat fields above (total_training_steps,
    # gradient_accumulate_steps, clip_grad_norm) are intentionally kept.
    # ------------------------------------------------------------------ #
    strategy: TrainStrategy = TrainStrategy.Zero1
    fp16: bool = False
    bf16: bool = True
    total_num_steps: int = 1000
    warmup_num_steps: int = 60
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    save_dir: str = "./checkpoints"
    save_batch_interval: int = 0
    val_batch_interval: int = 0
    calculate_metrics: bool = False
    ifresume: bool = False
    load_ckpt: bool = False
    finetune_from_checkpoint_dir: str = None
    finetune_from_checkpoint_id: str = None

    # optimizer hyperparameters
    optimizer: str = "adamw"
    max_lr: float = 0.0001
    init_lr: float = 8e-5
    min_lr: float = 8e-6
    weight_decay: float = 0.0
    beta1: float = 0.9  # Adam
    beta2: float = 0.999  # Adam
    eps: float = 1e-8  # Adam

    # lr_scheduler
    lr_scheduler: str = "cosine_decay"
    warmup_epochs: int = 10

    # for diffusion
    num_timesteps_stepsize: int = -250
    ddpm_schedule: str = "sigmoid"
    num_timesteps: int = 5000
    ddpm_beta_start: float = 1e-7
    ddpm_beta_end: float = 2e-3
    diffusion_noise_std: float = 1.0

    # only for dpm solver
    is_solver: bool = False
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2
    solver_type: str = "dpmsolver"  # "midpoint"
    solver_steps: int = 20

    # for diffloss
    diff_width: int = 1024
    diff_depth: int = 3
    diff_steps: str = "100"
    diff_mul: int = 4
    # Learned-sigma variance channels in the diffusion head. True (default) ->
    # out_channels = target_channels * 2 (6 for xyz), matching all released
    # checkpoints except mol_qm9.pt (3-channel, learn_sigma=False).
    # unigenx_infer.py auto-detects this from the checkpoint on load.
    learn_sigma: bool = True

    # diffloss type
    diff_type: str = "diffloss"

    attn_implementation: str = "sdpa"

    freeze_llm: bool = False
    rotation_augmentation: bool = False
    translation_augmentation: bool = False
    edm_ori: bool = False

    # for future
    target: str = "material"

    # for denovo
    sample_size: int = 20000
    top_p: float = None
    temperature: float = None
    sample_max_length: int = 25

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def copy(self):
        """Deep copy of this config (UnifiedUniGenXDataset forks per-sub-target
        configs via ``args.copy()``)."""
        return copy.deepcopy(self)


@dataclass
class UniGenXInferenceConfig:
    input_file: str = None
    output_file: str = None
    infer_batch_size: int = 128
    max_length: int = None
    max_new_tokens: int = None
    verbose: bool = False
    space_group: bool = True
    sample: bool = False


@dataclass
class UniGenXInferencedenovoConfig:
    input_file: str = None
    output_file: str = None
    infer_batch_size: int = 128
    max_length: int = None
    max_new_tokens: int = None
    verbose: bool = False
    space_group: bool = True
    sample: bool = False
    sample_size: int = 100
    sample_max_length: int = 1000
    temperature: float = 0.75
    top_p: float = 0.95


def UniGenX_tiny_config(config: UniGenXConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def UniGenX_base_config(config: UniGenXConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 24
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def UniGenX_200m_config(config: UniGenXConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 12
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def UniGenX_100m_config(config: UniGenXConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 6
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def UniGenX_800m_config(config: UniGenXConfig):
    config.hidden_size = 1280
    config.intermediate_size = 5120
    config.num_hidden_layers = 32
    config.num_attention_heads = 20
    config.num_key_value_heads = 20
    return config


def UniGenX_1_6_b_config(config: UniGenXConfig):
    config.hidden_size = 2048
    config.intermediate_size = 8192
    config.num_hidden_layers = 24
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def UniGenX_3_3_b_config(config: UniGenXConfig):
    config.hidden_size = 2560
    config.intermediate_size = 10240
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config
