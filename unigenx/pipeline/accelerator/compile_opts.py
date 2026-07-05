# -*- coding: utf-8 -*-
import torch

from unigenx.logging import logger


def torch_compile(fn, state: bool):
    # check if cuda is available
    if not torch.cuda.is_available():
        logger.info("Torch.compile is disabled because cuda is not available.")
        return fn

    # individual set
    device_name = torch.cuda.get_device_name(0)
    fullgraph = False
    dynamic = None
    backend = "inductor"
    mode = "default"
    if device_name == "NVIDIA TITAN V":
        dynamic = True
        mode = "max-autotune-no-cudagraphs"
    elif device_name == "NVIDIA A100 80GB PCIe":
        dynamic = True

    if state:
        logger.info(
            f"Torch.compile is enabled with "
            f"fullgraph:{fullgraph}, dynamic:{dynamic}, backend:{backend}, mode={mode}."
        )

        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.suppress_errors = True
        return torch.compile(
            fn.cuda() if torch.cuda.is_available() else fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            disable=(not state),
        )
    else:
        return fn
