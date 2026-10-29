from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(__name__, level="ERROR")


def get_optimizer(
        class_name: str,
        optimizer_kwargs: dict[str, Any],
        parameters: Iterable[nn.Parameter],
    ) -> Optimizer:
    """Returns an initialized PyTorch optimizer.

    Parameters
    ----------
    class_name : str
        The name of the optimizer class defined in `torch.optim`,
        e.g. "Adam", "AdamW", or "SGD".

    optimizer_kwargs : Dict[str, Any]
        The keyword arguments to pass to the optimizer constructor,
        e.g. {"lr": 1e-3, "weight_decay": 1e-4}.

    parameters : Iterable[nn.Parameter]
        The model parameters to be optimized.

    Returns
    -------
    Optimizer
        An initialized instance of the specified PyTorch optimizer.
    """
    optimizer_class = getattr(torch.optim, class_name, None)

    if (
        not isinstance(optimizer_class, type)
        or not issubclass(optimizer_class, Optimizer)
    ):
        logger.error(f"Value error: '{class_name}' is not a valid PyTorch optimizer class.", ValueError)

    return optimizer_class(parameters, **optimizer_kwargs)


def get_lr_scheduler(
        class_name: str,
        scheduler_kwargs: dict[str, Any],
        optimizer: Optimizer,
    ) -> LRScheduler:
    """Returns an initialized PyTorch learning-rate scheduler.

    Parameters
    ----------
    class_name : str
        The name of the scheduler class defined in
        `torch.optim.lr_scheduler`,
        e.g. "StepLR", "CosineAnnealingLR", or "ExponentialLR".

    scheduler_kwargs : Dict[str, Any]
        The keyword arguments to pass to the scheduler constructor,
        e.g. {"step_size": 10, "gamma": 0.1}.

    optimizer : Optimizer
        The optimizer whose learning rate will be scheduled.

    Returns
    -------
    LRScheduler
        An initialized instance of the specified PyTorch scheduler.
    """
    scheduler_class = getattr(torch.optim.lr_scheduler, class_name, None)

    if (
        not isinstance(scheduler_class, type)
        or not issubclass(scheduler_class, LRScheduler)
    ):
        logger.error(
            f"Value error: '{class_name}' is not a valid PyTorch "
            "learning-rate scheduler class.", ValueError
        )

    return scheduler_class(optimizer, **scheduler_kwargs)