from collections.abc import Iterable
from typing import Any, List, Optional, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aidevelopmenttoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(__name__, level="ERROR")


class SmartLRScheduler:
    """Wrap a PyTorch LRScheduler and perform the step of the
    chosen scheduler only when the current epoch is greater than or equal to the specified start epoch.

    Parameters
    ----------
    class_name : str
        Name of the learning-rate scheduler class defined in `torch.optim`, e.g. "StepLR" or "CosineAnnealingLR".

    scheduler_kwargs : dict[str, Any]
        The keyword arguments to pass to the scheduler constructor.

    optimizer : Optimizer
        The optimizer instance to which the scheduler will be applied.

    start_epoch : Optional[int], default=None
        The epoch number from which the scheduler should start updating the learning rate. 
        If `None`, the scheduler will start immediately.

    Notes
    -----
    Only explicit calls to `step` are delayed. The wrapped scheduler is
    constructed immediately and may change the optimizer's learning rate
    during initialization.
    """

    def __init__(
            self,
            class_name: str,
            scheduler_kwargs: dict[str, Any],
            optimizer: Optimizer,
            start_epoch: Optional[int] = None,
        ) -> None:

        self.class_name = class_name
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer = optimizer
        self.start_epoch = start_epoch

        scheduler_class = getattr(torch.optim.lr_scheduler, class_name, None)
        
        if not isinstance(scheduler_class, type) or not issubclass(scheduler_class, LRScheduler):
            logger.error(
                f"Value error: '{class_name}' is not a valid PyTorch "
                "learning-rate scheduler class.", ValueError
            )

        self.scheduler = scheduler_class(optimizer, **scheduler_kwargs)


    def step(
            self,
            current_epoch: int,
            scheduler_args: Optional[List[Any]] = None,
        ) -> None:
        """This method updates the learning rate scheduler based on the current epoch and provided arguments.
        The step will only be executed if the current epoch is greater than or equal to the start epoch, 
        or if no start epoch is specified.

        Parameters
        ----------
        current_epoch : int
            The current epoch number.

        scheduler_args : Optional[List[Any]], default=None
            The positional arguments to pass to the scheduler's `step` method,
            such as `[val_loss]` for ReduceLROnPlateau.
        """

        if self.start_epoch is None or current_epoch >= self.start_epoch:
            self.scheduler.step(*(scheduler_args if scheduler_args is not None else []))


    def get_last_lr(self) -> List[float]:
        """Return the learning rates reported by the wrapped scheduler."""
        return self.scheduler.get_last_lr()


    def state_dict(self) -> Dict[str, Any]:
        """Return wrapper configuration and wrapped scheduler state."""
        return {
            "start_epoch": self.start_epoch,
            "scheduler": self.scheduler.state_dict(),
        }


    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state into a wrapper built with the same scheduler class."""
        self.start_epoch = state_dict["start_epoch"]
        self.scheduler.load_state_dict(state_dict["scheduler"])


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

    if not isinstance(optimizer_class, type) or not issubclass(optimizer_class, Optimizer):
        logger.error(f"Value error: '{class_name}' is not a valid PyTorch optimizer class.", ValueError)

    return optimizer_class(parameters, **optimizer_kwargs)