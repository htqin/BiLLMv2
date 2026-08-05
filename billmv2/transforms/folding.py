"""Function-preserving linear rotation folding."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .rotation import fold_input_rotation


def fold_linear_input(linear: nn.Linear, rotation: Tensor) -> None:
    """Fold an input-side orthogonal transform into a linear module."""

    folded = fold_input_rotation(linear.weight.detach().float(), rotation.float())
    linear.weight.data.copy_(folded.to(linear.weight))
