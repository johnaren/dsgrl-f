import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

import numpy as np

from enum import Enum

from typing import Optional

import csv


class Keys(str, Enum):
    REPR = "representation"
    PRED = "prediction"
    LOSS = "loss"
    VIEW = "view"
    LOGI = "logits"
    TRAI = "training"
    VALI = "validation"
    TEST = "test"
    WEIG = "weight"
    BIAS = "bias"
    PRET = "pretraining"
    INVA = "invariance"
    VARI = "variance"
    COVA = "covariance"
    MREG = "model regularization"
    ACCU = "accuracy",
    AOUT = "aug_out",
    DROP = "dropout",
    ALPH = "alpha",
    BETA = "beta",
    GAMM = "gamma",
    EPSI = "epsilon",
    LAMB = "lambda",
    LRAT = "learning rate",
    SCOR = "score",
    HIST = "history",
    RUN_ = "run",
    PARA = "parameters",
    MASK = "mask"

    def __str__(self) -> str:
        return self.value


def numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def update(loss: Tensor, optimizer: Optimizer) -> None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_weights(module: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in module.named_parameters() if str(Keys.WEIG) in name]

"""
def split(
    data: Data,
    *splits: float,
    shuffle: Optional[bool] = True,
    balanced: Optional[bool] = False
) -> list[Tensor]:
    if sum(splits) > 1.0:
        raise Exception("Split sum must not exceed 1.0!")
    
    num_nodes = data.num_nodes
    device = data.x.device
    order = torch.randperm(data.num_nodes) if shuffle else torch.arange(data.num_nodes)
    order.to(device)
    masks = [torch.zeros(num_nodes, dtype=torch.bool, device=device) for _ in splits]
    labels = data.y.unique()
    num_labels = len(labels)

    for label in labels:
        class_indices, = torch.where(data.y[order] == label)
        num_classes = len(class_indices)
        start = 0

        for i, ratio in enumerate(splits):
            size = ratio * num_nodes / num_labels if balanced and i == 0 else ratio * num_classes
            end = start + int(size)
            selection = class_indices[start:end]
            masks[i][selection] = True
            start = end

    return masks
"""

def split_data(
    data: Data,
    train: float,
    val: float,
    num_splits: int=10
) -> Data:
    """
    Split data into training, validation and test splits.
    Masks are saved as attributes in the data object,
    and have shape (data.num_nodes, num_splits).
    """
    num_nodes = data.num_nodes
    num_labels = data.y.unique().size(0)
    num_train = int(train * num_nodes / num_labels)
    transform = RandomNodeSplit(
        split="test_rest",
        num_splits=num_splits,
        num_train_per_class=num_train,
        num_val=val
    )
    data = transform(data)

    if num_splits == 1:
        for key in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, key)
            setattr(data, key, mask[:, torch.newaxis])

    return data


def save_history(history: dict, filename: str) -> None:
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

class EarlyStop():
    def __init__(self, patience: int=5):
        self.best = torch.inf
        self.patience = patience

    def __call__(self, value: Tensor) -> bool:
        if value < self.best:
            self.best = value
        else:
            self.patience -= 1

        return self.patience <= 0


class EarlyStop2():
    def __init__(self, patience: int=5):
        self.best = torch.inf
        self.patience = patience
        self.counter = 0

    def __call__(self, value: Tensor) -> bool:
        if value < self.best:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class EarlyStop3():
    def __init__(self, patience: int=5, comparator=None):
        self.best = None
        self.patience = patience
        self.counter = 0

        if comparator is None:
            self.compare = lambda x, y: x < y
        else:
            self.compare = comparator

    def __call__(self, value: Tensor) -> bool:
        if self.best is None:
            self.best = value
        elif self.compare(value, self.best):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
