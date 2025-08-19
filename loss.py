import torch
import torch.nn as nn

from torch import Tensor

from model import Model

from utils import Keys, numpy, get_weights

import matplotlib.pyplot as plt


class Invariance(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        difference = x1 - x2
        matrix_norm = torch.linalg.norm(difference, dim=1)
        return torch.mean(matrix_norm)


class Variance(nn.Module):
    def __init__(
        self,
        floor: float = 1.0e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.floor = floor
 
    def forward(self, x: Tensor) -> Tensor:
        variance = torch.var(x, dim=0)
        std = torch.sqrt(variance + self.floor)
        loss = torch.relu(1.0 - std)
        return torch.mean(loss)


class Covariance(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, dims = x.shape
        center = torch.mean(x, dim=0, keepdim=True)
        centered = x - center
        covariance = centered.T @ centered / (batch_size - 1)
        mask = 1 - torch.eye(dims, device=x.device)
        squared_masked_covariance = (covariance * mask) ** 2
        return torch.sum(squared_masked_covariance) / dims


class ModelRegularization(nn.Module):
    def __init__(
        self,
        model: Model,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        weights = self.stack_weights()
        self.mask = [torch.eye(w.size(0), device=w.device, dtype=w.dtype) for w in weights]

    def stack_weights(self) -> list[Tensor]:
        w1 = get_weights(self.model.ffn1)
        w2 = get_weights(self.model.ffn2)
        weights = []
        
        for layer in zip(w1, w2):
            w = torch.cat(layer)
            w = torch.nn.functional.normalize(w)
            weights.append(w)

        return weights

    def forward(self) -> Tensor:
        weights = self.stack_weights()
        return sum(torch.linalg.norm(w @ w.T - self.mask[i]) for i, w in enumerate(weights))


class Loss(nn.Module):
    def __init__(
        self,
        model: Model,
        alpha: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 1.0e-6,
        gamma: float = 1.0,
        lam: float = 1.0,
        record: bool = False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inv_factor = alpha
        self.var_factor = beta
        self.cov_factor = gamma
        self.modelreg_factor = lam
        self.inv = nn.MSELoss()
        self.var = Variance(floor=epsilon)
        self.cov = Covariance()
        self.modelreg = ModelRegularization(model)
        self.record_self = record
        self.reset_history()

    def reset_history(self) -> None:
        self.history = {
            Keys.INVA: [],
            Keys.VARI: [],
            Keys.COVA: [],
            Keys.MREG: []
        }
    
    def record(self, x: Tensor, key: str) -> None:
        if key not in self.history:
            raise Exception("Invalid key")
        
        self.history[key].append(numpy(x).item())

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        invariance = self.inv(x1, x2)
        variance = self.var(x1) + self.var(x2)
        covariance = self.cov(x1) + self.cov(x2)
        latentreg = self.var_factor * variance + self.cov_factor * covariance
        modelreg = self.modelreg()
        loss = self.inv_factor * invariance + latentreg + self.modelreg_factor * modelreg

        if self.record_self:
            self.record(invariance, Keys.INVA)
            self.record(variance, Keys.VARI)
            self.record(covariance, Keys.COVA)
            self.record(modelreg, Keys.MREG)
        
        return torch.mean(loss)


def plot_train(loss: Loss, filename: str) -> None:
    _, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    keys = [Keys.INVA, Keys.VARI, Keys.COVA, Keys.MREG]
    
    for ax, key in zip(axs, keys):
        ax.plot(loss.history[key])
        ax.set_title(key)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_loss(loss: Loss, filename: str) -> None:
    torch.save(loss, filename)


def load_loss(filename: str) -> None:
    return torch.load(filename, weights_only=False)
