import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from typing import Self, Callable

from utils import Keys, numpy, update

import matplotlib.pyplot as plt


class Model(nn.Module):
    """
    A discriminative graph neural network for self-supervised learning in node
    classification tasks based on DSGRL feature augmentation described in [1].

    The model uses two learnable augmenters to produce two augmented views of
    the graph's nodes. The views are then passed independently through a
    shared graph convolutional network to produce two latent representations.
    A linear classifier uses the final representations to predict labels.

    [1] Ahmed E. Samy, Zekarias T. Kefato & Sarunas Girdzijauskas. (2024).
    "Data-Driven Self-Supervised Graph Representation Learning".
    https://arxiv.org/pdf/2412.18316
    """
    def __init__(
        self,
        in_features: int,
        aug_out: int,
        repr_features: int,
        out_labels: int,
        aug_layers: int,
        gnn_layers: int,
        dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        aug_hidden = (in_features + aug_out) // 2
        gnn_hidden = (aug_out + repr_features) // 2
        self.ffn1 = gnn.models.MLP(
            in_channels=in_features,
            hidden_channels=aug_hidden,
            num_layers=aug_layers, 
            out_channels=aug_out,
            norm=None,
            bias=False
        )
        self.ffn2 = gnn.models.MLP(
            in_channels=in_features,
            hidden_channels=aug_hidden,
            num_layers=aug_layers, 
            out_channels=aug_out,
            norm=None,
            bias=False
        )
        self.gnn = gnn.models.GCN(
            in_channels=aug_out,
            hidden_channels=gnn_hidden,
            out_channels=repr_features,
            num_layers=gnn_layers,
            dropout=dropout,
            bias=False
        )
        self.model = nn.ModuleList([self.ffn1, self.ffn2, self.gnn])
        self.classifier = nn.Linear(2 * repr_features, out_labels)
        self.optim1 = None
        self.optim2 = None
        self.reset_classifier()
        self.reset_history()
        self.pretrain(False)
        self.train(False)
        self._dat = {}

    def reset_classifier(self) -> Self:
        nn.init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.fill_(0.0)
        self.optim2 = None
        return self

    def pretrain(self, mode: bool = True) -> Self:
        self.pretraining = mode
        self.model.train(mode)
        self.classifier.train(not mode)
        return self
    
    def augment(self, x: Tensor) -> tuple[Tensor, Tensor]:
        view1 = self.ffn1.forward(x)
        view2 = self.ffn2.forward(x)
        return view1, view2

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        representation = self.gnn.forward(x, edge_index)
        return representation

    def predict(self, x: Tensor) -> Tensor:
        logits = self.classifier.forward(x)
        return logits

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        view1, view2 = self.augment(x)
        repr1 = self.encode(view1, edge_index)
        repr2 = self.encode(view2, edge_index)
        representation = torch.cat((repr1, repr2), dim=1)
        self._dat[Keys.VIEW] = (view1, view2)
        self._dat[Keys.REPR] = (repr1, repr2)

        if self.pretraining:
            return representation
        
        logits = self.predict(representation)
        self._dat[Keys.LOGI] = logits
        return logits

    def reset_history(self) -> None:
        self.history = {
            Keys.TRAI: [],
            Keys.VALI: []
        }

    def record(self, x: Tensor, key: str) -> None:
        if key not in self.history:
            raise Exception("invalid key")
        
        self.history[key].append(numpy(x).item())

    def prefit(
        self,
        data: Data,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        num_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = None,
        hop: list[int] = [-1],
        split: int | None = None,
        resume: bool = False
    ) -> Self:
        if split is None:
            mask = torch.BoolTensor(data.num_nodes).fill_(True).to(data.x.device)
        else:
            mask = data.train_mask[:, split] + data.val_mask[:, split] + data.test_mask[:, split]

        if batch_size is None:
            loader = [data.subgraph(mask)]
        else:
            loader = NeighborLoader(
                data,
                input_nodes=mask,
                num_neighbors=hop,
                batch_size=batch_size,
                shuffle=True
            )

        if self.optim1 is None or resume is False:
            self.optim1 = Adam(self.model.parameters(), lr)

        total_loss = torch.empty(1, device=data.x.device)
        self.pretrain()
        self.reset_history()

        for _ in range(num_epochs):
            total_loss.fill_(0.0)
            count = 0

            for batch in loader:
                self.forward(batch.x, batch.edge_index)
                loss = loss_fn(*self._dat[Keys.REPR])
                update(loss, self.optim1)
                total_loss += loss
                count += 1
            
            self.record(total_loss / count, Keys.TRAI)

        return self

    def fit(
        self,
        data: Data,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = None,
        num_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: tuple[int, int] = (None, None),
        hop: list[int] = [-1],
        early_stopping: Callable[[Tensor], bool] = None,
        split: int | None = None,
        resume: bool = False
    ) -> Self:
        if self.pretraining:
            return self.prefit(data, loss_fn, num_epochs, lr, batch_size[0], hop, split, resume)
        
        if split is not None:
            train_mask = data.train_mask[:, split]
            val_mask = data.val_mask[:, split]
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask

        if batch_size[0] is None:
            train = [data.subgraph(train_mask)]
        else:
            train = NeighborLoader(
                data,
                input_nodes=train_mask,
                num_neighbors=hop,
                batch_size=batch_size[0],
                shuffle = True
            )
        
        if batch_size[1] is None:
            val = [data.subgraph(val_mask)]
        else:
            val = NeighborLoader(
                data,
                input_nodes=val_mask,
                num_neighbors=hop,
                batch_size=batch_size[1]
            )

        if early_stopping is None:
            early_stopping = lambda x: False

        if self.optim2 is None or resume is False:
            self.optim2 = Adam(self.classifier.parameters(), lr)

        total_loss = torch.empty(1, device=data.x.device)
        self.reset_history()
        best_state = self.classifier.state_dict()
        best_score = torch.inf

        for _ in range(num_epochs):
            self.classifier.eval()
            total_loss.fill_(0.0)
            count = 0

            for val_batch in val:
                self.forward(val_batch.x, val_batch.edge_index)
                total_loss += loss_fn(self._dat[Keys.LOGI], val_batch.y)
                count += 1

            mean_loss = total_loss / count
            self.record(mean_loss, Keys.VALI)

            if mean_loss < best_score:
                best_state = self.classifier.state_dict()
                best_score = mean_loss

            if early_stopping(mean_loss):
                break

            self.classifier.train()
            total_loss.fill_(0.0)
            count = 0

            for train_batch in train:
                self.forward(train_batch.x, train_batch.edge_index)
                train_loss = loss_fn(self._dat[Keys.LOGI], train_batch.y)
                update(train_loss, self.optim2)
                total_loss += train_loss
                count += 1

            self.record(total_loss / count, Keys.TRAI)
        
        self.classifier.load_state_dict(best_state)
        return self
    
    def test(self, data: Data, split: int | None = None, mask = True) -> Tensor:
        self.pretrain(False)
        self.eval()

        if mask:
            mask = data.test_mask if split is None else data.test_mask[:, split]
            test = data.subgraph(mask)
        else:
            test = data
            
        logits = self.forward(test.x, test.edge_index)
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.count_nonzero(prediction == test.y) / test.num_nodes
        return accuracy


def plot_eval(model: Model, filename: str) -> None:
    plt.plot(model.history[Keys.TRAI], label=Keys.TRAI)
    plt.plot(model.history[Keys.VALI], label=Keys.VALI)
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def save_model(model: Model, filename: str) -> None:
    torch.save(model, filename)


def load_model(filename: str) -> Model:
    return torch.load(filename, weights_only=False)
