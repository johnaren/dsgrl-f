from model import Model, plot_eval
from loss import Loss
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit
from utils import split_data, EarlyStop2 as EarlyStop, Keys
import json
from torch import BoolTensor
import numpy as np
from tqdm import tqdm


dataset = Planetoid("./planetoid", "Cora").to("cuda")
data = dataset[0]

with open("study/splits.json") as file:
    splits = json.load(file)

with open("study/params.json") as file:
    params = json.load(file)

for key, mask in splits.items():
    mask = BoolTensor(mask).to("cuda")
    setattr(data, key, mask)

num_splits = 5
num_runs = 3
total_score = np.zeros((num_splits, num_runs), dtype=float)
data = RandomNodeSplit("test_rest", num_splits)(data)

for split in range(num_splits):
    for run in tqdm(range(num_runs)):
        model = Model(
            in_features=data.num_node_features,
            aug_out=params[Keys.AOUT],
            repr_features=64,
            out_labels=data.y.unique().size(0),
            aug_layers=2,
            gnn_layers=2,
            dropout=params[Keys.DROP],
        ).to("cuda")
        model.pretrain(False).fit(
            data,
            CrossEntropyLoss(),
            1000,
            0.01,
            split=split,
            early_stopping=EarlyStop(patience=10)
        )
        total_score[split, run] = model.test(data, split=split).item()

mean = np.mean(total_score)
std = np.std(total_score)
print(f"{mean:.3f} +/- {std:.3f}")
plot_eval(model, "eval.png")