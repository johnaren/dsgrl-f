from model import Model, plot_eval
from loss import Loss, plot_train
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import Planetoid
from utils import split_data, EarlyStop3 as EarlyStop, Keys
import json
from torch import BoolTensor
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from torch_geometric.transforms import RandomNodeSplit
import torch


dataset = Planetoid("./planetoid", "Cora").to("cuda")
data = dataset[0]

"""
with open("study/splits.json") as file:
    splits = json.load(file)

with open("study/params.json") as file:
    params = json.load(file)

for key, mask in splits.items():
    mask = BoolTensor(mask).to("cuda")
    setattr(data, key, mask)
"""

num_splits = 1
num_runs = 1
num_epochs = 100
warmup = 0
train_loss = np.zeros((num_splits, num_runs, num_epochs + warmup), dtype=float)
invariance = np.zeros((num_splits, num_runs, num_epochs + warmup), dtype=float)
variance = np.zeros((num_splits, num_runs, num_epochs + warmup), dtype=float)
covariance = np.zeros((num_splits, num_runs, num_epochs + warmup), dtype=float)
modelreg = np.zeros((num_splits, num_runs, num_epochs + warmup), dtype=float)
train_score = np.zeros((num_splits, num_runs, 1 + num_epochs), dtype=float)
val_score = np.zeros((num_splits, num_runs, 1 + num_epochs), dtype=float)
test_score = np.zeros((num_splits, num_runs), dtype=float)
data = RandomNodeSplit("test_rest", num_splits, 19, 406)(data).to("cuda")

if num_splits == 1:
    for key in ("train", "val", "test"):
        attr = f"{key}_mask"
        mask = getattr(data, attr)
        setattr(data, attr, mask[:, torch.newaxis])

params = {
  "aug_out": 512,
  "dropout": 0.2,
  "alpha": 1.0,
  "gamma": 1.0,
  "learning rate": 0.005
}

for split in range(num_splits):
    train_data = data.subgraph(data.train_mask[:, split])
    val_data = data.subgraph(data.val_mask[:, split])

    for run in range(num_runs):
        model = Model(
            in_features=data.num_node_features,
            aug_out=params[Keys.AOUT],
            repr_features=64,
            out_labels=data.y.unique().size(0),
            aug_layers=2,
            gnn_layers=2,
            dropout=params[Keys.DROP],
        ).to("cuda")
        loss = Loss(model, alpha=params[Keys.ALPH], gamma=params[Keys.GAMM], record=True)
        model.pretrain(True).fit(
            data,
            loss,
            warmup,
            lr=params[Keys.LRAT]
        )
        train_loss[split, run, :warmup] = model.history[Keys.TRAI]
        model.pretrain(False).fit(
            data,
            CrossEntropyLoss(),
            1000,
            0.01,
            split=split,
            early_stopping=EarlyStop(patience=10)
        )
        train_score[split, run, 0] = model.test(train_data, mask=False).item()
        val_score[split, run, 0] = model.test(val_data, mask=False).item()
        best_score = val_score[split, run, 0]
        best_state = model.state_dict()
        best_loss_state = loss.state_dict()
        #earlystop = EarlyStop(patience=10, comparator=lambda x, y: x > y)

        for epoch in tqdm(range(num_epochs), desc=f"split {split}, run {run}"):
            model.pretrain(True).fit(
                data,
                loss,
                1,
                lr=params[Keys.LRAT],
                resume=True
            )
            train_loss[split, run, warmup + epoch] = model.history[Keys.TRAI][-1]
            model.reset_classifier().pretrain(False).fit(
                data,
                CrossEntropyLoss(),
                1000,
                0.01,
                split=split,
                early_stopping=EarlyStop(patience=10)
            )
            train_score[split, run, 1 + epoch] = model.test(train_data, mask=False).item()
            score = model.test(val_data, mask=False).item()
            val_score[split, run, 1 + epoch] = score

            if score > best_score:
                best_score = score
                best_state = model.state_dict()

            print(f"epoch {epoch}, score={val_score[split, run, epoch]}, best={best_score}")

            #if earlystop(score):
                #break

        invariance[split, run, :] = loss.history[Keys.INVA]
        variance[split, run, :] = loss.history[Keys.VARI]
        covariance[split, run, :] = loss.history[Keys.COVA]
        modelreg[split, run, :] = loss.history[Keys.MREG]
        model.load_state_dict(best_state)
        loss.load_state_dict(best_loss_state)
        test_score[split, run] = model.test(data, split=split).item()
        print(f"test score = {test_score[split, run]}")


def save_score(score, filename):
    mean = score.mean(axis=(0, 1))
    std = score.std(axis=(0, 1))
    maxi = score.max(axis=(0, 1))
    mini = score.min(axis=(0, 1))

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mean", "std", "max", "min"])
        writer.writerows(zip(mean, std, maxi, mini))

save_score(train_loss, "train_loss.csv")
save_score(invariance, "invariance.csv")
save_score(variance, "variance.csv")
save_score(covariance, "covariance.csv")
save_score(modelreg, "model regularization.csv")
save_score(train_score, "train_score.csv")
save_score(val_score, "val_score.csv")

test_mean = np.mean(test_score)
test_std = np.std(test_score)

with open("test_score.json", "w") as file:
    json.dump({"mean": test_mean, "std": test_std}, file, indent=2)

print(f"{test_mean:.3f} +/- {test_std:.3f}")

plot_train(loss, "train.png")
plot_eval(model, "eval.png")
