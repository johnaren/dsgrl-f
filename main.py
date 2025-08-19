from model import Model, save_model, load_model, plot_eval
from loss import Loss, plot_train, save_loss, load_loss
from utils import split_data, save_history, numpy, EarlyStop3 as EarlyStop, Keys
from torch_geometric.datasets import CitationFull, Planetoid
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial
import csv
from tqdm import tqdm
import os
import numpy as np
import json
import inspect


def evaluate(
    data: Data,
    split: int,
    model: Model,
    loss_fn,
    lr: float,
    patience: int,
    num_epochs: int
) -> dict:
    model.reset_classifier().pretrain(False).fit(
        data=data,
        split=split,
        loss_fn=loss_fn,
        lr=lr,
        num_epochs=num_epochs,
        early_stopping=EarlyStop(patience)
    )
    train_data = data.subgraph(data.train_mask[:, split])
    val_data = data.subgraph(data.val_mask[:, split])
    score = {
        Keys.TRAI: model.test(train_data, mask=False).item(),
        Keys.VALI: model.test(val_data, mask=False).item()
    }
    return score


def train_and_eval(
    data: Data,
    split: int,
    params: dict,
    warm_up: int,
    train_epochs: int,
    eval_epochs: int,
    eval_lr: float,
    patience: int,
    num_runs: int,
    device: int | str = "cuda"
) -> dict[Keys, list[list]]:
    data.to(device)
    features = data.num_node_features
    labels = data.y.unique().size(0)
    train_score = np.zeros((1 + train_epochs, num_runs), dtype=float)
    val_score = np.zeros((1 + train_epochs, num_runs), dtype=float)
    test_score = np.zeros((1, num_runs), dtype=float)

    for run in range(num_runs):
        model = Model(
            in_features=features,
            aug_out=params[Keys.AOUT],
            repr_features=64,
            out_labels=labels,
            aug_layers=2,
            gnn_layers=2,
            dropout=params[Keys.DROP]
        ).to(device)
        train_loss = Loss(
            model,
            alpha=params[Keys.ALPH],
            gamma=params[Keys.GAMM]
        )
        model.pretrain(True).fit(
            data,
            train_loss,
            num_epochs=warm_up,
            lr=params[Keys.LRAT],
            split=split
        )
        eval_loss = CrossEntropyLoss()
        initial_score = evaluate(data, split, model, eval_loss, eval_lr, patience, eval_epochs)
        train_score[0, run] = initial_score[Keys.TRAI]
        val_score[0, run] = initial_score[Keys.VALI]
        best_score = val_score[0, run]
        best_state = model.state_dict()

        for epoch in tqdm(range(1, 1 + train_epochs), desc=f"Training model"):
            model.pretrain(True).fit(
                data,
                train_loss,
                num_epochs=1,
                lr=params[Keys.LRAT],
                split=split
            )
            score = evaluate(data, split, model, eval_loss, eval_lr, patience, eval_epochs)
            train_score[epoch, run] = score[Keys.TRAI]
            val_score[epoch, run] = score[Keys.VALI]

            if val_score[epoch, run] > best_score:
                best_score = val_score[epoch, run]
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        test_score[:, run] = model.test(data, split).item()
    
    score = {
        Keys.TRAI: train_score.tolist(),
        Keys.VALI: val_score.tolist(),
        Keys.TEST: test_score.tolist()
    }
    return score


def save_score(score: dict, dir: str):
    os.makedirs(dir, exist_ok=True)

    for key, value in score.items():
        filename = os.path.join(dir, f"{key}.csv")

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(value)


def train_and_eval_splits(
    data: Data,
    params: dict,
    num_runs: int,
    warm_up: int,
    train_epochs: int,
    eval_epochs: int,
    eval_lr: int,
    patience: int,
    device: int | str = "cuda",
    savedir: str = "",
) -> list[dict[Keys, list[list]]]:
    num_splits = 1 if len(data.train_mask.shape) == 1 else data.train_mask.size(1)
    result = None

    for split in range(num_splits):
        score = train_and_eval(
            data,
            split,
            params,
            warm_up,
            train_epochs,
            eval_epochs,
            eval_lr,
            patience,
            num_runs,
            device
        )
        split_dir = f"split{split}" if num_splits > 1 else ""
        dir = os.path.join(savedir, split_dir)
        save_score(score, dir)
        
        if result is None:
            result = {key: [] for key in score.keys()}
        
        for key in score.keys():
            result[key].append(score[key])

    return result


def run_trials(
    data: Data,
    num_trials: int,
    num_runs: int,
    warm_up: int,
    train_epochs: int,
    eval_epochs: int,
    eval_lr: float,
    patience: int,
    device: int | str = "cuda",
    study_name: str = "study",
    load_study: bool = False
) -> tuple[dict, dict]:
    os.makedirs(study_name, exist_ok=True)
    study_file = os.path.join(study_name, f"{study_name}.db")
    tune_dir = os.path.join(study_name, "tuning")
    data.to(device)
    features = data.num_node_features
    aug_lo = int(features * 0.2)
    aug_hi = int(features * 0.8)
    aug_step = 5
    aug_lo = aug_hi - (aug_hi - aug_lo) // aug_step * aug_step
    assert (aug_hi - aug_lo) % aug_step == 0, "error calculating bounds"

    def objective(trial: Trial) -> float:
        trial.suggest_int(Keys.AOUT, aug_lo, aug_hi)
        trial.suggest_float(Keys.DROP, 0.1, 0.8, step=0.1)
        trial.suggest_float(Keys.ALPH, 0.1, 1.0, step=0.1)
        trial.suggest_float(Keys.BETA, 0.1, 1.0, step=0.1)
        trial.suggest_float(Keys.GAMM, 0.1, 1.0, step=0.1)
        trial.suggest_float(Keys.LRAT, 0.001, 0.01, step=0.001)
        trial_dir = os.path.join(tune_dir, f"trial{trial.number}")
        score = train_and_eval_splits(
            data,
            trial.params,
            num_runs,
            warm_up,
            train_epochs,
            eval_epochs,
            eval_lr,
            patience,
            device,
            savedir=trial_dir
        )
        trial.set_user_attr(Keys.SCOR, score)
        best_score = np.max(score[Keys.VALI], axis=1)
        mean_score = np.mean(best_score).item()
        return mean_score

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{study_file}",
        load_if_exists=load_study
    )
    masks = {
        Keys.TRAI: data.train_mask.tolist(),
        Keys.VALI: data.val_mask.tolist(),
        Keys.TEST: data.test_mask.tolist()
    }
    study.set_user_attr(Keys.MASK, masks)
    study.optimize(objective, num_trials, show_progress_bar=False)
    score = study.best_trial.user_attrs[Keys.SCOR]
    params = study.best_params
    data = {
        Keys.SCOR: score,
        Keys.PARA: params
    }
    return data


def run_experiment(
    num_train: float,
    num_val: float,
    trial_splits: int,
    train_splits: int,
    num_trials: int,
    num_runs: int,
    warm_up: int,
    train_epochs: int,
    eval_epochs: int,
    eval_lr: float,
    patience: int,
    device: int | str="cuda",
    name: str="study",
    load_study: bool=False
) -> tuple[float, float]:
    num_splits = trial_splits + train_splits
    dataset = Planetoid("./planetoid", "Cora")
    data = split_data(dataset[0], num_train, num_val, num_splits)
    masks = {
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask
    }
    os.makedirs(name, exist_ok=True)
    split_file = os.path.join(name, "splits.json")

    with open(split_file, "w") as file:
        json_data = {key: mask.tolist() for key, mask in masks.items()}
        json.dump(json_data, file, indent=2)

    for key, mask in masks.items():
        setattr(data, key, mask[:, :trial_splits])

    tune_result = run_trials(
        data,
        num_trials,
        num_runs,
        warm_up,
        train_epochs,
        eval_epochs,
        eval_lr,
        patience,
        device,
        study_name=name,
        load_study=load_study
    )
    params = tune_result[Keys.PARA]
    param_file = os.path.join(name, "params.json")

    with open(param_file, "w") as file:
        json.dump(params, file, indent=2)

    if num_trials > 0:
        tune_score = tune_result[Keys.SCOR]
        tune_acc = np.array(tune_score[Keys.TEST]).flatten()
    else:
        tune_acc = np.array([])

    if train_splits > 0:
        for key, mask in masks.items():
            setattr(data, key, mask[:, trial_splits:])

        train_score = train_and_eval_splits(
            data,
            params,
            num_runs,
            warm_up,
            train_epochs,
            eval_epochs,
            eval_lr,
            patience,
            device,
            savedir=os.path.join(name, "training")
        )
        train_acc = np.array(train_score[Keys.TEST]).flatten()
    else:
        train_acc = np.array([])

    acc = np.concatenate((tune_acc, train_acc))
    mean = np.nanmean(acc).item()
    std = np.nanstd(acc).item()
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_data = {arg: values[arg] for arg in args}
    res_data = {
        "mean": mean,
        "std": std
    }
    data = arg_data | res_data
    experiment_file = os.path.join(name, "experiment.json")

    with open(experiment_file, "w") as file:
        json.dump(data, file, indent=2)

    return mean, std


def main():
    mean, std = run_experiment(
        num_train=0.05,
        num_val=0.15,
        trial_splits=5,
        train_splits=0,
        num_trials=50,
        num_runs=3,
        warm_up=50,
        train_epochs=50,
        eval_epochs=1000,
        eval_lr=0.01,
        patience=10,
        device="cuda",
        name="study",
        load_study=True
    )
    print(f"experiment finished: accuracy = {mean:.3f} +/- {std:.3f}")


if __name__ == "__main__":
    main()
