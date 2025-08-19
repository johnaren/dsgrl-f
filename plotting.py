import csv
import matplotlib.pyplot as plt
import numpy as np
from utils import Keys

def read(filename):
    mean = []
    std = []
    maxi = []
    mini = []

    with open(filename, "r") as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            mean.append(float(row[0]))
            std.append(float(row[1]))
            maxi.append(float(row[2]))
            mini.append(float(row[3]))

    return np.array(mean), np.array(std), np.array(maxi), np.array(mini)

def plot(mean, std, maxi, mini, label=None, color=None, linestyle=None, isstd=True, ismaxmin=False):
    plt.plot(mean, label=label, color=color, linestyle=linestyle)
    if isstd:
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2, color=color)
    if ismaxmin:
        plt.fill_between(np.arange(len(mean)), mini, maxi, alpha=0.2, color=color)

def plot_ax(mean, std, maxi, mini, ax, label=None, color=None, isstd=False):
    ax.plot(mean, label=label, color=color)
    if isstd:
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2, color=color)

plot(*read("train_loss.csv"), color="steelblue", isstd=False)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.savefig("train_loss.png")
plt.close()

plot(*read("train_score.csv"), "training", "darkorange", ":", isstd=False, ismaxmin=True)
plot(*read("val_score.csv"), "validation", "steelblue", "--", isstd=False, ismaxmin=True)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("eval_score.png")
plt.close()

_, axs = plt.subplots(2, 2)
axs = axs.flatten()
keys = [Keys.INVA, Keys.VARI, Keys.COVA, Keys.MREG]

for ax, key in zip(axs, keys):
    plot_ax(*read(f"{key}.csv"), ax=ax, color="steelblue")
    ax.set_title(str.capitalize(key))

plt.tight_layout()
plt.savefig("train.png")
plt.close()
