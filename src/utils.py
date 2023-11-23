import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def fix_seed(seed=None):
    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"

    # reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    g = None

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 混同行列の作成
def create_confusion_matrix(targets, preds, title, save_path, is_one_dice=False):
    conf_matrix = confusion_matrix(targets, preds)
    if is_one_dice:
        classname = range(1, 7)
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
    else:
        classname = range(1, 13)
        plt.figure(figsize=(15, 12))
        sns.set(font_scale=1.4)

    sns.heatmap(
        conf_matrix,
        annot=True,
        annot_kws={"size": 16},
        xticklabels=classname,
        yticklabels=classname,
        fmt="d",
        cmap="Blues",
    )
    plt.xlabel("Predict")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.savefig(save_path)


# logitsの分布を可視化
def create_logits_distribution(logits, targets, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(logits[targets == 0], bins=50, label="normal", alpha=0.5)
    plt.hist(logits[targets == 1], bins=50, label="abnormal", alpha=0.5)
    plt.xlabel("Logit")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(save_path)
