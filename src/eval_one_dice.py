import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.utils.data
from easydict import EasyDict
from tqdm import tqdm

sys.path.append(".")
from configs.config import CONF  # noqa: E402
from dataset.one_dice import CustomDataset  # noqa: E402
from model.one_dice import CNNModel, CustomResNetModel  # noqa: E402
from utils import create_confusion_matrix, fix_seed, seed_worker  # noqa: E402


def main(args):
    g = fix_seed(args.seed)

    # GPU設定
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # データセットの読み込み
    X_test = np.load(
        os.path.join(CONF.PATH.DATASET, cfg.data_dir, "X_test_filtered.npy")
    )
    y_test = np.load(
        os.path.join(CONF.PATH.DATASET, cfg.data_dir, "y_test_filtered.npy")
    )

    test_dataset = CustomDataset(X_test, y_test)

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # モデルの読み込み
    if cfg.model == "cnn":
        model = CNNModel()
    elif cfg.model == "resnet":
        model = CustomResNetModel(model=cfg.model_layer, pretrained=cfg.pretrained)
    else:
        raise NotImplementedError
    model.to(device)

    weights = torch.load(
        os.path.join(
            CONF.PATH.OUTPUT,
            args.folder,
            args.pretrained_model,
        ),
        map_location=device,
    )
    model.load_state_dict(weights)
    model.eval()

    results_list = []
    correct = 0
    total = 0
    for X, y in tqdm(dataloader):
        X = X.to(device)

        with torch.no_grad():
            output = model(X)
            logits = torch.sigmoid(output["logits"]).cpu()

        _, pred = torch.max(logits, 1)
        pred = pred.item() + 1
        y = y + 1

        result_dict = {
            "target": y.item(),
            "pred": pred,
        }
        results_list.append(result_dict)

        total += y.size(0)
        correct += (pred == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {accuracy:.3f}")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(
        os.path.join(
            CONF.PATH.OUTPUT,
            args.folder,
            f"eval_result_{accuracy:.3f}.csv",
        )
    )

    if not args.no_plots:
        # 混同行列の作成
        create_confusion_matrix(
            results_df["target"],
            results_df["pred"],
            title=(f"Confusion Matrix (Acc: {accuracy:.3f})"),
            save_path=os.path.join(
                CONF.PATH.OUTPUT,
                args.folder,
                "confusion_matrix.png",
            ),
            is_one_dice=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder", type=str, default="20231114-130123_one_dice_cnn"
    )
    parser.add_argument("-p", "--pretrained_model", type=str, default="model.pth")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--no-plots", action="store_true", help="no plots")
    args = parser.parse_args()
    args = EasyDict(vars(args))

    with open(os.path.join(CONF.PATH.OUTPUT, args.folder, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    cfg.update(args)
    print(cfg)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
