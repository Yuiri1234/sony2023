import argparse
import csv
import datetime
import os
import pickle
import re
import sys

import numpy as np
import torch.optim
import torch.utils.data
import yaml
from easydict import EasyDict
from tqdm import tqdm
from utils import fix_seed

sys.path.append(".")
from configs.config import CONF  # noqa: E402
from dataset.one_dice import CustomDataset  # noqa: E402
from model.one_dice import CNNModel, CustomResNetModel  # noqa: E402
from utils import seed_worker  # noqa: E402


class Trainer:
    def __init__(self, cfg, generator):
        self.cfg = cfg

        # GPU設定
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # データセットの読み込み
        X_train = np.load(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "X_train_filtered.npy")
        )
        y_train = np.load(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "y_train_filtered.npy")
        )
        X_val = np.load(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "X_val_filtered.npy")
        )
        y_val = np.load(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "y_val_filtered.npy")
        )

        self.train_dataset = CustomDataset(X_train, y_train)
        self.val_dataset = CustomDataset(X_val, y_val)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,  # メモリのページングをしないように設定
            worker_init_fn=seed_worker,  # シード固定
            generator=generator,  # シード固定
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        # モデルの読み込み
        if cfg.model == "cnn":
            self.model = CNNModel()
        elif cfg.model == "resnet":
            self.model = CustomResNetModel(
                model=self.cfg.model_layer, pretrained=self.cfg.pretrained
            )
        else:
            raise NotImplementedError
        self.model.to(self.device)

        optimizer_grouped_parameters = [
            # weight_decayの設定
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in cfg.no_decay) and p.requires_grad
                ],
                "weight_decay": cfg.weight_decay,
            },
            # weight_decayを設定しない場合
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in cfg.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=float(cfg.lr), eps=float(cfg.eps)
        )

        self.log_data = []
        self.folder = os.path.join(
            CONF.PATH.OUTPUT, f"{self.cfg.now_str}_{self.cfg.folder}"
        )
        self.csv_file = os.path.join(self.folder, "history.csv")
        self.yaml_file = os.path.join(self.folder, "config.yaml")
        self.pickle_file = os.path.join(self.folder, "config.pkl")

        # フォルダがなければ作成
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # 既存のcsvファイルを空にする
        with open(self.csv_file, "w", newline="") as f:
            f.truncate(0)

        # EasyDictをYAMLファイルに保存
        with open(self.yaml_file, "w") as yaml_file:
            yaml.dump(self.cfg, yaml_file, default_flow_style=False)

        # EasyDictをpickleファイルに保存
        with open(self.pickle_file, "wb") as pickle_file:
            pickle.dump(self.cfg, pickle_file)

    def train_one_epoch(self, epoch=0):
        """
        Trains the model for one epoch.
        Config setting:
            None
        Inputs:
            current epoch num
        Returns:
            None
        Ourputs:
            None
        """
        self.model.train()
        for i, (X, y) in enumerate(self.train_loader):
            global_step = i + epoch * len(self.train_loader)
            X = X.to(self.device)
            y = y.to(self.device)

            output = self.model(X, y)
            loss = output["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if global_step % 100 == 0:
                log = {"epoch": epoch, "global_step": global_step, "loss": loss.item()}
                print(log)
                self.log_data.append(log)
                self.write_log_to_csv()

    def train(self):
        """
        Trains the model for a specified number of epochs.
        Config setting:
            epochs
        Inputs:
            None
        Returns:
            None
        Ourputs:
            trained model
        """
        for epoch in tqdm(range(self.cfg.epochs)):
            self.train_one_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                self.save_model(f"epoch{epoch + 1:04d}")

        self.save_model("model")

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.folder, f"{name}.pth"))

    def write_log_to_csv(self):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(self.log_data[-1])


def main(args):
    g = fix_seed(args.seed)
    cfg = EasyDict(yaml.load(open(args.config), yaml.SafeLoader))
    cfg.update(args)

    trainer = Trainer(cfg, generator=g)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/one_dice_default.yaml",
        help="config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--now_str", type=str)
    args = parser.parse_args()
    args = EasyDict(vars(args))

    # folder名の決定（yamlファイルと同様）
    match = re.search(r"\w+/([\w-]+)\.yaml", args.config)
    if match:
        args.folder = match.group(1)
        print(args.folder)

    # 現在の時間を取得
    if args.now_str is None:
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.now_str = now_str
    print(f"now_str: {args.now_str}")

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
