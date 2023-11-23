import argparse
import os
import sys

import numpy as np
import torch.optim
import torch.utils.data
import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import fix_seed

sys.path.append(".")
from configs.config import CONF  # noqa: E402
from dataset.all import CustomDataset  # noqa: E402
from model.all import CNNModel, CustomResNetModel, CustomViTModel  # noqa: E402
from utils import seed_worker  # noqa: E402


class AdversarialValidation:
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
        train = np.load(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "X_test_filtered.npy")
        )
        test = np.load(os.path.join(CONF.PATH.DATASET, "all", "X_test.npy"))
        print(train.shape, test.shape)

        X = np.concatenate([train, test], axis=0)
        y = np.concatenate([np.zeros(train.shape[0]), np.ones(test.shape[0])], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.train_dataset = CustomDataset(X_train, y_train, adverval=True)
        self.test_dataset = CustomDataset(X_test, y_test, adverval=True)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,  # メモリのページングをしないように設定
            worker_init_fn=seed_worker,  # シード固定
            generator=generator,  # シード固定
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        # モデルの読み込み
        if cfg.model == "cnn":
            self.model = CNNModel()
        elif cfg.model == "vit":
            self.model = CustomViTModel(num_classes=2)
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

    def train(self):
        for epoch in tqdm(range(self.cfg.epochs)):
            self.train_one_epoch(epoch)

    def eval(self):
        results_list = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        self.model.eval()
        for X, y in tqdm(self.test_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                output = self.model(X, y)
                logits = torch.sigmoid(output["logits"]).cpu()
            _, pred = torch.max(logits, 1)

            result_dict = {
                "target": y.item(),
                "pred": pred,
            }
            results_list.append(result_dict)

            # tp, tn, fp, fnの計算
            for p, t in zip(pred, y):
                if t.item() == 1 and p == 1:
                    tp += 1
                elif t.item() == 0 and p == 0:
                    tn += 1
                elif t.item() == 0 and p == 1:
                    fp += 1
                elif t.item() == 1 and p == 0:
                    fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        recall = tp / (tp + fn)
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        result_dict = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        print(result_dict)


def main(args):
    g = fix_seed(args.seed)
    cfg = EasyDict(yaml.load(open(args.config), yaml.SafeLoader))
    cfg.update(args)

    adverval = AdversarialValidation(cfg, generator=g)
    adverval.train()
    adverval.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/all_default_cnn.yaml",
        help="config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    args = parser.parse_args()
    args = EasyDict(vars(args))

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
