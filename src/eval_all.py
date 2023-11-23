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
from dataset.all import CustomDataset, DetectionDataset  # noqa: E402
from model.all import (  # noqa: E402
    CNNModel,
    CustomFasterRCNN1,
    CustomFasterRCNN2,
    CustomMobileNetV2Model,
    CustomResNetModel,
    CustomViTModel,
)
from utils import create_confusion_matrix, fix_seed, seed_worker  # noqa: E402


def main(cfg):
    g = fix_seed(cfg.seed)

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
    print(X_test.shape, y_test.shape)

    if not cfg.detection:
        test_dataset = CustomDataset(X_test, y_test)
    else:
        test_df = pd.read_json(
            os.path.join(CONF.PATH.DATASET, cfg.data_dir, "test.json")
        ).reset_index(drop=True)
        test_dataset = DetectionDataset(X_test, y_test, test_df)

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # モデルの読み込み
    if not cfg.detection:
        if cfg.model == "cnn":
            model = CNNModel()
        elif cfg.model == "vit":
            model = CustomViTModel()
        elif cfg.model == "resnet":
            model = CustomResNetModel(model=cfg.model_layer, pretrained=cfg.pretrained)
        elif cfg.model == "mobilenetv2":
            model = CustomMobileNetV2Model()
        else:
            raise NotImplementedError
    else:
        if cfg.model == "fasterrcnn1":
            model = CustomFasterRCNN1(model=cfg.model_layer, pretrained=cfg.pretrained)
        elif cfg.model == "fasterrcnn2":
            model = CustomFasterRCNN2(pretrained=cfg.pretrained)

    model.to(device)
    weights = torch.load(
        os.path.join(CONF.PATH.OUTPUT, cfg.folder, cfg.pretrained_model),
        map_location=torch.device(device),
    )
    model.load_state_dict(weights)
    model.eval()

    # 推論
    results_list = []
    correct = 0
    total = 0
    if not cfg.detection:
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
                cfg.folder,
                f"eval_result_{accuracy:.3f}.csv",
            )
        )
    else:
        for i, batch in enumerate(tqdm(dataloader)):
            if cfg.debug and i == 100:
                break
            image, answer, image_id, boxes, labels, areas, iscrowd = batch
            image = image.to(device)
            with torch.no_grad():
                output = model(image)[0]
                boxes = output["boxes"].cpu()
                labels = output["labels"].cpu()
                scores = output["scores"].cpu()
            boxes = boxes[scores >= 0.5].long()
            labels = labels[scores >= 0.5].long()
            pred = sum(labels.tolist())
            # print(pred, answer.item())
            rate = 20 / 224
            boxes = boxes * rate
            result_dict = {
                "target": answer.item(),
                "pred": pred,
                "image_id": image_id.item(),
                "boxes": boxes.tolist(),
                "labels": labels.tolist(),
            }
            results_list.append(result_dict)

            total += answer.size(0)
            correct += (pred == answer).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy:.3f}")
        results_df = pd.DataFrame(results_list)
        results_df.to_json(
            os.path.join(
                CONF.PATH.OUTPUT,
                cfg.folder,
                f"eval_result_{accuracy:.3f}.json",
            ),
            orient="records",
            indent=4,
        )

    if not cfg.no_plots:
        # 混同行列の作成
        create_confusion_matrix(
            results_df["target"],
            results_df["pred"],
            title=(f"Confusion Matrix (Acc: {accuracy:.3f})"),
            save_path=os.path.join(
                CONF.PATH.OUTPUT,
                cfg.folder,
                "confusion_matrix.png",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder", type=str, default="20231110-220551_all_default"
    )
    parser.add_argument("-p", "--pretrained_model", type=str, default="model.pth")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--no-plots", action="store_true", help="no plots")
    parser.add_argument("--debug", action="store_true", help="only evaluate 100 images")
    args = parser.parse_args()
    args = EasyDict(vars(args))

    with open(os.path.join(CONF.PATH.OUTPUT, args.folder, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    cfg.update(args)
    print(cfg)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(cfg)
