import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(".")
from configs.config import CONF  # noqa: E402


def add_random_noise(data, composite=False):
    if composite:
        data = data.flatten()
    # 正規分布からランダムな値を20から200個選ぶ
    num_values = np.random.randint(20, 200)
    values = np.random.normal(loc=np.random.randint(20, 120), scale=30, size=num_values)
    values[values < 0] = np.random.randint(1, 50, size=np.sum(values < 0))

    # ランダムな位置に値を追加
    indices = np.random.choice(range(20 * 20), size=num_values, replace=False)
    data[indices] = values

    # 0~255に収める
    data[data < 0] = 0
    data[data > 255] = 255
    if composite:
        data = data.reshape(20, 20)
    return data


def denoise(X):
    X_denoise = []
    for x in X:
        neiborhood = np.array([[1, 1]], np.uint8)
        x_erode = cv2.erode(x, neiborhood, iterations=1)
        x_dilate = cv2.dilate(x_erode, neiborhood, iterations=1)
        X_denoise.append(x_dilate)
    return np.array(X_denoise)


def create_data_dict(id, boxes, labels, area):
    return {
        "image_id": id,
        "boxes": boxes,
        "labels": labels,
        "area": area,
        "iscrowd": [0 for _ in range(len(boxes))],
    }


def create_one_dice_data(X, y, id, noise=False):
    label = np.random.randint(1, 7)
    indices = np.where(y == label)[0]
    dice_index = np.random.choice(indices)

    data = np.zeros(shape=(20, 20))
    if noise:
        data = add_random_noise(data, composite=True)
    pos_x = np.random.randint(0, 10)
    pos_y = np.random.randint(0, 10)
    data[pos_x : pos_x + 10, pos_y : pos_y + 10] = X[dice_index]
    data_dict = create_data_dict(
        id, [[pos_x, pos_y, pos_x + 10, pos_y + 10]], [label], [10]
    )

    return data, label, data_dict


def create_two_dices_data(X, y, id, noise=False):
    pattern = id % 2
    dice_pattern = np.random.randint(1, 7, size=2)
    label = dice_pattern.sum()
    dice_index = []
    for i in dice_pattern:
        indices = np.where(y == i)[0]
        dice_index.append(np.random.choice(indices))

    data = np.zeros(shape=(20, 20))
    if noise:
        data = add_random_noise(data, composite=True)
    pos_1 = np.random.randint(0, 10)
    pos_2 = np.random.randint(0, 10)
    boxes = []
    if pattern == 0:
        data[0:10, pos_1 : pos_1 + 10] = X[dice_index[0]]
        data[10:20, pos_2 : pos_2 + 10] = X[dice_index[1]]
        boxes.append([0, pos_1, 10, pos_1 + 10])
        boxes.append([10, pos_2, 20, pos_2 + 10])
    elif pattern == 1:
        data[pos_1 : pos_1 + 10, 0:10] = X[dice_index[0]]
        data[pos_2 : pos_2 + 10, 10:20] = X[dice_index[1]]
        boxes.append([pos_1, 0, pos_1 + 10, 10])
        boxes.append([pos_2, 10, pos_2 + 10, 20])
    area = [100 for _ in range(2)]
    data_dict = create_data_dict(id, boxes, dice_pattern, area)

    return data, label, data_dict


def create_three_dices_data(X, y, id, noise=False):
    pattern = id % 4
    dice_pattern = np.random.randint(1, 7, size=3)
    label = dice_pattern.sum()
    dice_index = []
    for i in dice_pattern:
        indices = np.where(y == i)[0]
        dice_index.append(np.random.choice(indices))

    data = np.zeros(shape=(20, 20))
    if noise:
        data = add_random_noise(data, composite=True)
    pos = np.random.randint(0, 10)

    boxes = []
    if pattern == 0:
        data[0:10, 0:10] = X[dice_index[0]]
        data[0:10, 10:20] = X[dice_index[1]]
        data[10:20, pos : pos + 10] = X[dice_index[2]]
        boxes.append([0, 0, 10, 10])
        boxes.append([0, 10, 10, 20])
        boxes.append([10, pos, 20, pos + 10])
    elif pattern == 1:
        data[0:10, pos : pos + 10] = X[dice_index[0]]
        data[10:20, 0:10] = X[dice_index[1]]
        data[10:20, 10:20] = X[dice_index[2]]
        boxes.append([0, pos, 10, pos + 10])
        boxes.append([10, 0, 20, 10])
        boxes.append([10, 10, 20, 20])
    elif pattern == 2:
        data[0:10, 0:10] = X[dice_index[0]]
        data[10:20, 0:10] = X[dice_index[1]]
        data[pos : pos + 10, 10:20] = X[dice_index[2]]
        boxes.append([0, 0, 10, 10])
        boxes.append([10, 0, 20, 10])
        boxes.append([pos, 10, pos + 10, 20])
    elif pattern == 3:
        data[pos : pos + 10, 0:10] = X[dice_index[0]]
        data[0:10, 10:20] = X[dice_index[1]]
        data[10:20, 10:20] = X[dice_index[2]]
        boxes.append([pos, 0, pos + 10, 10])
        boxes.append([0, 10, 10, 20])
        boxes.append([10, 10, 20, 20])
    area = [100 for _ in range(3)]
    data_dict = create_data_dict(id, boxes, dice_pattern, area)

    return data, label, data_dict


def preprocessing_train(cfg):
    data = np.load(os.path.join(CONF.PATH.DATASET, "all", "X_train.npy"))
    labels = np.load(os.path.join(CONF.PATH.DATASET, "all", "y_train.npy"))
    print(f"X_train.npy: {data.shape}, y_train.npy: {labels.shape}")
    if cfg.three_dices:
        print("create three dices data")
        cfg.data = f"{cfg.data}+three"
        X_one = np.load(
            os.path.join(CONF.PATH.DATASET, "one_dice_default", "one_dice.npy")
        )
        y_one = np.load(
            os.path.join(CONF.PATH.DATASET, "one_dice_default", "one_dice_labels.npy")
        )
        print(f"one_dice.npy: {X_one.shape}, one_dice_labels.npy: {y_one.shape}")
        X_three, y_three = [], []
        for i in tqdm(range(100000)):
            X_th, y_th, _ = create_three_dices_data(X_one, y_one, i)
            X_three.append(X_th)
            y_three.append(y_th)
        X_three = np.array(X_three)
        X_three = X_three.reshape(-1, 400)
        y_three = np.array(y_three)
        print(f"X_three: {X_three.shape}, y_three: {y_three.shape}")
        data = np.concatenate([data, X_three], axis=0)
        labels = np.concatenate([labels, y_three])
    if cfg.composite:
        print("create comp dices data")
        noise_flag = True if cfg.data == "noise" else False
        if cfg.num_data == 300000:
            cfg.data = f"{cfg.data}+comp"
        else:
            cfg.data = f"{cfg.data}+comp_{cfg.num_data}"

        X_one = np.load(
            os.path.join(CONF.PATH.DATASET, "one_dice_default", "one_dice.npy")
        )
        y_one = np.load(
            os.path.join(CONF.PATH.DATASET, "one_dice_default", "one_dice_labels.npy")
        )
        print(f"one_dice.npy: {X_one.shape}, one_dice_labels.npy: {y_one.shape}")

        X_data, y_data, data_dicts = [], [], []
        for i in tqdm(range(cfg.num_data)):
            if i % 3 == 0:
                # 1 dice
                data, label, data_dict = create_one_dice_data(
                    X_one, y_one, i, noise_flag
                )
                X_data.append(data)
                y_data.append(label)
                data_dicts.append(data_dict)
            elif i % 3 == 1:
                # 2 dices
                data, label, data_dict = create_two_dices_data(
                    X_one, y_one, i, noise_flag
                )
                X_data.append(data)
                y_data.append(label)
                data_dicts.append(data_dict)
            elif i % 3 == 2:
                # 3 dices
                data, label, data_dict = create_three_dices_data(
                    X_one, y_one, i, noise_flag
                )
                X_data.append(data)
                y_data.append(label)
                data_dicts.append(data_dict)
        X_data = np.array(X_data)
        X_data = X_data.reshape(-1, 400)
        y_data = np.array(y_data)
        data_dicts_df = pd.DataFrame(data_dicts)
        print(f"X: {X_data.shape}, y: {y_data.shape}, df: {data_dicts_df.shape}")
        data = X_data
        labels = y_data
    else:
        if cfg.data == "noise":
            print("add random noise")
            data = np.where(data == 1, 0, data)
            data = np.array([add_random_noise(d) for d in data])

    indices = range(len(data))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train, X_val, X_test = data[idx_train], data[idx_val], data[idx_test]
    y_train, y_val, y_test = labels[idx_train], labels[idx_val], labels[idx_test]
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    if not os.path.exists(os.path.join(CONF.PATH.DATASET, cfg.data)):
        os.mkdir(os.path.join(CONF.PATH.DATASET, cfg.data))

    if cfg.composite:
        train_df, val_df, test_df = (
            data_dicts_df.iloc[idx_train],
            data_dicts_df.iloc[idx_val],
            data_dicts_df.iloc[idx_test],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df["image_id"] = range(len(train_df))
        val_df["image_id"] = range(len(val_df))
        test_df["image_id"] = range(len(test_df))
        data_dicts_df.to_json(
            os.path.join(CONF.PATH.DATASET, cfg.data, "all.json"),
            orient="records",
            indent=4,
        )
        train_df.to_json(
            os.path.join(CONF.PATH.DATASET, cfg.data, "train.json"),
            orient="records",
            indent=4,
        )
        val_df.to_json(
            os.path.join(CONF.PATH.DATASET, cfg.data, "val.json"),
            orient="records",
            indent=4,
        )
        test_df.to_json(
            os.path.join(CONF.PATH.DATASET, cfg.data, "test.json"),
            orient="records",
            indent=4,
        )

    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X.npy"), data)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y.npy"), labels)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_train_filtered.npy"), X_train)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_val_filtered.npy"), X_val)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_test_filtered.npy"), X_test)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_train_filtered.npy"), y_train)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_val_filtered.npy"), y_val)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_test_filtered.npy"), y_test)

    print("preprocessing done")


def preprocessing_test():
    X_test_original = np.load(os.path.join(CONF.PATH.DATASET, "all", "X_test.npy"))
    print(X_test_original.shape)
    X_test_original = X_test_original.reshape(-1, 20, 20)
    X_test_denoise = denoise(X_test_original)
    X_test_denoise = X_test_denoise.reshape(-1, 400)
    print(X_test_denoise.shape)

    np.save(
        os.path.join(CONF.PATH.DATASET, "all", "X_test_denoise.npy"), X_test_denoise
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="all", choices=["all", "noise"])
    parser.add_argument("--three_dices", action="store_true")
    parser.add_argument("--composite", action="store_true")
    parser.add_argument("--num_data", type=int, default=300000)
    parser.add_argument("--train-test", choices=["train", "test"], default="train")
    args = parser.parse_args()
    args = EasyDict(vars(args))

    if args.train_test == "train":
        preprocessing_train(args)
    elif args.train_test == "test":
        preprocessing_test()
