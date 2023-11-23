import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(".")
from configs.config import CONF  # noqa: E402


def get_bounding_box(data, i, wh_max=10):
    image = data[i]
    image = image.reshape(20, 20)

    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]

    result = np.zeros((wh_max, wh_max), dtype=np.uint8)
    flag = 0

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
        if (w <= wh_max) and (h <= wh_max):
            flag = 1
            w_over = wh_max - w
            h_over = wh_max - h
            # はみ出しに対する処理
            # 0以下にならないようにする
            if w_over % 2 == 0:
                x = max(0, x - w_over // 2)
            else:
                x = max(0, x - w_over // 2 - 1)
            if h_over % 2 == 0:
                y = max(0, y - h_over // 2)
            else:
                y = max(0, y - h_over // 2 - 1)
            # 20を超えないようにする
            if x + wh_max > 20:
                x = 20 - wh_max
            if y + wh_max > 20:
                y = 20 - wh_max
            result = image[y : y + wh_max, x : x + wh_max]

    return result, flag


def get_dices(X, wh_max=10):
    results, flags = [], []
    for j in range(len(X)):
        result, flag = get_bounding_box(X, j, wh_max)
        results.append(result)
        flags.append(flag)
    results = np.array(results)
    flags = np.array(flags)
    return results, flags


def get_one_dice(X, y, flags):
    X_filtered = X[flags == 1]
    y_filtered = y[flags == 1]
    return X_filtered, y_filtered


def preprocessing(cfg):
    data = np.load(os.path.join(CONF.PATH.DATASET, "all", "X_train.npy"))
    labels = np.load(os.path.join(CONF.PATH.DATASET, "all", "y_train.npy"))

    data, flags = get_dices(data)
    data, labels = get_one_dice(data, labels, flags)
    print(data.shape, labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    if not os.path.exists(os.path.join(CONF.PATH.DATASET, cfg.data)):
        os.mkdir(os.path.join(CONF.PATH.DATASET, cfg.data))

    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "one_dice.npy"), data)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "one_dice_labels.npy"), labels)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_train_filtered.npy"), X_train)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_val_filtered.npy"), X_val)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "X_test_filtered.npy"), X_test)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_train_filtered.npy"), y_train)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_val_filtered.npy"), y_val)
    np.save(os.path.join(CONF.PATH.DATASET, cfg.data, "y_test_filtered.npy"), y_test)

    print("preprocessing done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="one_dice_default")
    args = parser.parse_args()
    preprocessing(args)
