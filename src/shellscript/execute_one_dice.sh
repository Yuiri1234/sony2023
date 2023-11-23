#!/bin/bash

now_str=$(date +"%Y%m%d-%H%M%S")
config="${1:-one_dice_resnet}"
gpu="${2:-0}"

# 学習コマンド
python src/train_one_dice.py -c "configs/${config}.yaml" --now_str "${now_str}" --gpu "${gpu}"

# 評価コマンド
python src/eval_one_dice.py -f "${now_str}_${config}" --gpu "${gpu}"
