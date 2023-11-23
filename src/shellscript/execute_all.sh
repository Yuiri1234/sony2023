#!/bin/bash

now_str=$(date +"%Y%m%d-%H%M%S")
config="${1:-all_default_resnet}"
gpu="${2:-0}"

# 学習コマンド
python src/train_all.py -c "configs/${config}.yaml" --now_str "${now_str}" --gpu "${gpu}"

# 評価コマンド
python src/eval_all.py -f "${now_str}_${config}" --gpu "${gpu}"

# テストデータの予測
python src/predict_all.py -f "${now_str}_${config}" --gpu "${gpu}"
