#!/bin/bash

python src/preprocess/one_dice_preprocess.py
python src/preprocess/all_preprocess.py --data all
python src/preprocess/all_preprocess.py --data all --three_dices
python src/preprocess/all_preprocess.py --data all --composite
python src/preprocess/all_preprocess.py --data all --composite --num_data 1000000
python src/preprocess/all_preprocess.py --data noise
python src/preprocess/all_preprocess.py --data noise --three_dices
python src/preprocess/all_preprocess.py --data noise --composite
python src/preprocess/all_preprocess.py --train-test test