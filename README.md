# SONY2023

- このプロジェクトはSIGNATEの[ソニーグループ合同 データ分析コンペティション](https://signate.jp/courses/OJXBVN6v3M9RYvdZ)用のレポジトリである。
- 1つ〜3つのサイコロの画像のバイナリデータから、目の合計（18種）を分類するモデルを作成していただきます。
- 目が最大3つの特殊なサイコロを使用しており、目の数え方の定義も異なるため、黒い点の数を数えるだけでは予測はできません。

## 必要要件
### Docker
- Docker version 20.10.17
- Docker Compose version v2.17.2

### 環境
- Ubuntu 22.04.2 LTS
- Python 3.10.12

### 主要ライブラリ
```
easydict
pyyaml
pandas
transformers
ultralytics
opencv-python-headless
timm
scikit-learn
```

## ディレクトリ構造

以下はこのプロジェクトのディレクトリ構造です．

```
├── configs
│   ├── config.py
│   ├── all_default_cnn.yaml
│   ├── ...
├── datasets
│   └── all
│       ├── X_train.npy
│       ├── X_test.npy
│       └── y_train.npy
├── docker-compose.yml
├── Dockerfile
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    ├── adversarial_validation.py
    ├── dataset
    │   ├── all.py
    │   └── one_dice.py
    ├── eval_all.py
    ├── eval_one_dice.py
    ├── model
    │   ├── all.py
    │   └── one_dice.py
    ├── predict_all.py
    ├── preprocess
    │   ├── all_preprocess.py
    │   └── one_dice_preprocess.py
    ├── shellscript
    │   ├── create_dataset.sh
    │   ├── execute_all.sh
    │   └── execute_one_dice.sh
    ├── train_all.py
    ├── train_one_dice.py
    └── utils.py
```

## ファイル説明

- `configs/`: 設定ファイルやAPIキーなどの設定関連のファイルを格納
- `docker-compose.yml`: Dockerコンテナを管理するためのComposeファイル
- `Dockerfile`: DockerイメージをビルドするためのDockerfile
- `datasets`: データセットを格納するためのフォルダ
- `pyproject.toml`: pysenを実行するためのファイル
- `requirements.txt`: プロジェクトの依存関係を示すためのファイル
- `src/`: プロジェクトのソースコードを格納

## 実行方法
### 事前準備
- `configs/config.py`の`CONF.PATH.BASE`をプロジェクトのホームディレクトリのパスに設定
- フォルダ構造の`datasets`のように`all`下にデータを格納

### Docker run & execute
Dockerイメージのビルド
```shell
docker compose build
```
Docker Compose によりコンテナを立ち上げる
```shell
docker compose up -d
```
Docker Compose によりコンテナに入る
```shell
docker compose exec main bash
```

### 使い方
#### データセット作成
実験用のデータセットを準備する。前処理用のコードを実行
```shell
bash src/shellscript/create_dataset.sh
```

#### 実験
configファイル名を指定して実験を実施。※ `configs/all_default_cnn.yaml`を実行したい場合
```shell
bash src/shellscript/execute_all.sh all_default_cnn
```

### 結果
- 67位／184人（Accuracy: 0.9348367）
- [all-plus-three_default_resnet_34_np.yaml](./configs/all-plus-three_default_resnet_34_np.yaml)
- モデル概要
    - trainデータから1つしかサイコロが含まれないデータから10*10で切り抜く
    - そのデータを用いて3つのサイコロが含まれる画像を作成する
    - 合成したデータを利用して、ResNet34を学習
- 最も効果があった方法
    - 3つのサイコロが含まれる画像を作成する
- 他に試した方法
    - ガウシアンノイズを学習データに付与（精度低下）
    - テストデータからノイズを取り除く（精度低下）
    - 10*10を用いて1~3個含まれる画像データを作成し、そのデータを用いて物体検出手法（Faster R-CNN）を用いて学習（うまく学習できず検出できずに終了）→原因調査中
- 他にやるべき方法
    - 物体検出ではなく、サイコロの位置のみを特定するネットワークを作成し、それぞれに対して1つの画像のみで学習されたモデルで推論することで、その合計値を計算する