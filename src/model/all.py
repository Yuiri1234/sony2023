from collections import OrderedDict

# from torchvision.models.detection.rpn import AnchorGenerator
# from linformer import Linformer
# from vit_pytorch.efficient import ViT
import timm
import torch.nn.functional as F

# import torchvision
import torchvision.models as models
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CNNModel(nn.Module):
    def __init__(self, num_classes=18):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1
        )  # 64はbatch_sizeのことで良いと思う
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # フラット化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        out["logits"] = x.squeeze(1)
        if targets is not None:
            out["loss"] = self.compute_loss(out["logits"], targets.long())
        return out

    def compute_loss(self, preds, targets):
        return self.criteria(preds, targets)


# class CustomViTModel(nn.Module):
#     def __init__(self,
#                  dim=128,
#                  seq_len=50, # 7x7 patches + 1 cls-token
#                  depth=12,
#                  heads=8,
#                  k=64,
#                  image_size=224,
#                  patch_size=32,
#                  num_classes=18
#         ):
#         super(CustomViTModel, self).__init__()
#         efficient_transformer = Linformer(
#             dim=dim,
#             seq_len=seq_len,
#             depth=depth,
#             heads=heads,
#             k=k
#         )
#         self.model = ViT(
#             dim=dim,
#             image_size=image_size,
#             patch_size=patch_size,
#             num_classes=num_classes,
#             transformer=efficient_transformer,
#             channels=1,
#         )
#         self.criteria = nn.CrossEntropyLoss()

#     def forward(self, x, targets=None):
#         out = OrderedDict()
#         x = x.unsqueeze(1)
#         x = self.model(x)

#         out["logits"] = x
#         if targets is not None:
#             out["loss"] = self.compute_loss(out["logits"], targets.long())
#         return out

#     def compute_loss(self, preds, targets):
#         return self.criteria(preds, targets)


class CustomViTModel(nn.Module):
    def __init__(self, num_classes=18):
        super(CustomViTModel, self).__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=num_classes
        )
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = transforms.functional.resize(x, (224, 224), antialias=False)
        x = self.model(x)

        out["logits"] = x
        if targets is not None:
            out["loss"] = self.compute_loss(out["logits"], targets.long())
        return out

    def compute_loss(self, preds, targets):
        return self.criteria(preds, targets)


class CustomResNetModel(nn.Module):
    def __init__(self, model=18, pretrained=False, num_classes=18):
        super(CustomResNetModel, self).__init__()

        # Use a pre-trained ResNet model
        if model == 18:
            if pretrained:
                self.resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.resnet_model = models.resnet18()
        elif model == 34:
            if pretrained:
                self.resnet_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            else:
                self.resnet_model = models.resnet34()
        elif model == 50:
            if pretrained:
                self.resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.resnet_model = models.resnet50()

        # Modify the classifier layer to match the number of classes
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, num_classes)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)

        # Forward pass through ResNet
        x = self.resnet_model(x)

        out["logits"] = x
        if targets is not None:
            out["loss"] = self.compute_loss(out["logits"], targets.long())
        return out

    def compute_loss(self, preds, targets):
        return self.criteria(preds, targets)


class CustomMobileNetV2Model(nn.Module):
    def __init__(self, num_classes=18):
        super(CustomMobileNetV2Model, self).__init__()

        self.model = models.mobilenet_v2()

        # Modify the classifier layer to match the number of classes
        in_features = self.model.classifier[1].in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)

        # Forward pass through ResNet
        x = self.model(x)

        out["logits"] = x
        if targets is not None:
            out["loss"] = self.compute_loss(out["logits"], targets.long())
        return out

    def compute_loss(self, preds, targets):
        return self.criteria(preds, targets)


# class CustomFasterRCNN(nn.Module):
#     def __init__(self, num_classes=7):
#         super(CustomFasterRCNN, self).__init__()
#         backbone = torchvision.models.mobilenet_v2().features
#         backbone.out_channels = 1280
#         anchor_generator = AnchorGenerator(sizes=((10,),), aspect_ratios=((1.0,),))
#         self.frcnn = FasterRCNN(
#             backbone,
#             num_classes=num_classes,
#             rpn_anchor_generator=anchor_generator,
#         )

#     def forward(self, x, targets=None):
#         x = x.unsqueeze(1)
#         x = x.repeat(1, 3, 1, 1)
#         if targets is not None:
#             out = self.frcnn(x, targets)
#         else:
#             out = self.frcnn(x)
#         return out


class CustomFasterRCNN1(nn.Module):
    def __init__(self, model=34, pretrained=False, num_classes=7):
        super(CustomFasterRCNN1, self).__init__()
        if model == 18:
            if pretrained:
                backbone = resnet_fpn_backbone(
                    backbone_name="resnet18", weights=ResNet18_Weights.DEFAULT
                )
            else:
                backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
        elif model == 34:
            if pretrained:
                backbone = resnet_fpn_backbone(
                    backbone_name="resnet34", weights=ResNet34_Weights.DEFAULT
                )
            else:
                backbone = resnet_fpn_backbone(backbone_name="resnet34", weights=None)
        elif model == 50:
            if pretrained:
                backbone = resnet_fpn_backbone(
                    backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT
                )
            else:
                backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None)
        self.frcnn = FasterRCNN(
            backbone,
            num_classes=num_classes,
        )

    def forward(self, x, targets=None):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = transforms.functional.resize(x, (224, 224), antialias=False)
        if targets is not None:
            out = self.frcnn(x, targets)
        else:
            out = self.frcnn(x)
        return out


class CustomFasterRCNN2(nn.Module):
    def __init__(self, pretrained=False, num_classes=7):
        super(CustomFasterRCNN2, self).__init__()
        if pretrained:
            self.frcnn = fasterrcnn_resnet50_fpn(
                weights_backbone=ResNet50_Weights.DEFAULT
            )
        else:
            self.frcnn = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.frcnn.roi_heads.box_predictor.cls_score.in_features
        self.frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, targets=None):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = transforms.functional.resize(x, (224, 224), antialias=False)
        if targets is not None:
            out = self.frcnn(x, targets)
        else:
            out = self.frcnn(x)
        return out
