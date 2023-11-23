from collections import OrderedDict

import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class CNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # フラット化
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        out["logits"] = x.squeeze(1)
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
