import torch
import torch.nn as nn
import torchvision.models as models

class EthnicityModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, freeze_backbone=False):
        super(EthnicityModel, self).__init__()

        # Load pretrained ResNet50
        self.base = models.resnet50(pretrained=True)

        # Optional: Freeze backbone
        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        # Remove the original FC layer
        self.feature_extractor = nn.Sequential(*list(self.base.children())[:-1])  # (B, 2048, 1, 1)
        in_features = self.base.fc.in_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 2048)
        x = self.classifier(x)
        return x
