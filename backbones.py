import torch.nn as nn
import torchvision.models as models

# =============================================================================
# Backbones: Adapted for 1-channel event count maps & feature resolution
# =============================================================================

class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)

        # 1-channel input adaptation for event maps
        old_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        self.features = mobilenet.features

        # Apply dilation to maintain spatial resolution for dense prediction
        for i in range(14, 19):
            for m in self.features[i].modules():
                if isinstance(m, nn.Conv2d):
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                    if m.kernel_size == (3, 3):
                        m.dilation = (2, 2)
                        m.padding = (2, 2)

    def forward(self, x):
        return self.features(x)  # Output channels: 1280


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        resnet = models.resnet50(weights=None)

        # 1-channel input adaptation for event maps
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Apply dilation to layer4 to maintain spatial resolution
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # Output channels: 2048