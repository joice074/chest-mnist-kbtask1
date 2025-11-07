import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121_Improved(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, pretrained=True, freeze_features=False):
        super().__init__()
        # Load pretrained DenseNet
        self.model = models.densenet121(pretrained=pretrained)
        
        # Ubah input agar bisa grayscale
        self.model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Ubah layer output ke jumlah kelas kamu
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Freeze feature extractor kalau perlu
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = DenseNet121_Improved(in_channels=1, num_classes=2, pretrained=True, freeze_features=False)
    x = torch.randn(16, 1, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
